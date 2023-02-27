from detectron2.engine import hooks
from fvcore.nn.precise_bn import get_bn_modules
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader
from detectron2.engine.hooks import HookBase
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import build_detection_test_loader
import detectron2.data.detection_utils as utils
from detectron2.data import DatasetMapper, detection_utils
import detectron2.utils.comm as comm

from data_prep.augmentations import *

import os
import numpy as np
import torch
import time
import datetime
import logging
import copy

patience = 30



class Mapper(DatasetMapper):

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format="BGR") #so I can apply saturation
        detection_utils.check_image_size(dataset_dict, image)
        image = image[:,:,::-1]
        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        image = image[:,:,::-1]
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2,0,1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict


class LossEvalHook(HookBase):
    def __init__(self, cfg, model, data_loader):
        self._model = model
        self._period = cfg.TEST.EVAL_PERIOD
        self._root = cfg.OUTPUT_DIR
        self._data_loader = data_loader
        self._min_mean_loss = 0.0
        self._bfirst = True

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        #log on Neptune
        self.trainer.neptune_run['metrics/total_val_loss'].append(mean_loss)
        self.trainer.storage.put_scalar('validation_loss', mean_loss, smoothing_hint=False)
        comm.synchronize()
        return mean_loss

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            mean_loss = self._do_loss_eval()
            if self._bfirst:
                self._min_mean_loss = mean_loss
                self._bfirst = False
            # -------- save best model according to metrics --------
            if mean_loss < self._min_mean_loss:
                self._min_mean_loss = mean_loss
                self.trainer.checkpointer.save('model_best_validationLoss')
                with open(os.path.join(self._root, 'bestiter.txt'), 'a+') as f:
                    f.write('min val loss: ' + str(mean_loss) + ' at iter: ' + str(self.trainer.iter) + '\n')

class EarlyStopping(HookBase):
    def __init__(self, cfg, patience):
        self._patience = patience
        self._period = cfg.TEST.EVAL_PERIOD
        self._count = 0  # count for patience
        self._bfirst = True
        self._current_best_iteration = 0
        self._current_best_validation_loss = 0
        self._metric = "validation_loss"

    def after_step(self):

        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter

        if is_final or (
                self._period > 0 and next_iter % self._period == 0):  # this should be called at each validation period after the loss has been already stored

            # now read the best loss and the current loss
            metric_tuple = self.trainer.storage.latest().get(self._metric)
            latest_metric, metric_iter = metric_tuple

            if self._bfirst:
                self._bfirst = False
                self._current_best_iteration, self._current_best_validation_loss = metric_iter, latest_metric

            else:
                if (latest_metric > self._current_best_validation_loss):
                    self._count = self._count + 1
                    if (self._count == self._patience):
                        print("Early stopping at iteration %d, because patience of %d reached" % (
                        self.trainer.iter, self._count))
                        self.trainer.iter == self.trainer.max_iter #telling the system to stop training

                else:
                    self._current_best_iteration, self._current_best_validation_loss = metric_iter, latest_metric
                    self._count = 0

            log_every_n_seconds(
                logging.INFO,
                "EARLYSTOPPING: current val_loss {} at iter {} || best val loss {} at iter {}|| patience: {}".format(
                    latest_metric, metric_iter, self._current_best_validation_loss, self._current_best_iteration,
                    self._count
                ),
                n=2,
            )


class Trainer(DefaultTrainer):

    def __init__(self, cfg, neptune_session):
        """
        Args:
            cfg (CfgNode):
            Added logging on Neptune
        """
        super().__init__()
        self.neptune_run = neptune_session

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
        ret.append(hooks.BestCheckpointer(cfg.TEST.EVAL_PERIOD, self.checkpointer, "bbox/AP50", "max",
                                          file_prefix="model_best_bbox"))
        ret.append(hooks.BestCheckpointer(cfg.TEST.EVAL_PERIOD, self.checkpointer, "segm/AP50", "max",
                                          file_prefix="model_best_segm"))

        mapper = Mapper(cfg, is_train=True, augmentations=[])
        ret.append(LossEvalHook(self.cfg, self.model,
                                build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0], mapper)))
        ret.append(hooks.BestCheckpointer(cfg.TEST.EVAL_PERIOD, self.checkpointer, "validation_loss", "min",
                                          file_prefix="model_valLoss"))
        
        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=39))

        # append our EarlyStopping Hook in order to stop computation if the patience is reached (patience is 25 * validation period )
        ret.append(EarlyStopping(self.cfg, patience))

        return ret

    @classmethod
    def build_train_loader(cls, cfg):
        custom_mapper = Mapper(cfg, is_train=True, augmentations=create_augmentations_train(cfg))
        return build_detection_train_loader(cfg, mapper=custom_mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        custom_mapper = Mapper(cfg, is_train=True, augmentations=[])
        return build_detection_test_loader(cfg, dataset_name, mapper=custom_mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "detectron2_coco_eval")
        return COCOEvaluator(dataset_name, ("bbox", "segm"), True, output_folder, use_fast_impl=False)
