import os
from modules.det2modules import Trainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import sys
import logging

from data_prep.wgisd_utils import init_dataset
#from dataviz import visualize_loss_plot

from detectron2.modeling import build_model
from params import get_parser
from data_prep.cattolica_utils import select_dataset
from dataviz import load_json_arr

import neptune.new as neptune


def main():

    parser = get_parser()
    args_dict, unknown = parser.parse_known_args()

    #Logging on Neptune
    logger = logging.getLogger("detectron2")
    run = neptune.init_run(project='AIRLab/agri-robotics-grape-segmentation',
                           mode='async',  # use 'debug' to turn off logging, 'async' otherwise
                           name='%s_%s' % (args_dict.model_cfg.split("/")[-1], 'pre-train'),
                           tags=[args_dict.mode, args_dict.dataset, args_dict.var]) #, "pre-train"]) 

    #params
    variety = args_dict.var  # grape variety
    dtrain_name = args_dict.dataset + '_train_%s' % variety
    dval_name = args_dict.dataset + '_val_%s' % variety

    if args_dict.dataset== 'vinepics21':
        training_annp = os.path.join(args_dict.trainval_path, 'annotations/annotations_train.json')
        training_imgs = os.path.join(args_dict.trainval_path, 'train/')
        val_annp = os.path.join(args_dict.trainval_path, 'annotations/annotations_val.json')
        val_imgs = os.path.join(args_dict.trainval_path, 'val/')
    else: #wgisd
        training_annp = os.path.join(args_dict.trainval_path, 'training/annotations_split.json')
        #training_annp = os.path.join(args_dict.trainval_path, 'training/annotations.json')
        training_imgs = os.path.join(args_dict.trainval_path, 'training/images')
        val_annp = os.path.join(args_dict.trainval_path, 'val/annotations_split.json')
        val_imgs = os.path.join(args_dict.trainval_path, 'val/images/')

    _, _ = init_dataset(dtrain_name, training_annp, training_imgs, data=args_dict.dataset)
    _, _ = init_dataset(dval_name, val_annp, val_imgs, data=args_dict.dataset)

    #Load model config
    cfg = get_cfg()
    custom_cfg = args_dict.model_cfg #custom config in our detectron2 fork
    cfg.merge_from_file(custom_cfg)

    cfg.DATASETS.TRAIN = (dtrain_name,)
    cfg.DATASETS.TEST = (dval_name,)

    cfg.OUTPUT_DIR = os.path.join(args_dict.out_dir, custom_cfg.split('/')[-1].replace(".yaml",""))
    print("Results and models saved under %s" % cfg.OUTPUT_DIR)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # ------ NEPTUNE LOGGING ------

    # Log fixed parameters in Neptune
    PARAMS = {'dataset_train': cfg.DATASETS.TRAIN,
              'dataset_test': cfg.DATASETS.TEST,
              'dataloader_num_workers': cfg.DATALOADER.NUM_WORKERS,
              'freeze_at': cfg.MODEL.BACKBONE.FREEZE_AT,
              'batch_size_train': cfg.SOLVER.IMS_PER_BATCH,
              'max_iter': cfg.SOLVER.MAX_ITER,
              'base_lr': cfg.SOLVER.BASE_LR,
              'momentum': cfg.SOLVER.MOMENTUM,
              'weight_decay': cfg.SOLVER.WEIGHT_DECAY,
              'steps': cfg.SOLVER.STEPS,
              'eval_period': cfg.TEST.EVAL_PERIOD,
              'optimizer': 'SGD',
              'min_size_train': cfg.INPUT.MIN_SIZE_TRAIN,
              'min_size_test': cfg.INPUT.MIN_SIZE_TEST
              }

    # Pass parameters to the Neptune run object.
    run['cfg_parameters'] = PARAMS  # This will create a ‘parameters' directory containing the PARAMS dictionary

    trainer = Trainer(cfg,run)
 
    trainer.resume_or_load(resume=False) 
    #Uncomment to check which parameters will be tuned

    model = build_model(cfg)
    print("Tuned modules:")
    for name, p in model.named_parameters():
        if p.requires_grad: 
            print(name)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("No of parameters to update: %i" % total_params)
    trainer.train() #training starts here

    #Log metrics on Neptune
    #visualize_loss_plot(cfg.OUTPUT_DIR)


if __name__ == "__main__":
    main()
    sys.exit(0)


