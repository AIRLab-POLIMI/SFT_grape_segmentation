import os
import sys

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import inference_on_dataset

from modules.det2modules import *
from  data_prep.wgisd_utils import init_dataset


def main():

    #Init test set
    variety = ''  # grape variety
    dtest_name = '_%s' % variety
    test_annp = './data/wgisd_split_byvariety/test/annotations_%s.json' % variety
    test_imgp = './data/wgisd_split_byvariety/test/images/%s/' % variety
    init_dataset(dtest_name, test_annp, test_imgp)

    #Load model
    cfg = get_cfg()
    custom_cfg = 'Misc/mask_rcnn_R_50_SFT_3x_WGISD.yaml'  # custom config in our detectron2 fork
    cfg.merge_from_file(model_zoo.get_config_file(custom_cfg))
    cfg.OUTPUT_DIR = "./RGB_MaskRCNN_SFT_wgisd_output_%s" % variety

    cfg_test = cfg
    cfg_test.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    predictor = DefaultPredictor(cfg_test)
    # TODO add custom metrics
    evaluator = COCOEvaluator(dtest_name, ("bbox", "segm",), False, output_dir="./output_best_wgisd/",
                              use_fast_impl=False)
    test_mapper = Mapper(cfg_test, is_train=False, augmentations=[])
    test_loader = build_detection_test_loader(cfg_test, dtest_name, test_mapper)
    print(inference_on_dataset(predictor.model, test_loader, evaluator))

if __name__ == "__main__":
    main()
    sys.exit(0)
