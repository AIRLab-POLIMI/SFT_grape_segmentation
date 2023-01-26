import os
import sys

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import inference_on_dataset
from detectron2.evaluation.coco_evaluation import COCOEvaluatorCustomized

from modules.det2modules import *
from  data_prep.wgisd_utils import init_dataset
from  data_prep.cattolica_utils import *

from params import get_parser

def main():

    parser = get_parser()
    args_dict, unknown = parser.parse_known_args()

    #Init test set
    variety = args_dict.var  # grape variety
    dtest_name = args_dict.dataset + '_test_%s' % variety
    annp = os.path.join(args_dict.trainval_path, 'annotations/instances_default.json')

    if args_dict.dataset== 'cattolica22':
        subfolder = select_dataset(args_dict.var, args_dict.view, args_dict.defol)
        if subfolder is None:
            print("No dataset with required features found")
            return

        basep = os.path.join(args_dict.trainval_path, subfolder) #/path/to/vine_cvat_subset_rotated (full, non-split sets)
        # prep annotations for target test set
        test_ann = subset_annotations(basep, annp)
        test_annp = os.path.join(basep,"annotations.json")
        with open(test_annp,'rb') as otf:
            json.dump(test_ann, otf)
        test_imgp = basep
    else:
        test_annp = os.path.join(args_dict.test_path, 'annotations.json')
        test_imgp = os.path.join(args_dict.test_path, 'images')
    _,_ = init_dataset(dtest_name, test_annp, test_imgp)

    #Load model
    cfg = get_cfg()
    custom_cfg = args_dict.model_cfg  # custom config in our detectron2 fork
    cfg.merge_from_file(model_zoo.get_config_file(custom_cfg))
    cfg.OUTPUT_DIR = args_dict.out_dir + "%s" % variety


    cfg_test = cfg
    cfg_test.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") #'../data/models_ceruti_final/split_80/model_RGB.pth' 
    cfg_test.DATASETS.TEST = (dtest_name,)
    cfg_test.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args_dict.conf_thresh

    #Eval on test set
    result_path = os.path.join(cfg.OUTPUT_DIR,'cthresh_%s' % str(args_dict.conf_thresh))
    if not os.path.isdir(result_path):
        os.mkdir(result_path) 
    predictor = DefaultPredictor(cfg_test)
    evaluator = COCOEvaluator(dtest_name, ("bbox", "segm",), False, output_dir=result_path,use_fast_impl=False)
    evaluator_cstm = COCOEvaluatorCustomized(dtest_name, ("bbox", "segm",), False, output_dir=result_path,use_fast_impl=False)
    test_mapper = Mapper(cfg_test, is_train=False, augmentations=[])
    test_loader = build_detection_test_loader(cfg_test, dtest_name, test_mapper)
    print(inference_on_dataset(predictor.model, test_loader, [evaluator,evaluator_cstm]))

if __name__ == "__main__":
    main()
    sys.exit(0)
