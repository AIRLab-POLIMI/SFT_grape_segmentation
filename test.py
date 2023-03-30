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

    if args_dict.dataset== 'cattolica22':
        test_annp = os.path.join(args_dict.test_path, 'annotations/annotations_%s.json' % args_dict.mode) #e.g., cattolica22A, cattolica22B, etc
        test_imgp = os.path.join(args_dict.test_path, args_dict.mode) #cattolica22A, 22B, etc...
        print(test_annp)
        print(test_imgp)

    elif args_dict.dataset== 'cattolica21':
        test_annp = os.path.join(args_dict.test_path, 'annotations/annotations_test.json')
        test_imgp = os.path.join(args_dict.test_path, 'test')

    else:
        test_annp = os.path.join(args_dict.test_path, 'annotations.json')
        test_imgp = os.path.join(args_dict.test_path, 'images')
    dres,metadata = init_dataset(dtest_name, test_annp, test_imgp, data=args_dict.dataset)
    #print(metadata)
    #print(dres)

    #Load model
    cfg = get_cfg()
    custom_cfg = args_dict.model_cfg  # custom config in our detectron2 fork
    cfg.merge_from_file(custom_cfg) #model_zoo.get_config_file(custom_cfg))
    cfg.OUTPUT_DIR = args_dict.out_dir
    #cfg.OUTPUT_DIR = os.path.join(args_dict.out_dir, custom_cfg.split('/')[-1].replace(".yaml",""))

    cfg_test = cfg
    if args_dict.best_val_AP:
        cfg_test.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_best_segm.pth")  #"/data/weights/wgisd_scratch_R50_bestval.pth"
    else: 
        cfg_test.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_valLoss.pth")   #"/data/weights/wgisd_scratch_R50_bestval.pth"

    cfg_test.DATASETS.TEST = (dtest_name,)
    cfg_test.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args_dict.conf_thresh

    #Eval on test set
    result_path = os.path.join(cfg.OUTPUT_DIR,args_dict.mode,'cthresh_%s' % str(args_dict.conf_thresh))
    if not os.path.isdir(result_path):
        os.makedirs(result_path) 
    predictor = DefaultPredictor(cfg_test)
    evaluator = COCOEvaluator(dtest_name, ("bbox", "segm",), False, output_dir=result_path,use_fast_impl=False)
    evaluator_cstm = COCOEvaluatorCustomized(dtest_name, ("bbox", "segm",), False, output_dir=result_path,use_fast_impl=False)
    test_mapper = Mapper(cfg_test, is_train=False, augmentations=[])
    test_loader = build_detection_test_loader(cfg_test, dtest_name, test_mapper)
    results = inference_on_dataset(predictor.model, test_loader, [evaluator,evaluator_cstm])
    print(results)


if __name__ == "__main__":
    main()
    sys.exit(0)
