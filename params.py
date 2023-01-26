import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--trainval_path', default='./data/wgisd_split_byvariety/trainval/output/',
                        help='Path to train and val splits and annotations')
    parser.add_argument('--test_path', default='./data/wgisd_split_byvariety/test/',
                        help='Path to test splits and annotations')
    parser.add_argument('--var', default='CSV', choices=['CDY', 'CFR', 'CSV', 'SVB', 'SYH','all'
                                                         'red_globe', 'white_ortrugo', 'cabernet_sauvignon'],
                        help='Grape variety')
    parser.add_argument('--model_cfg', default='Misc/mask_rcnn_R_50_SFT_3x_WGISD.yaml',
                        help='Path to detectron2 model config file - relative to detectron2 source code')
    parser.add_argument('--out_dir', default='../data/RGB_MaskRCNN_SFT_wgisd_output_',
                        help='output dir for results')
    parser.add_argument('--dataset', default='wgisd', choices=['wgisd','cattolica21', 'cattolica22'])
    parser.add_argument('--conf_thresh', default=0.9, type=float,
                        help='Confidence threshold for grape class on inference')
    parser.add_argument('--pred_path', default=None,
                        help='Path to pth file with model predictions')
    parser.add_argument('--train_mode', default='tune', choices=['tune','scratch'],
                        help='Whether to fine-tune or re-train from scratch')
    parser.add_argument('--view', default=45,
                        help='Degrees indicating camera viewpoint')
    parser.add_argument('--defol', default=0, choices=[0,1,2],
                        help='Levels of defoliation, where 0 is no defoliation')
    return parser
