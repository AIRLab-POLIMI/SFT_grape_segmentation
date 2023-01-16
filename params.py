import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--trainval_path', default='./data/wgisd_split_byvariety/trainval/output/',
                        help='Path to train and val splits and annotations')
    parser.add_argument('--test_path', default='./data/wgisd_split_byvariety/test/',
                        help='Path to test splits and annotations')
    parser.add_argument('--var', default='CSV', choices=['CDY', 'CFR', 'CSV', 'SVB', 'SYH'],
                        help='Grape variety')
    parser.add_argument('--model_cfg', default='Misc/mask_rcnn_R_50_SFT_3x_WGISD.yaml',
                        help='Path to detectron2 model config file - relative to detectron2 source code')
    parser.add_argument('--out_dir', default='./RGB_MaskRCNN_SFT_wgisd_output_',
                        help='output dir for results')
    parser.add_argument('--dataset', default='wgisd',)

    return parser