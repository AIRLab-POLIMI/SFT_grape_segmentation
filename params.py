import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--basepath', default='../data/cattolica2022_labelled/',
                        help='Base path to data')
    parser.add_argument('--trainval_path', default='../data/red_globe_2021_07-27_09-06_train_val_test',
                        help='Path to train and val splits and annotations')
    parser.add_argument('--test_path', default='../data/red_globe_2021_07-27_09-06_train_val_test',
                        help='Path to test splits and annotations')
    parser.add_argument('--var', default='red_globe', choices=['CDY', 'CFR', 'CSV', 'SVB', 'SYH','all',
                                                         'red_globe', 'white_ortrugo', 'cabernet_sauvignon'],
                        help='Grape variety')
    parser.add_argument('--model_cfg', default='Misc/mask_rcnn_R_50_SFT_3x_cat22.yaml',
                        help='Path to detectron2 model config file - relative to detectron2 source code')
    parser.add_argument('--out_dir', default='../data/results',
                        help='output dir for results')
    parser.add_argument('--dataset', default='cattolica22', choices=['wgisd','cattolica21', 'cattolica22'])
    parser.add_argument('--mode', default='tunelast', help='Tag for training mode.')
    parser.add_argument('--best_val_AP', action="store_true", help='If true, selects best model on val AP50 for inference.'\
                                                                    'Otherwise (default), selects the best model based on val loss. ')
    parser.add_argument('--conf_thresh', default=0.9, type=float,
                        help='Confidence threshold for grape class on inference')
    parser.add_argument('--weights', default=None,
                        help='Path to model weights if fine-tuning is applied')
    return parser
