import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--basepath', default='./data/vinepics22/',
                        help='Base path to data')
    parser.add_argument('--trainval_path', default='./data/vinepics21',
                        help='Path to train and val splits and annotations')

    parser.add_argument('--test_path', default='./data/vinepics22/vinepics22R',
                        help='Path to test splits and annotations')
    parser.add_argument('--var', default='red_globe', choices=['CDY', 'CFR', 'CSV', 'SVB', 'SYH','all',
                                                         'red_globe', 'white_ortrugo', 'cabernet_sauvignon'],
                        help='Grape variety')
    parser.add_argument('--model_cfg', default='./configs/scratch_mask_rcnn_R_50_FPN_9x_gn.yaml',
                        help='Path to detectron2 model config file - relative to detectron2 source code')

    parser.add_argument('--out_dir', default='./data/results',
                        help='output dir for results')
    parser.add_argument('--dataset', default='vinepics22', choices=['wgisd','vinepics21', 'vinepics22'])
    parser.add_argument('--mode', default='tunelast', help='Tag for training mode.')
    parser.add_argument('--best_val_AP', action="store_true", help='If true, selects best model on val AP50 for inference.'\
                                                                    'Otherwise (default), selects the best model based on val loss. ')
    parser.add_argument('--conf_thresh', default=0.9, type=float,
                        help='Confidence threshold for grape class on inference')
    parser.add_argument('--weights', default=None,
                        help='Path to model weights if fine-tuning is applied')
    return parser
