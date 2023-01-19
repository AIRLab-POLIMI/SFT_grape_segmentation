from detectron2.utils.visualizer import Visualizer, ColorMode
from params import get_parser
from  data_prep.wgisd_utils import init_dataset
import torch

import sys
import os
import cv2

def main():

    parser = get_parser()
    args_dict, unknown = parser.parse_known_args()

    # Init test set
    variety = args_dict.var  # grape variety
    dtest_name = '_%s' % variety
    if variety != 'all':
        test_annp = os.path.join(args_dict.test_path, 'annotations_%s.json' % variety)
        test_imgp = os.path.join(args_dict.test_path, 'images/%s/' % variety)
    else:
        test_annp = os.path.join(args_dict.test_path, 'annotations.json')
        test_imgp = os.path.join(args_dict.test_path, 'images')
    test_dict, test_metadata = init_dataset(dtest_name, test_annp, test_imgp)

    predictions = torch.load(args_dict.pred_path)  #path to output model predictions
    outp = os.path.join(args_dict.pred_path,'img_predictions')
    os.makedirs(outp,exist_ok=True)

    for d in test_dict:
        im = cv2.imread(d["file_name"])

        # draw instance predictions on rgb image and on depth image
        v = Visualizer(im[:, :, ::-1],
                       metadata=test_metadata,
                       scale=1.0,
                       instance_mode=ColorMode.SEGMENTATION
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )
        out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
        img1 = out.get_image()[:, :, ::-1]

        cv2.imwrite(os.path.join(outp,im), img1)
        break

if __name__ == "__main__":
    main()
    sys.exit(0)