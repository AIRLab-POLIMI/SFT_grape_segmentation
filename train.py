import os
from modules.det2modules import Trainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import sys

from data_prep.wgisd_utils import init_dataset
from dataviz import visualize_loss_plot

def main():

    #params #TODO read through argparse and from params.py
    variety = ''  # grape variety
    dtrain_name = '_%s' % variety
    dval_name = '_%s'% variety

    training_annp = './data/wgisd_split_byvariety/trainval/output/train/annotations_%s.json' % variety
    training_imgs = './data/wgisd_split_byvariety/trainval/output/train/%s/' % variety
    val_annp = './data/wgisd_split_byvariety/trainval/output/val/annotations_%s.json' % variety
    val_imgs = './data/wgisd_split_byvariety/trainval/output/val/%s/' % variety

    init_dataset(dtrain_name,training_annp, training_imgs)

    #Load model config
    cfg = get_cfg()
    custom_cfg = 'Misc/mask_rcnn_R_50_SFT_3x_WGISD.yaml' #custom config in our detectron2 fork
    cfg.merge_from_file(model_zoo.get_config_file(custom_cfg))

    cfg.DATASETS.TRAIN = (dtrain_name,)
    cfg.DATASETS.TEST = (dval_name,)
    cfg.OUTPUT_DIR = "./RGB_MaskRCNN_SFT_wgisd_output_%s" % variety

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)

    """
    #Uncomment to check which parameters will be tuned

    from detectron2.modeling import build_model
    model = build_model(cfg)

    cnt = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            cnt += 1
    print("%i parameters to be updated" % cnt)
    """

    trainer.train() #training starts here

    visualize_loss_plot(cfg.OUTPUT_DIR)


if __name__ == "__main__":
    main()
    sys.exit(0)


