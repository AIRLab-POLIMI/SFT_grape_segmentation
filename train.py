import os
from modules.det2modules import Trainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import sys

from data_prep.wgisd_utils import init_dataset
from dataviz import visualize_loss_plot

from params import get_parser


def main():

    parser = get_parser()
    args_dict, unknown = parser.parse_known_args()

    #params
    variety = args_dict.var  # grape variety
    dtrain_name = args_dict.dataset + '_train_%s' % variety
    dval_name = args_dict.dataset + '_val_%s'% variety

    training_annp = os.path.join(args_dict.trainval_path, 'train/annotations_%s.json' % variety)
    training_imgs = os.path.join(args_dict.trainval_path, 'train/%s/' % variety)
    val_annp = os.path.join(args_dict.trainval_path,'val/annotations_%s.json' % variety)
    val_imgs = os.path.join(args_dict.trainval_path,'val/%s/' % variety)

    init_dataset(dtrain_name,training_annp, training_imgs)
    init_dataset(dval_name,val_annp, val_imgs)

    #Load model config
    cfg = get_cfg()
    custom_cfg = args_dict.model_cfg #custom config in our detectron2 fork
    cfg.merge_from_file(model_zoo.get_config_file(custom_cfg))

    cfg.DATASETS.TRAIN = (dtrain_name,)
    cfg.DATASETS.TEST = (dval_name,)
    cfg.OUTPUT_DIR = args_dict.out_dir +"%s" % variety

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)

    
    #Uncomment to check which parameters will be tuned

    from detectron2.modeling import build_model
    model = build_model(cfg)

    cnt = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            cnt += 1
    print("%i parameters to be updated" % cnt)
    

    trainer.train() #training starts here

    visualize_loss_plot(cfg.OUTPUT_DIR)


if __name__ == "__main__":
    main()
    sys.exit(0)


