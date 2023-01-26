import os
from modules.det2modules import Trainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import sys

from data_prep.wgisd_utils import init_dataset
from dataviz import visualize_loss_plot

from detectron2.modeling import build_model
from params import get_parser
from data_prep.cattolica_utils import select_dataset

def main():

    parser = get_parser()
    args_dict, unknown = parser.parse_known_args()

    #params
    variety = args_dict.var  # grape variety
    dtrain_name = args_dict.dataset + '_train_%s' % variety
    dval_name = args_dict.dataset + '_val_%s' % variety

    if args_dict.dataset== 'cattolica22':
        #select subset of data based on variety, viewpoint and defoliation
        subfolder = select_dataset(args_dict.var, args_dict.view, args_dict.defol)
        if subfolder is None:
            print("No dataset with required features found")
            return

        basep = os.path.join(args_dict.trainval_path, subfolder) #/path/to/vine_cvat_subset_rotated_split
        training_annp = os.path.join(basep, 'train/annotations.json')
        training_imgs = os.path.join(basep, 'train')
        val_annp = os.path.join(basep, 'val/annotations.json')
        val_imgs = os.path.join(basep, 'val')

    else:
        training_annp = os.path.join(args_dict.trainval_path, 'annotations.json')
        training_imgs = os.path.join(args_dict.trainval_path, 'images')
        val_annp = os.path.join(args_dict.trainval_path, 'val/annotations_%s.json' % variety)
        val_imgs = os.path.join(args_dict.trainval_path, 'val/%s/' % variety)

    _,_= init_dataset(dtrain_name, training_annp, training_imgs)
    _,_= init_dataset(dval_name,val_annp, val_imgs)

    #Load model config
    cfg = get_cfg()
    custom_cfg = args_dict.model_cfg #custom config in our detectron2 fork
    cfg.merge_from_file(model_zoo.get_config_file(custom_cfg))

    cfg.DATASETS.TRAIN = (dtrain_name,)
    if variety !='all': 
        cfg.DATASETS.TEST = (dval_name,)
    else:
        cfg.DATASETS.TEST = ()
    cfg.OUTPUT_DIR = args_dict.out_dir +"_%s_%s" % (args_dict.dataset, variety)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    if args_dict.train_mode =='tune':
        cfg.MODEL.WEIGHTS ='../data/models_ceruti_final/split_80/model_RGB.pth'
        #otherwise, model is trained without adding pre-trained weights

    trainer = Trainer(cfg)
 
    trainer.resume_or_load(resume=False) 
    #Uncomment to check which parameters will be tuned

    model = build_model(cfg)
    print("Tuned modules:")
    for name, p in model.named_parameters():
        if p.requires_grad: 
            print(name)
    total_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("No of parameters to update: %i" % total_params)
    trainer.train() #training starts here

    visualize_loss_plot(cfg.OUTPUT_DIR)


if __name__ == "__main__":
    main()
    sys.exit(0)


