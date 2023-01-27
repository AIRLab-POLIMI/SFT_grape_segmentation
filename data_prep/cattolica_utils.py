import json
import sys
import os
import splitfolders
from params import get_parser

#select dataset based on viewpoint, variety, defoliation

def select_dataset(var, view, defol):

    with open('./data/cattolica2022_dataindex.json') as ind:
        data_index = json.load(ind)['datasets']

    tgt_fol = None
    for data_node in data_index:
        if data_node['variety'] == var and data_node['viewpoint'] == view \
            and data_node['defoliation'] == defol:

            tgt_fol = data_node['bagname']
            break

    return tgt_fol

def split_trainval(in_path, tgt_path, r=0.8):

    os.makedirs(tgt_path,exist_ok=True)
    splitfolders.ratio(in_path, output= tgt_path, #os.path.join(tgt_path,'output'),
                       seed=1337, ratio=(r, (1.-r)))

def subset_annotations(tgt_dirname, jpath):

    with open(jpath) as inj:
        all_annotations = json.load(inj)
    ann_subset = {k: {} for k in all_annotations.keys()}
    ann_subset['images'] = []
    ann_subset['annotations'] = []

    imglist = os.listdir(tgt_dirname)

    for d_ in all_annotations['images']:
        img_name = d_["file_name"].split('/')[-1]
        if img_name in imglist:
            d_["file_name"] = img_name
            ann_subset['images'].append(d_)
            img_id = d_["id"]
            a_ = [ann for ann in all_annotations['annotations'] if ann['image_id'] == img_id][0]
            ann_subset['annotations'].append(a_)

    ann_subset['categories'] = all_annotations['categories']
    ann_subset['licenses'] = all_annotations['licenses']
    ann_subset['info'] = all_annotations['info']

    return ann_subset


def main():

    #If run directly as script, prepares cattolica data for training/val/test
    parser = get_parser()
    args_dict, unknown = parser.parse_known_args()

    subfolder = select_dataset("red_globe", 45, 1) #use as training/val set
    if subfolder is None:
        print("No dataset with required features found")
        return

    # split into train and val
    ipath = os.path.join(args_dict.trainval_path, 'images/vine_cvat_subset_rotated/')
    tpath = os.path.join(args_dict.trainval_path, 'images/vine_cvat_subset_rotated_split/') #/path/to/cattolica2022_labelled/
    annp = os.path.join(args_dict.trainval_path, 'annotations/instances_default.json')
    anntrain = os.path.join(tpath, 'train', subfolder)
    annval = os.path.join(tpath, 'val', subfolder)

    if not os.path.exists(tpath):
        split_trainval(ipath, tpath)

    train_ann = subset_annotations(anntrain,annp)
    val_ann = subset_annotations(annval,annp)

    with open(os.path.join(anntrain, 'annotations.json'), 'w') as of1, \
        open(os.path.join(annval, 'annotations.json'), 'w') as of2:
        json.dump(train_ann, of1)
        json.dump(val_ann, of2)


if __name__ == "__main__":
    main()
    sys.exit(0)