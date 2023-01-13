from detectron2.data.datasets import register_coco_instances
import os
from detectron2.data.datasets import load_coco_json
from detectron2.data import MetadataCatalog


def init_dataset(dataset_name, ann_path, img_path):

    metadata_dict = {'name' : dataset_name, 'thing_classes' : ['grape bunch'], 'thing_colors' : [(255, 0, 0)]}
    register_coco_instances(dataset_name, metadata_dict,
                            ann_path, img_path)
    training_dict = load_coco_json(ann_path, img_path, dataset_name=dataset_name)
    metadata = MetadataCatalog.get(dataset_name)


