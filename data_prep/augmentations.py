"""
Define custom augmentations that are provided in the albumentations lib but not natively in detectron2
"""
from detectron2.data.transforms import Transform
import albumentations as A
from albumentations.augmentations.transforms import GaussNoise
import detectron2.data.transforms as T

class GaussBlur(Transform):

    def __init__(self):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        transform = A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=0.33)
        augmented_image = transform(image=img)['image']
        return augmented_image

    def apply_coords(self, coords):
        # coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        # coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation)
        return segmentation

    def inverse(self):
        return GaussBlur()


class GaussianNoise(Transform):
    def __init__(self):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        # determine type of noise based on noise_type parameters
        transform = GaussNoise(var_limit=(10, 50), mean=0, per_channel=True, always_apply=False, p=0.33)
        augmented_image = transform(image=img)['image']
        return augmented_image

    def apply_coords(self, coords):
        # coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        # coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation)
        return segmentation

    def inverse(self):
        return AddGaussNoise()


class PixelDropout(Transform):
    def __init__(self, prob):
        super().__init__()
        self.prob = prob
        self._set_attributes(locals())

    def apply_image(self, img):
        transform = A.PixelDropout(dropout_prob=0.01, per_channel=False, p=self.prob)

        augmented_image = transform(image=img)['image']
        return augmented_image

    def apply_coords(self, coords):
        # coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        # coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation)
        return segmentation

    def inverse(self):
        return PixelDropout(self.prob)



def create_augmentations_train(cfg):
    augmentations = []
    _PROB_HIGH = 0.5
    _PROB_LOW = 0.33

    #augmentations.append(T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING, interp=Image.NEAREST))
    augmentations.append(T.RandomApply(T.RandomBrightness(0.75, 1.25), prob=_PROB_LOW))
    augmentations.append(T.RandomApply(T.RandomContrast(0.75, 1.25), prob=_PROB_LOW))
    augmentations.append(T.RandomApply(T.RandomSaturation(0.75, 1.25), prob=_PROB_LOW))
    augmentations.append(GaussBlur())
    augmentations.append(GaussianNoise())
    augmentations.append(PixelDropout(prob=0.33))
    augmentations.append(T.RandomFlip(prob = _PROB_HIGH, horizontal=True, vertical=False))
    return augmentations


def create_augmentations_test(cfg):
    augmentations = []
    return augmentations