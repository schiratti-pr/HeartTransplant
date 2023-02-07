from monai.transforms import (Compose, RandRotated, RandGaussianNoise, RandZoomd, RandAffined, RandGaussianNoised,
                              RandRotated, OneOf, Rand2DElasticd, Rand3DElasticd, LoadImaged, RandZoom, RandAffine)

basic_transforms2d = [
    RandZoomd(keys=["image", "label"], prob=0.5, min_zoom=0.9, max_zoom=1.2),
    RandAffined(keys=['image', 'label'], prob=0.3, translate_range=10),
    RandGaussianNoised(keys='image', prob=0.4)
]

basic_transforms3d = [
    RandZoomd(keys=["image", "label"], prob=0.5, min_zoom=0.9, max_zoom=1.2),
    RandAffined(keys=['image', 'label'], prob=0.3, translate_range=10),
    RandGaussianNoised(keys='image', prob=0.4),
]

AUGMENTATIONS = {
    'basic2d': Compose(basic_transforms2d),
    'basic3d': Compose(basic_transforms3d),
}
