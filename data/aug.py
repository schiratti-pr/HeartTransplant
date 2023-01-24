from monai.transforms import (Compose, RandRotated, RandGaussianNoise, RandZoomd, RandAffined, RandGaussianNoised,
                              RandRotated, OneOf, Rand2DElasticd, Rand3DElasticd, LoadImaged)

basic_transforms2d = [
    RandZoomd(keys=["image", "label"], prob=0.2, min_zoom=0.8, max_zoom=1),
    RandRotated(keys=["image", "label"], range_x=10.0, range_y=10.0, prob=0.2),
    Rand2DElasticd(keys=["image", "label"], prob=0.1, spacing=(30, 30), magnitude_range=(5, 6))
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
