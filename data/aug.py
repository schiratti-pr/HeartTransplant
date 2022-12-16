from monai.transforms import (Compose, RandRotated, RandGaussianNoise, RandZoomd,
                              RandRotated, OneOf, Rand2DElasticd, Rand3DElasticd, LoadImaged)

basic_transforms2d = [
    RandZoomd(keys=["image", "label"], prob=0.2, min_zoom=0.8, max_zoom=1),
    RandRotated(keys=["image", "label"], range_x=10.0, range_y=10.0, prob=0.2),
    Rand2DElasticd(keys=["image", "label"], prob=0.1, spacing=(30, 30), magnitude_range=(5, 6))
]

basic_transforms3d = [
    RandZoomd(keys=["image", "label"], prob=0.2, min_zoom=0.8, max_zoom=1),
    RandRotated(keys=["image", "label"], range_x=10.0, range_y=10.0, range_z=10.0, prob=0.2),
    Rand3DElasticd(keys=["image", "label"], prob=0.1, sigma_range=[0, 1], magnitude_range=(5, 6))
]

AUGMENTATIONS = {
    'basic2d': Compose(basic_transforms2d),
    'basic3d': Compose(basic_transforms3d),
}
