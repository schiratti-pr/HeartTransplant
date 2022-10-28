from monai.transforms import (Compose, RandRotated, RandGaussianNoise, RandFlip, RandAxisFlip, RandZoom,
                              RandRotate, OneOf, Rand2DElastic, Rand3DElastic)

basic_transforms2d = [
    RandZoom(prob=0.1, min_zoom=0.8, max_zoom=1),
    RandRotate(range_x=10.0, range_y=10.0, range_z=10.0, prob=0.2),
    Rand2DElastic(prob=0.1, spacing=(30, 30), magnitude_range=(5, 6))
]

basic_transforms3d = [
    RandZoom(prob=0.1, min_zoom=0.8, max_zoom=1),
    RandRotate(range_x=10.0, range_y=10.0, range_z=10.0, prob=0.2),
    Rand3DElastic(prob=0.1, sigma_range=[0, 1], magnitude_range=(5, 6))
]

AUGMENTATIONS = {
    'basic2d': Compose(basic_transforms2d),
    'basic3d': Compose(basic_transforms3d),
}
