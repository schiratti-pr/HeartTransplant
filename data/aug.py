from monai.transforms import (Compose, RandRotated, RandGaussianNoise, RandFlip, RandAxisFlip, RandZoom, RandRotate, OneOf)

basic_transforms = [
    RandZoom(prob=0.1, min_zoom=0.8, max_zoom=1),
    RandRotate(range_x=10.0, range_y=10.0, range_z=10.0, prob=0.2),
]

AUGMENTATIONS = {
    'basic': Compose(basic_transforms),
}
