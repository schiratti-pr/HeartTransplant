from monai.losses import DiceLoss, DiceFocalLoss, DiceCELoss
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

LOSSES = {
    'DiceLoss': DiceLoss,
    'DiceFocalLoss': DiceFocalLoss,
    'CrossEntropyLoss': CrossEntropyLoss,
    'BCEWithLogitsLoss': BCEWithLogitsLoss,
    'DiceCELoss': DiceCELoss
}
