from typing import Optional, Sequence, Type, Union
from tqdm import tqdm

import torch
from monai.networks.nets import SegResNet, SwinUNETR, UNet
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn, Tensor
from torch.optim import AdamW, Optimizer
from torchmetrics import Dice

from models.loss import LOSSES


def binary_dice_coefficient(pred: torch.Tensor, gt: torch.Tensor,
                            thresh: float = 0.5, smooth: float = 1e-7) -> torch.Tensor:
    """
    computes the dice coefficient for a binary segmentation task

    Args:
        pred: predicted segmentation (of shape Nx(Dx)HxW)
        gt: target segmentation (of shape NxCx(Dx)HxW)
        thresh: segmentation threshold
        smooth: smoothing value to avoid division by zero

    Returns:
        torch.Tensor: dice score
    """

    assert pred.shape == gt.shape

    pred_bool = pred > thresh

    intersec = (pred_bool * gt).float()
    return 2 * intersec.sum() / (pred_bool.float().sum()
                                 + gt.float().sum() + smooth)


class NLSTTrainingModule(LightningModule):

    def __init__(
        self,
        net: Union[nn.Module, str],
        pretrained_params: Optional[Sequence[str]] = None,
        lr: float = 1e-3,
        optimizer: Optional[Type[Optimizer]] = None,
        loss: dict = None,
        spatial_dims: int = 3,
        num_in_channels=1,
        pretrained_weights=None,
        log_images=False
    ):
        super().__init__()
        self.log_images = log_images

        if net == 'SegResNet':
            self.name = net
            net = SegResNet(
                spatial_dims=spatial_dims,
                blocks_down=[1, 2, 2, 4],
                blocks_up=[1, 1, 1],
                init_filters=16,
                in_channels=num_in_channels,
                out_channels=1,
                dropout_prob=0.2,
            )
        elif net == 'SwinUNETR':
            net = SwinUNETR(
                img_size=512,
                in_channels=num_in_channels,
                out_channels=1,
                feature_size=48,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=0.0,
                spatial_dims=spatial_dims
            )
        elif net == 'UNet':
            net = UNet(
                spatial_dims=spatial_dims,
                in_channels=num_in_channels,
                out_channels=1,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )
        self.spatial_dims = spatial_dims
        if spatial_dims == 3:
            self.net = net.double()
        else:
            self.net = net

        if pretrained_weights:
            self.net.load_state_dict(torch.load(pretrained_weights))

        if loss.get('params'):
            self.loss = LOSSES[loss['name']](**loss['params'])
        else:
            if 'dice' in loss['name']:
                self.loss = LOSSES[loss['name']](sigmoid=True)
            else:
                self.loss = LOSSES[loss['name']]()

        self.pretrained_params = set(pretrained_params) if pretrained_params else set()
        for n, param in self.net.named_parameters():
            param.requires_grad = bool(n not in self.pretrained_params)
        self.learning_rate = lr
        self.optimizer = optimizer or AdamW

        self.train_dice = Dice()
        self.val_dice = Dice()

    def configure_optimizers(self):
        optimizer = self.optimizer(self.net.parameters(), lr=self.learning_rate)
        # scheduler = CosineAnnealingLR(optimizer, self.trainer.max_epochs, 0)
        return [optimizer] #, [scheduler]

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, y = batch["data"], batch["label"]
        y = torch.tensor(y, dtype=torch.double)

        y_hat = self(img)
        sigmoid_pred = torch.nn.functional.sigmoid(y_hat)
        class_pred = torch.round(sigmoid_pred)

        # Calculate losses
        loss = self.loss(y_hat, y)

        # calculate dice coefficient
        dice_coeff = binary_dice_coefficient(class_pred, y)

        # log values
        self.log('Train/DiceCoeff', dice_coeff)
        self.log('Train/Loss', loss)

        return {'loss': loss, 'dice': dice_coeff}

    def validation_step(self, batch, batch_idx):
        img, y = batch["data"], batch["label"]
        y = torch.tensor(y, dtype=torch.double)

        y_hat = self(img)
        sigmoid_pred = torch.nn.functional.sigmoid(y_hat)
        class_pred = torch.round(sigmoid_pred)

        loss = self.loss(y_hat, y)

        # metric
        dice_coeff = binary_dice_coefficient(class_pred, y)
        self.log('Val/DiceCoeff', dice_coeff)
        self.log('Val/Loss', loss)

        if batch_idx % 200:
            # Get tensorboard logger
            tb_logger = None
            for logger in self.trainer.loggers:
                if isinstance(logger, TensorBoardLogger):
                    tb_logger = logger.experiment
                    break
            if self.spatial_dims != 3 and self.log_images:
                viz_batch = (img, y, class_pred)
                for img_idx, (image, y_true, y_pred) in enumerate(zip(*viz_batch)):
                    tb_logger.add_image(f"Image/{batch_idx}_{img_idx}", image, 0)
                    tb_logger.add_image(f"GroundTruth/{batch_idx}_{img_idx}", y_true, 0)
                    tb_logger.add_image(f"Prediction/{batch_idx}_{img_idx}", y_pred, 0)
                    break

        return {'loss': loss, 'dice': dice_coeff}

    def validation_epoch_end(self, outputs: list):
        """Aggregates data from each validation step

        Args:
            outputs: the returned values from each validation step

        Returns:
            dict: the aggregated outputs
        """
        mean_outputs = {}
        for k in outputs[0].keys():
            mean_outputs[k] = torch.stack([x[k] for x in outputs]).mean()

        tqdm.write('Dice: \t%.3f' % mean_outputs['dice'].item())
        self.log("val_epoch/loss", mean_outputs['loss'].item())

    def training_epoch_end(self, outputs: list):
        """Aggregates data from each training step
        """
        mean_outputs = {}
        for k in outputs[0].keys():
            mean_outputs[k] = torch.stack([x[k] for x in outputs]).mean()

        tqdm.write('Dice Train: \t%.3f' % mean_outputs['dice'].item())
