from typing import Any, Optional, Sequence, Tuple, Type, Union

import torch
from monai.networks.nets import SegResNet, SwinUNETR, UNet
from pytorch_lightning import Callback, LightningModule
from torch import nn, Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Dice

from models.loss import LOSSES


class NLSTTrainingModule(LightningModule):

    def __init__(
        self,
        net: Union[nn.Module, str],
        pretrained_params: Optional[Sequence[str]] = None,
        lr: float = 1e-3,
        optimizer: Optional[Type[Optimizer]] = None,
        loss: dict = None,
        spatial_dims: int = 3,
        num_in_channels=1
    ):
        super().__init__()

        if net == 'SegResNet':
            self.name = net
            net = SegResNet(
                spatial_dims=spatial_dims,
                blocks_down=[1, 2, 2, 4],
                blocks_up=[1, 1, 1],
                init_filters=16,
                in_channels=num_in_channels,
                out_channels=3,
                dropout_prob=0.2,
            )
        elif net == 'SwinUNETR':
            net = SwinUNETR(
                img_size=256,
                in_channels=num_in_channels,
                out_channels=3,
                feature_size=48,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=0.0,
                use_checkpoint=True,
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

        if spatial_dims == 3:
            self.net = net.double()
        else:
            self.net = net

        if loss.get('params'):
            self.loss = LOSSES[loss['name']](**loss['params'])
        else:
            self.loss = LOSSES[loss['name']]()

        self.pretrained_params = set(pretrained_params) if pretrained_params else set()
        for n, param in self.net.named_parameters():
            param.requires_grad = bool(n not in self.pretrained_params)
        self.learning_rate = lr
        self.optimizer = optimizer or AdamW

        self.train_dice = Dice()
        self.val_dice = Dice()

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, y = batch["data"], batch["label"]
        y_hat = self(img)
        y = torch.tensor(y, dtype=torch.long)
        loss = self.loss(y_hat, y)

        return {'loss': loss, 'preds': y_hat, 'target': y}

    def training_step_end(self, outputs):
        # update and log
        self.train_dice.update(outputs['preds'], outputs['target'])

    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_epoch/dice', self.train_dice.compute())
        self.train_dice.reset()

    def validation_step(self, batch, batch_idx):
        img, y = batch["data"], batch["label"]
        y_hat = self(img)
        y = torch.tensor(y, dtype=torch.long)
        loss = self.loss(y_hat, y)

        self.val_dice.update(y_hat, y)

        return loss

    def validation_epoch_end(self, outs):
        loss = torch.stack(outs).mean()

        self.log("val_epoch/loss", loss, sync_dist=True)
        self.log('val_epoch/dice', self.val_dice.compute(), sync_dist=True)

        self.val_dice.reset()

    def configure_optimizers(self):
        optimizer = self.optimizer(self.net.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, self.trainer.max_epochs, 0)
        return [optimizer], [scheduler]
