from typing import Optional, Sequence, Type, Union
from tqdm import tqdm

import torch
from monai.networks.nets import SegResNet, SwinUNETR, UNet
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn, Tensor
from torch.optim import AdamW, Optimizer


class NLSTReconstructModule(LightningModule):

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

        self.loss = nn.MSELoss()

        self.pretrained_params = set(pretrained_params) if pretrained_params else set()
        for n, param in self.net.named_parameters():
            param.requires_grad = bool(n not in self.pretrained_params)
        self.learning_rate = lr
        self.optimizer = optimizer or AdamW


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

        # log values
        self.log('Train/Loss', loss)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        img, y = batch["data"], batch["label"]
        y = torch.tensor(y, dtype=torch.double)

        y_hat = self(img)
        sigmoid_pred = torch.nn.functional.sigmoid(y_hat)
        class_pred = torch.round(sigmoid_pred)

        loss = self.loss(y_hat, y)

        # metric
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

        return {'loss': loss}

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

        self.log("val_epoch/loss", mean_outputs['loss'].item())

    def training_epoch_end(self, outputs: list):
        """Aggregates data from each training step
        """
        mean_outputs = {}
        for k in outputs[0].keys():
            mean_outputs[k] = torch.stack([x[k] for x in outputs]).mean()
