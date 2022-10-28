import os
import argparse
from datetime import datetime
from omegaconf import OmegaConf
import collections

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from data.dataloader import NLST_2D_DataModule
from models import NLSTTrainingModule
from utils.create_2d_data import data_to_slices


def main():
    parser = argparse.ArgumentParser(description='Processing configuration for training')
    parser.add_argument('--config', type=str, help='path to config file', default='configs/config_2d.yaml')
    args = parser.parse_args()

    pl.seed_everything(int(os.environ.get('LOCAL_RANK', 0)))

    # Load configuration file
    config = OmegaConf.load(args.config)

    # TODO: add dictionary creation from files paths
    # data_dict_train = None
    # data_dict_val = None

    data_dict = {
        '/Users/mariadobko/Documents/Cornell/LAB/NLST First 60 Raw/NLST First 60 Raw - Part01 - 10Pats/100004/'
        '01-02-1999-NLST-LSS-63991/1.000000-0OPAGELSPLUSD4102.512080.00.10.75-24639':
            '/Users/mariadobko/Downloads/Annotations - VT/100004.nii',
        '/Users/mariadobko/Documents/Cornell/LAB/NLST First 60 Raw/NLST First 60 Raw - Part01 - 10Pats/100004/'
        '01-02-1999-NLST-LSS-63991/1.000000-0OPAGELSPLUSD4102.512080.00.10.75-24639':
            '/Users/mariadobko/Downloads/Annotations - VT/100004.nii',
        '/Users/mariadobko/Documents/Cornell/LAB/NLST First 60 Raw/NLST First 60 Raw - Part01 - 10Pats/100004/'
        '01-02-1999-NLST-LSS-63991/1.000000-0OPAGELSPLUSD4102.512080.00.10.75-24639':
            '/Users/mariadobko/Downloads/Annotations - VT/100004.nii',
        '/Users/mariadobko/Documents/Cornell/LAB/NLST First 60 Raw/NLST First 60 Raw - Part01 - 10Pats/100004/'
        '01-02-1999-NLST-LSS-63991/1.000000-0OPAGELSPLUSD4102.512080.00.10.75-24639':
            '/Users/mariadobko/Downloads/Annotations - VT/100004.nii'
    }

    data_dict = collections.OrderedDict(data_dict)
    data_dict_train = data_to_slices(data_dict)
    data_dict_val = data_to_slices(data_dict)

    # Init Lightning Data Module
    dm = NLST_2D_DataModule(
        data_dict_train=data_dict_train,
        data_dict_val=data_dict_val,
        batch_size=config['data']['batch_size_per_gpu'],
        num_workers=config['data']['dataloader_workers_per_gpu'],
        target_size=config['data']['target_size'],
        transform=config['train']['aug'],
    )

    # Init model
    model = NLSTTrainingModule(
        net=config['model']['name'],
        lr=config['train']['lr'],
        loss=config['train']['loss'],
        spatial_dims=2
    )

    # Set callbacks
    if os.environ.get('LOCAL_RANK', 0) == 0:
        experiment_name = '{}__{}'.format(
            config['model']['name'],
            str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        )
    else:
        experiment_name, log_path = '', ''

    # create logging directory
    log_path = os.path.join(config['logging']['root_path'], config['logging']['name'], experiment_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    # write main config
    with open(os.path.join(log_path, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(config))

    checkpoint_callback = ModelCheckpoint(
        dirpath=log_path,
        filename='{step:d}',
        every_n_epochs=1,
        save_top_k=10,
        mode='min',
        monitor='val_epoch/loss',
        auto_insert_metric_name=True
    )
    lr_monitor = LearningRateMonitor()
    callbacks = [checkpoint_callback, lr_monitor]

    tb_logger = TensorBoardLogger(config['logging']['root_path'], config['logging']['name'], version=experiment_name)

    trainer = pl.Trainer(
        gpus=config.get('gpus'),
        max_epochs=config['train']['epochs'],
        accelerator='cpu', #"cuda",
        gradient_clip_val=config['train'].get('grad_clip', 0),
        log_every_n_steps=config['logging']['train_logs_steps'],
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=[tb_logger],
        precision=config['train'].get('precision', 32),
    )

    trainer.fit(model, datamodule=dm, ckpt_path=config.get('checkpoint'))


if __name__ == '__main__':
    main()