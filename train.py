import os
import glob
import argparse
from datetime import datetime
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from data.dataloader import NLSTDataModule
from models import NLSTTrainingModule


def main():
    parser = argparse.ArgumentParser(description='Processing configuration for training')
    parser.add_argument('--config', type=str, help='path to config file', default='configs/config.yaml')
    args = parser.parse_args()

    pl.seed_everything(int(os.environ.get('LOCAL_RANK', 0)))

    # Load configuration file
    config = OmegaConf.load(args.config)

    raw_directory = '/Users/mariadobko/Documents/Cornell/LAB/NLST_nifti/'
    annotations_dir = '/Users/mariadobko/Downloads/Annotations - VT 2/'
    data_dict = {}
    for raw_sample in glob.glob(raw_directory + '**'):
        sample_id = raw_sample.split('/')[-1][:-4]
        for label_path in glob.glob(annotations_dir + '**'):
            if sample_id in label_path and '.nii' in label_path:
                data_dict.update({raw_sample:label_path})

    # Data split
    train_set, testing_sets, _, _ = train_test_split(list(data_dict.keys()), list(data_dict.keys()), test_size=0.3,
                                                        random_state=42)
    val_set, test_set, _, _ = train_test_split(testing_sets, testing_sets, test_size=0.5, random_state=42)
    data_dict_train = dict((k, data_dict[k]) for k in train_set)
    data_dict_val = dict((k, data_dict[k]) for k in val_set)
    print('Train:', len(data_dict_train), 'Val:', len(data_dict_val))

    # Init Lightning Data Module
    dm = NLSTDataModule(
        data_dict_train=data_dict_train,
        data_dict_val=data_dict_val,
        nii_format=True,
        batch_size=config['data']['batch_size_per_gpu'],
        num_workers=config['data']['dataloader_workers_per_gpu'],
        target_size=config['data']['target_size'],
        transform=config['train']['aug'],
    )

    # Init model
    model = NLSTTrainingModule(
        net=config['model']['name'],
        lr=config['train']['lr'],
        loss=config['train']['loss']
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
