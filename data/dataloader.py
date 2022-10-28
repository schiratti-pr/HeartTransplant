import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.dataset import NLSTDataset, NLST_2D_Dataset
from data.aug import AUGMENTATIONS


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, data_dict_train, data_dict_val, batch_size, num_workers, target_size):
        super(BaseDataModule, self).__init__()
        self.data_dict_train = data_dict_train
        self.data_dict_val = data_dict_val
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_batch_collate_fn = None
        self.val_batch_collate_fn = None

        self.target_size = target_size

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.train_batch_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=1,
            num_workers=1,
            collate_fn=self.val_batch_collate_fn
        )


class NLSTDataModule(BaseDataModule):
    def __init__(self, data_dict_train, data_dict_val, batch_size, num_workers, target_size,
                 transform: dict = None):
        super(NLSTDataModule, self).__init__(
            data_dict_train, data_dict_val, batch_size, num_workers, target_size,
        )
        self.target_size = target_size
        self.transform = AUGMENTATIONS.get(transform['name'])

    def setup(self, stage) -> None:
        self.train_dataset = NLSTDataset(
            patients_paths=self.data_dict_train,
            target_size=self.target_size,
            transform=self.transform
        )
        self.val_dataset = NLSTDataset(
            patients_paths=self.data_dict_val,
            target_size=self.target_size,
            transform=None
        )


class NLST_2D_DataModule(BaseDataModule):
    def __init__(self, data_dict_train, data_dict_val, batch_size, num_workers, target_size,
                 transform: dict = None):
        super(NLST_2D_DataModule, self).__init__(
            data_dict_train, data_dict_val, batch_size, num_workers, target_size,
        )
        self.target_size = target_size
        self.transform = AUGMENTATIONS.get(transform['name'])

    def setup(self, stage) -> None:
        self.train_dataset = NLST_2D_Dataset(
            patients_paths=self.data_dict_train,
            target_size=self.target_size,
            transform=self.transform
        )
        self.val_dataset = NLST_2D_Dataset(
            patients_paths=self.data_dict_val,
            target_size=self.target_size,
            transform=None
        )
