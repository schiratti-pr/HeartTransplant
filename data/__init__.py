from .dataset import NLSTDataset
from .dataloader import NLSTDataModule

datasets = {
    # add datasets here
    'NLSTDataset': NLSTDataset
}

data_modules = {
    # add new data modules here
    'NLSTDataModule': NLSTDataModule,
}

__all__ = ['datasets', 'data_modules']
