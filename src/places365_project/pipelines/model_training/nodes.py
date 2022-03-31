import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import LightningModule
import wandb
import torchvision as tv
import sys
import os

#print(sys.path.append(os.path.join('/', *(os.getcwd().split('/')[:-2])))) # add places365_project dir to path
from .model import PlacesModel


def create_data_loader(input_dir: str, train_val_test: str, batch_size: int, shuffle: bool, num_workers: int) -> torch.utils.data.DataLoader:
    """Creates pytorch data loader from places365 dataset

        :param data_dir: Path to the root of processed dataset
        :type data_dir: str
        :param train_val_test: DataLoader for which subset of data to return, train, validation or test
        :type train_val_test: str
        :param batch_size: Number of images in a batch
        :type batch_size: int
        :param shuffle: Whether or not to shuffle the data
        :type shuffle: bool
        :param num_workers: How many subprocesses are used for data loading
        :type num_workers: int
        :return: Data loader for places365 dataset
        :rtype: torch.utils.data.DataLoader
        """
    if train_val_test not in ['train', 'val', 'test']:
        raise Exception("Wrong subset required")
    dataset = tv.datasets.ImageFolder(
        os.path.join(input_dir,train_val_test),
        transform=tv.transforms.Compose([
            tv.transforms.PILToTensor(),
            tv.transforms.Resize((224,224)),
            tv.transforms.ConvertImageDtype(torch.float),
            tv.transforms.Lambda(lambda x: x/255)
        ])
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def initialize(input_dir: str, batch_size: int, lr: float, patience: int, frequency: int, no_classes: int)-> [PlacesModel, tuple]:
    model=PlacesModel(lr, patience, frequency, no_classes)
    #train_set=create_data_loader(input_dir="../../../../data/05_model_input", train_val_test='train', batch_size=256, shuffle=True, num_workers=1)
    #val_set=create_data_loader(input_dir="../../../../data/05_model_input", train_val_test='val', batch_size=256, shuffle=False, num_workers=1)
    #test_set=create_data_loader(input_dir="../../../../data/05_model_input", train_val_test='test', batch_size=256, shuffle=False, num_workers=1)
    train_set=create_data_loader(input_dir=input_dir, train_val_test='train', batch_size=batch_size, shuffle=True, num_workers=1)
    val_set=create_data_loader(input_dir=input_dir, train_val_test='val', batch_size=batch_size, shuffle=False, num_workers=1)
    test_set=create_data_loader(input_dir=input_dir, train_val_test='test', batch_size=batch_size, shuffle=False, num_workers=1)

    wandb.init(project="Places365", entity="dl_image_classification")

    return model, (train_set, val_set, test_set)

def train_model(model: PlacesModel, sets: tuple, checkpoint_path: str):
    train_data, val_data, test_data = sets
    wandb_logger = WandbLogger(project="Places365")

    trainer = pl.Trainer(
        progress_bar_refresh_rate=10,
        check_val_every_n_epoch=2,
        max_epochs=100,
        callbacks=[ pl.callbacks.LearningRateMonitor(logging_interval="step"),
                    pl.callbacks.EarlyStopping('val_loss', patience=3),
                    pl.callbacks.ModelCheckpoint(dirpath=checkpoint_path)
                    ],
        logger=wandb_logger,
    )
    trainer.fit(model, train_data, val_data)
    trainer.test(model, test_data)
