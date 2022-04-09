import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
from typing import Tuple
from .model import PlacesModel


def create_data_loader(input_dir: str, train_val_test: str, batch_size: int, shuffle: bool,
                       num_workers: int) -> DataLoader:
    """
    Creates pytorch data loader from places365 dataset

    :param input_dir: Path to the root of processed dataset
    :type input_dir: str
    :param train_val_test: DataLoader for which subset of data to return, train, validation or test
    :type train_val_test: str
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

    dataset = ImageFolder(
        os.path.join(input_dir, train_val_test),
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def initialize(input_dir: str, batch_size: int, lr: float, patience: int, frequency: int, num_workers: int,
               no_classes: int) -> Tuple[PlacesModel, Tuple[DataLoader, DataLoader, DataLoader]]:
    """
    Initializes neural network for solving classification problem on Places365 dataset

    :param input_dir: Path to the splitted dataset
    :type input_dir: str
    :param batch_size: Number of images in a batch
    :type batch_size: int
    :param lr: Learning rate for model's optimizer
    :type lr: float
    :param patience: How to wait before learning rate scheduler fires up
    :type patience: number
    :param frequency: Frequency for model's optimizer
    :type frequency: int
    :param num_workers: How many subprocesses are used for data loading
    :type num_workers: int
    :param no_classes: How many calls are in the dataset, used for classier retraining
    :type no_classes: int
    :return: Model object alongside train, validation, test data loaders
    :rtype: Tuple[PlacesModel, Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]]
    """
    model = PlacesModel(lr, patience, frequency, no_classes)

    train_set = create_data_loader(input_dir=input_dir, train_val_test='train', batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers)
    val_set = create_data_loader(input_dir=input_dir, train_val_test='val', batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers)
    test_set = create_data_loader(input_dir=input_dir, train_val_test='test', batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers)

    wandb.init(project="Places365", entity="dl_image_classification")

    return model, (train_set, val_set, test_set)


def train_model(model: PlacesModel, sets: tuple, max_epochs: int, checkpoint_path: str):
    """Trains the model

    :param model: Model object created with initialize()
    :type model: PlacesModel
    :param sets: Tuple of 3 DataLoaders, each representing training, validation and test sets, in said order
    :type sets: tuple
    :param max_epochs: Max number of epochs
    :type max_epochs: number
    :param checkpoint_path: Path to directory in which to save model checkpoints
    :type checkpoint_path: str
    """

    train_data, val_data, test_data = sets
    wandb_logger = WandbLogger(project="Places365")

    trainer = pl.Trainer(
        progress_bar_refresh_rate=10,
        check_val_every_n_epoch=2,
        max_epochs=max_epochs,
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.EarlyStopping('val_loss', patience=3),
            pl.callbacks.ModelCheckpoint(dirpath=checkpoint_path)
        ],
        logger=wandb_logger,
    )

    trainer.fit(model, train_data, val_data)
    trainer.test(model, test_data)
