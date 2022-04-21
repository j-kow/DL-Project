import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import itertools
import shutil
import os
from typing import Tuple, Dict

from src.places365_project.pipelines.model_training.model import PlacesModel


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
    :param shuffle: Whether to shuffle the data
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


def load_model_and_data(model_path: str, input_dir: str, batch_size: int, num_workers: int) -> Tuple[PlacesModel, DataLoader]:
    """
    Loads neural network for solving classification problem on Places365 dataset

    :param model_path: Path to the model.
    :type model_path: str
    :param input_dir: Path to the split dataset.
    :type input_dir: str
    :param batch_size: Number of images in a batch.
    :type batch_size: int
    :param num_workers: How many subprocesses are used for data loading.
    :type num_workers: int
    :return: Model object alongside train, validation, test data loaders
    :rtype: Tuple[PlacesModel, torch.utils.data.DataLoader]
    """
    model = PlacesModel.load_from_checkpoint(model_path)
    test_set = create_data_loader(input_dir=input_dir, train_val_test='test', batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers)

    return model, test_set


def test_model(model: PlacesModel, test_dataset: DataLoader):
    """
    Trains the model

    :param model: Model object created with initialize()
    :type model: PlacesModel
    :param test_dataset: DataLoader of test dataset.
    :type test_dataset: tuple
    :return: Accuracy of the model on the test dataset and confusion matrix
    :rtype: Tuple[int, np.array]
    """
    gpu_devices = 1 if torch.cuda.is_available() else 0

    trainer = pl.Trainer(
        progress_bar_refresh_rate=10,
        gpus=gpu_devices,
    )

    print(trainer.predict(model, dataloaders=test_dataset, return_predictions=True))

model, datset = load_model_and_data(model_path='../../../../data/06_models/final_model.ckpt', )
