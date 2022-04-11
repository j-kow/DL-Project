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


def initialize(input_dir: str, batch_size: int, lr: float, patience: int, frequency: int, num_workers: int,
               no_classes: int) -> Tuple[PlacesModel, Tuple[DataLoader, DataLoader, DataLoader]]:
    """
    Initializes neural network for solving classification problem on Places365 dataset

    :param input_dir: Path to the split dataset.
    :type input_dir: str
    :param batch_size: Number of images in a batch.
    :type batch_size: int
    :param lr: Learning rate for model's optimizer.
    :type lr: float
    :param patience: How to wait before learning rate scheduler fires up.
    :type patience: number
    :param frequency: Frequency for model's optimizer.
    :type frequency: int
    :param num_workers: How many subprocesses are used for data loading.
    :type num_workers: int
    :param no_classes: How many calls are in the dataset, used for classifier retraining.
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


def train_model(model: PlacesModel, sets: tuple, max_epochs: int, checkpoint_path: str, frequency: int) -> pl.Trainer:
    """Trains the model

    :param model: Model object created with initialize()
    :type model: PlacesModel
    :param sets: Tuple of 3 DataLoaders, each representing training, validation and test sets, in said order
    :type sets: tuple
    :param max_epochs: Max number of epochs
    :type max_epochs: number
    :param checkpoint_path: Path to directory in which to save model checkpoints
    :type checkpoint_path: str
    :param frequency: Frequency of validating model. If passed 1, validation will be performed every epoch.
    Must be equal or lower than initialize() frequency argument
    :return: Trainer of a model which can be later used for evaluation on validset/testset
    :rtype: pl.Trainer
    """

    train_data, val_data, test_data = sets
    wandb_logger = WandbLogger(project="Places365")

    gpu_devices = 1 if torch.cuda.is_available() else 0

    trainer = pl.Trainer(
        progress_bar_refresh_rate=10,
        check_val_every_n_epoch=frequency,
        max_epochs=max_epochs,
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.EarlyStopping('val_loss', patience=3),
            pl.callbacks.ModelCheckpoint(dirpath=checkpoint_path)
        ],
        gpus=gpu_devices,
        logger=wandb_logger,
    )

    trainer.fit(model, train_data, val_data)
    return trainer


def create_gridsearch_parameters() -> Dict[str, Tuple]:
    """
    Creates a dictionary containing parameters to test during gridsearch.

    :return: Dictionary of possible values for a given parameter
    :rtype: Dict[str, Tuple]
    """
    params = {
        "lr": (0.003, 0.01, 0.03, 0.1, 0.3),
        "patience": (1, 2, 3, 4, 5),
        "frequency": (1, 2, 3, 4, 5)
    }
    return params


def run_gridsearch(input_dir: str, batch_size: int, lr: float, patience: int, frequency: int, num_workers: int,
                   no_classes: int, max_epochs: int, checkpoint_path: str, gridsearch_params: Dict[str, Tuple]):
    """
    Runs gridsearch over parameters defined in gridsearch_params
    This function trains model over every single combination of parameters defined in gridsearch_params,
    model with the best accuracy will be stored in checkpoint_path.

    Arguments input_dir, batch_size, lr, patience, frequency, num_workers and no_classes are default values for
    initialize() parameters. If these parameters are not defined in gridsearch_params, default values will be taken,
    otherwise these arguments will be ignored.

    :param input_dir: Path to the split dataset
    :type input_dir: str
    :param batch_size: Number of images in a batch.
    :type batch_size: int
    :param lr: Learning rate for model's optimizer.
    :type lr: float
    :param patience: How to wait before learning rate scheduler fires up.
    :type patience: number
    :param frequency: Frequency for model's optimizer.
    :type frequency: int
    :param num_workers: How many subprocesses are used for data loading.
    :type num_workers: int
    :param no_classes: How many calls are in the dataset, used for classifier retraining.
    :type no_classes: int
    :param max_epochs: Max number of epochs
    :type max_epochs: number
    :param checkpoint_path: Path to directory in which to save model checkpoints
    :type checkpoint_path: str
    :param gridsearch_params: Parameters over which to perform gridsearch. Output of create_gridsearch_parameters
    :type Dict[str, Tuple]:
    """
    best_path = os.path.join(checkpoint_path, "best")
    current_path = os.path.join(checkpoint_path, "current")
    default_parameters = {
        "input_dir": input_dir, "batch_size": batch_size, "lr": lr, "patience": patience, "frequency": frequency,
        "num_workers": num_workers, "no_classes": no_classes
    }
    name_variable_pair_list = [
        [(name, value) for value in gridsearch_params[name]] for name in gridsearch_params.keys()
    ]

    best_acc = 0
    for named_parameters in itertools.product(*name_variable_pair_list):
        named_parameters = dict(named_parameters)
        params_str = ", ".join([f"{name}={value}" for name, value in named_parameters.items()])
        
        print("     =================================== ")
        print("    RUNNING TRAINING OF MODEL WITH PARAMS:")
        print("    " + params_str)
        print("     =================================== ")

        for parameter_name, value in default_parameters.items():
            if named_parameters.get(parameter_name) is None:
                named_parameters[parameter_name] = value

        model, (train, val, test) = initialize(**named_parameters)
        trainer = train_model(model, (train, val, test), max_epochs, current_path, named_parameters["frequency"])
        metrics = trainer.validate(model, val)[0]
        if metrics["val_acc"] > best_acc:
            best_acc = metrics["val_acc"]
            try:
                shutil.rmtree(best_path)
            except FileNotFoundError:
                pass
            shutil.copytree(current_path, best_path)
        shutil.rmtree(current_path)
    for f in os.listdir(best_path):
        shutil.move(os.path.join(best_path, f), checkpoint_path)
    shutil.rmtree(best_path)
