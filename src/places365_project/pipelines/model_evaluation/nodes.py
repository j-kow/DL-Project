import torch
from torch.utils.data import DataLoader
from typing import Tuple
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..model_training.model import PlacesModel
from ..model_training.nodes import create_data_loader


def load_model_and_data(model_path: str, input_dir: str, batch_size: int, num_workers: int, model_checkpoint: str) -> Tuple[
    PlacesModel, DataLoader]:
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
    :param model_checkpoint: name of model checkpoint
    :type model_checkpoint: str
    :return: Model object alongside train, validation, test data loaders
    :rtype: Tuple[PlacesModel, torch.utils.data.DataLoader]
    """
    model = PlacesModel.load_from_checkpoint(os.path.join(model_path, model_checkpoint))
    test_set = create_data_loader(input_dir=input_dir, train_val_test='test', batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers)

    return model, test_set


def test_model(model: PlacesModel, test_dataset: DataLoader, result_path: str, figsize: int):
    """
    Trains the model

    :param model: Model object created with initialize()
    :type model: PlacesModel
    :param test_dataset: DataLoader of test dataset.
    :type test_dataset: DataLoader
    :param result_path: Path where the resulting confusion matrix will be stored
    :type result_path: str
    :param figsize: Size of confusion matrix image
    :type figsize: int
    :return: Accuracy of the model on the test dataset and confusion matrix
    :rtype: Tuple[int, np.array]
    """
    device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_predicted, all_true = [], []
    for x, y_true in tqdm(test_dataset):
        y_pred = model(x.to(device)).argmax(axis=1)
        all_true += y_true.tolist()
        all_predicted += y_pred.tolist()

    all_true = np.array(all_true)
    all_predicted = np.array(all_predicted)
    accuracy = np.mean(all_predicted == all_true)
    cf_matrix = confusion_matrix(all_true, all_predicted)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10)

    plt.figure(figsize=(figsize, figsize))
    sn.heatmap(df_cm)
    plt.savefig(os.path.join(result_path, "confusion_matrix.png"))

    print(" ============================ ")
    print(f" ACCURACY OF A TEST SET: {accuracy}")
    print(f" CONFUSION MATRIX WAS SAVED IN {result_path}")
    print(" ============================ ")

    return accuracy, cf_matrix
