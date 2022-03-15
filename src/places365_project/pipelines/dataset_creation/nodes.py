import os
import shutil
import torch
import torchvision as tv

try:
    from tqdm import tqdm as progress_bar
except ImportError:
    progress_bar = lambda x: x


def create_data_loader(data_dir: str, batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
    """Creates pytorch data loader from places365 dataset

        :param data_dir: Path to the root of processed dataset
        :type data_dir: str
        :param batch_size: Number of images in a batch
        :type batch_size: int
        :param num_workers: How many subprocesses are used for data loading
        :type num_workers: int
        :return: Data loader for places365 dataset
        :rtype: torch.utils.data.DataLoader
        """
    dataset = tv.datasets.ImageFolder(
        data_dir,
        transform=tv.transforms.Compose([
            tv.transforms.PILToTensor(),
            tv.transforms.ConvertImageDtype(torch.float),
            tv.transforms.Lambda(lambda x: x/255)
        ])
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def prune(data_dir: str, prune_frac: float):
    """Deletes set percentage of samples from each class

    :param data_dir: Path to the root of processed dataset
    :type data_dir: str
    :param prune_frac: Percentage of samples to stay in dataset
    :type data_dir: str
    """
    classes = os.listdir(data_dir)
    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        files = os.listdir(class_path)
        n_images = int(len(files) * prune_frac)
        images = sorted(files)[n_images:]
        for im in images:
            os.remove(os.path.join(class_path, im))


def refactor_data_structure(raw_dir: str, new_dir: str, delete_old: bool):
    """Refactors directory structure to match standard structure for image dataset.
    End result will look like this:

    root
    |- class1
       |- image1
       |- image2
       ...
    |- class2
    ...

    :param raw_dir: Path to the root of original dataset
    :type raw_dir: str
    :param new_dir: Path to the root of processed dataset. Entire dataset will be refactored to this path.
    :type new_dir: str
    :param delete_old: If true old dataset will be deleted from memory. Use true if memory is limited
    :type delete_old: bool
    """
    dirs = os.listdir(raw_dir)

    if delete_old:
        refactor_method = shutil.move
    else:
        refactor_method = shutil.copytree

    for d in progress_bar(dirs):
        old_path = os.path.join(raw_dir, d)
        subdirs = os.listdir(old_path)
        for class_name in subdirs:
            for _, subclasses, images in os.walk(os.path.join(old_path, class_name)):
                break

            if len(subclasses) > 0:
                for subclass in subclasses:
                    refactor_method(
                        os.path.join(old_path, class_name, subclass),
                        os.path.join(new_dir, f"{class_name}_{subclass}")
                    )
            else:
                assert len(images) > 0
                refactor_method(
                    os.path.join(old_path, class_name),
                    os.path.join(new_dir, class_name)
                )

        if delete_old:
            shutil.rmtree(old_path)
