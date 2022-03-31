import os
import shutil
import torch
import torchvision as tv
import math

try:
    from tqdm import tqdm as progress_bar
except ImportError:
    progress_bar = lambda x: x


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

    os.makedirs(new_dir, exist_ok=True)

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

def split_into_train_val_test(data_dir: str, split: tuple, delete_old: bool):
    """Refactors data directory structure into train/val/test directories.
    End result will look like this:
    root
    |- train
        |- class1
           |- image1
           |- image2
           ...
        |- class2
        ...
    |- val
        |- class1
           |- image1
           |- image2
           ...
        |- class2
        ...
    |- test
        ...

    :param data_dir: Path to the root of dataset
    :type data_dir: str
    :param split: three elements representing train-validation-test ratio. e.g. (6,2,2) would correspond to 60% of data used for training and 20% for both validation and testing
    :type split: tuple
    :param delete_old: If true old dataset will be deleted from memory. Use true if memory is limited
    :type delete_old: bool
    """

    if len(split)!=3:
        raise Exception("Invalid train:validation:test ratio")

    split_ratio=[]
    for s in split:
        split_ratio.append(s/sum(split))

    if delete_old:
        refactor_method = os.rename  # shutil.move works only on dirs
    else:
        refactor_method = shutil.copyfile

    classes = os.listdir(data_dir)
    for class_name in progress_bar(classes):
        if class_name in ['train', 'val', 'test']:
            raise Exception("Data has already been splitted!")
        class_path = os.path.join(data_dir, class_name)
        files = os.listdir(class_path)
        train_num = math.ceil(len(files) * split_ratio[0])
        val_num = train_num + int(len(files) * split_ratio[1])
        print(train_num, val_num)
        sorted_files=sorted(files)
        train_images = sorted_files[:train_num]
        val_images = sorted_files[train_num:val_num]
        test_images = sorted_files[val_num:]

        os.makedirs(os.path.join(data_dir, 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'val', class_name), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'test', class_name), exist_ok=True)
        for im in train_images:
            refactor_method(
                    os.path.join(data_dir, class_name, str(im)),
                    os.path.join(data_dir, 'train', class_name, str(im))
            )
        for im in val_images:
            refactor_method(
                    os.path.join(data_dir, class_name, str(im)),
                    os.path.join( data_dir, 'val', class_name, str(im))
            )
        for im in test_images:
            refactor_method(
                    os.path.join( data_dir, class_name, str(im)),
                    os.path.join(data_dir, 'test', class_name, str(im))
            )
        if delete_old:
            os.rmdir(os.path.join(data_dir, class_name))
