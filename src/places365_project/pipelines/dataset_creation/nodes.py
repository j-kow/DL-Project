import os
import shutil

try:
    from tqdm import tqdm as progress_bar
except ImportError:
    progress_bar = lambda x: x


def prune(data_dir: str, prune_frac: float):
    classes = os.listdir(data_dir)
    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        files = os.listdir(class_path)
        n_images = int(len(files) * prune_frac)
        images = sorted(files)[n_images:]
        for im in images:
            os.remove(os.path.join(class_path, im))


def refactor_data_structure(raw_dir: str, new_dir: str, delete_old: bool):
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
