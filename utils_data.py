import os
from abc import ABC, abstractmethod
from enum import Enum
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from loguru import logger
from PIL import Image
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from datasets import load_dataset
from imagenet_classes import IMAGENET2012_CLASSES

turn_on_download = True
num_cpu = 64
working_dir = Path.cwd()
absolute_path = os.path.expanduser("~/mimu/datasets")

logger.info(
    f"Data path for Torchvision datasets hard coded as: {absolute_path}, working dir {working_dir}"
)


class SupportedDatasets(Enum):
    MNIST = "MNIST"
    CIFAR10 = "CIFAR10"
    CIFAR100 = "CIFAR100"
    SVHN = "SVHN"
    IMAGENET_SMALL = "IMAGENET_SMALL"
    PLANT_CLASSIFICATION = "PLANT_CLASSIFICATION"
    POKEMON_CLASSIFICATION = "POKEMON_CLASSIFICATION"


class UnlearningDataset(ABC):
    def __init__(
        self,
        dataset_name: str,
        forget_class: int,
        batch_size: int,
        dataset_path: str,
        num_classes: int,
    ) -> None:
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.forget_class = forget_class
        self.num_classes = num_classes

    @abstractmethod
    def get_train_loader(self, unit_batch_size: bool = False) -> DataLoader:
        raise NotImplementedError()

    @abstractmethod
    def get_val_loader(self) -> DataLoader:
        raise NotImplementedError()

    @abstractmethod
    def get_single_class(self, class_id: int, is_train: bool = True) -> DataLoader:
        raise NotImplementedError()

    @abstractmethod
    def get_retain_set(self, is_train: bool = True) -> DataLoader:
        raise NotImplementedError()

    def get_forget_set(self, is_train: bool = True) -> DataLoader:
        return self.get_single_class(class_id=self.forget_class, is_train=is_train)

    def reset_batch_size(self, new_batch_size: int = 1) -> None:
        self.batch_size = new_batch_size

    def get_representatives(self) -> List[Tuple[Tensor, Tensor]]:
        # NOTE may need to over-ride for folder based custom datasets
        train_loader = iter(self.get_train_loader(unit_batch_size=True))
        seen_classes = set()
        representatives = []

        while len(seen_classes) < self.num_classes:
            data_point, class_id = next(train_loader)
            if class_id.item() not in seen_classes:
                representatives.append((data_point, class_id))
                seen_classes.add(class_id.item())
                # logger.info(f'Added {class_id.item()} | {len(seen_classes)} / {self.num_classes} classes recorded')

        return representatives


def get_unlearning_dataset(
    dataset: SupportedDatasets,
    batch_size: int,
    forget_class: int = 0,
    dataset_path: Optional[str | Path] = None,
) -> UnlearningDataset:

    if dataset == SupportedDatasets.MNIST:
        if dataset_path is None:
            return MIMU_mnist(forget_class=forget_class, batch_size=batch_size)
        else:
            return MIMU_mnist(
                forget_class=forget_class,
                batch_size=batch_size,
                dataset_path=dataset_path,
            )

    if dataset == SupportedDatasets.CIFAR10:
        if dataset_path is None:
            return MIMU_cifar10(forget_class=forget_class, batch_size=batch_size)
        else:
            return MIMU_cifar10(
                forget_class=forget_class,
                batch_size=batch_size,
                dataset_path=dataset_path,
            )

    if dataset == SupportedDatasets.CIFAR100:
        if dataset_path is None:
            logger.info(
                "here MIMU_cifar100(forget_class=forget_class, batch_size=batch_size)"
            )
            return MIMU_cifar100(forget_class=forget_class, batch_size=batch_size)
        else:
            return MIMU_cifar100(
                forget_class=forget_class,
                batch_size=batch_size,
                dataset_path=dataset_path,
            )
    if dataset == SupportedDatasets.SVHN:
        if dataset_path is None:
            return MIMU_svhn(forget_class=forget_class, batch_size=batch_size)
        else:
            return MIMU_svhn(
                forget_class=forget_class,
                batch_size=batch_size,
                dataset_path=dataset_path,
            )

    if dataset == SupportedDatasets.IMAGENET_SMALL:
        if dataset_path is None:
            return MIMU_imagenet_small(forget_class=forget_class, batch_size=batch_size)
        else:
            return MIMU_imagenet_small(
                forget_class=forget_class,
                batch_size=batch_size,
                dataset_path=dataset_path,
            )

    if dataset == SupportedDatasets.PLANT_CLASSIFICATION:
        if dataset_path is None:
            return MIMU_plant(forget_class=forget_class, batch_size=batch_size)
        else:
            return MIMU_plant(
                forget_class=forget_class,
                batch_size=batch_size,
                dataset_path=dataset_path,
            )

    if dataset == SupportedDatasets.POKEMON_CLASSIFICATION:
        if dataset_path is None:
            return MIMU_pokemon(forget_class=forget_class, batch_size=batch_size)
        else:
            return MIMU_pokemon(
                forget_class=forget_class,
                batch_size=batch_size,
                dataset_path=dataset_path,
            )


def get_vision_dataset_classes(dataset: SupportedDatasets) -> int:
    """Number of classes in respective vision datasets."""
    if dataset == SupportedDatasets.MNIST:
        return 10
    elif dataset == SupportedDatasets.CIFAR10:
        return 10
    elif dataset == SupportedDatasets.CIFAR100:
        return 100
    elif dataset == SupportedDatasets.SVHN:
        return 10
    elif dataset == SupportedDatasets.IMAGENET_SMALL:
        return 1000
    elif dataset == SupportedDatasets.PLANT_CLASSIFICATION:
        return 64
    elif dataset == SupportedDatasets.POKEMON_CLASSIFICATION:
        return 110
    else:
        return AssertionError(
            f"Dataset {dataset} not supported. Please add in utils_data.py"
        )


class MIMU_mnist(UnlearningDataset):
    def __init__(self, forget_class=0, batch_size=64, dataset_path=absolute_path):
        super().__init__(
            dataset_name=SupportedDatasets.MNIST.value,
            forget_class=forget_class,
            batch_size=batch_size,
            dataset_path=dataset_path,
            num_classes=10,
        )
        self.transform = Compose([ToTensor()])

    def get_train_loader(self, unit_batch_size: bool = False) -> DataLoader:
        dataset = datasets.MNIST(
            root=self.dataset_path,
            train=True,
            transform=self.transform,
            download=turn_on_download,
        )
        return DataLoader(
            dataset=dataset, batch_size=1 if unit_batch_size else self.batch_size
        )

    def get_val_loader(self) -> DataLoader:
        dataset = datasets.MNIST(
            root=self.dataset_path,
            train=False,
            transform=self.transform,
            download=turn_on_download,
        )
        return DataLoader(dataset=dataset, batch_size=self.batch_size)

    def get_retain_set(self, is_train: bool = True) -> DataLoader:
        dataset = datasets.MNIST(
            root=self.dataset_path,
            train=is_train,
            transform=self.transform,
            download=turn_on_download,
        )
        retain_indices = (dataset.targets != self.forget_class).nonzero(as_tuple=True)[
            0
        ]
        retain_set = Subset(dataset, retain_indices)
        return DataLoader(dataset=retain_set, batch_size=self.batch_size)

    def get_single_class(self, class_id: int, is_train: bool = True) -> DataLoader:
        dataset = datasets.MNIST(
            root=self.dataset_path,
            train=is_train,
            transform=self.transform,
            download=turn_on_download,
        )
        desired_indices = (dataset.targets == class_id).nonzero(as_tuple=True)[0]
        # logger.info(desired_indices)
        # logger.info(f'MNIST single class loader found {desired_indices.numel()} instances.')
        desired_data = Subset(dataset, desired_indices)
        return DataLoader(dataset=desired_data, batch_size=self.batch_size)


class MIMU_cifar10(UnlearningDataset):
    def __init__(
        self, forget_class=0, batch_size=64, dataset_path=absolute_path
    ) -> None:
        super().__init__(
            dataset_name=SupportedDatasets.CIFAR10.value,
            forget_class=forget_class,
            batch_size=batch_size,
            dataset_path=dataset_path,
            num_classes=10,
        )
        self.transform = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def get_train_loader(self, unit_batch_size: bool = False):

        dataset = datasets.CIFAR10(
            root=self.dataset_path,
            train=True,
            transform=self.transform,
            download=turn_on_download,
        )

        dataset.targets = torch.tensor(dataset.targets)

        return DataLoader(
            dataset=dataset, batch_size=1 if unit_batch_size else self.batch_size
        )

    def get_val_loader(self) -> DataLoader:

        dataset = datasets.CIFAR10(
            root=self.dataset_path,
            train=False,
            transform=self.transform,
            download=turn_on_download,
        )
        dataset.targets = torch.tensor(dataset.targets)

        return DataLoader(dataset=dataset, batch_size=self.batch_size)

    def get_single_class(self, class_id: int, is_train: bool = True) -> DataLoader:

        dataset = datasets.CIFAR10(
            root=self.dataset_path,
            train=is_train,
            transform=self.transform,
            download=turn_on_download,
        )
        dataset.targets = torch.tensor(dataset.targets)
        forget_indices = (dataset.targets == class_id).nonzero(as_tuple=True)[0]
        forget_set = Subset(dataset, forget_indices)
        return DataLoader(dataset=forget_set, batch_size=self.batch_size)

    def get_retain_set(self, is_train: bool = True) -> DataLoader:

        dataset = datasets.CIFAR10(
            root=self.dataset_path,
            train=is_train,
            transform=self.transform,
            download=turn_on_download,
        )
        dataset.targets = torch.tensor(dataset.targets)
        forget_indices = (dataset.targets != self.forget_class).nonzero(as_tuple=True)[
            0
        ]
        forget_set = Subset(dataset, forget_indices)
        return DataLoader(dataset=forget_set, batch_size=self.batch_size)


class MIMU_cifar100(UnlearningDataset):
    def __init__(self, forget_class=0, batch_size=64, dataset_path=absolute_path):
        super().__init__(
            dataset_name=SupportedDatasets.CIFAR100.value,
            forget_class=forget_class,
            batch_size=batch_size,
            dataset_path=dataset_path,
            num_classes=100,
        )
        self.transform = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def get_train_loader(self, unit_batch_size: bool = False):

        dataset = datasets.CIFAR100(
            root=self.dataset_path,
            train=True,
            transform=self.transform,
            download=turn_on_download,
        )

        dataset.targets = torch.tensor(dataset.targets)

        return DataLoader(
            dataset=dataset, batch_size=1 if unit_batch_size else self.batch_size
        )

    def get_val_loader(self) -> DataLoader:

        dataset = datasets.CIFAR100(
            root=self.dataset_path,
            train=False,
            transform=self.transform,
            download=turn_on_download,
        )
        dataset.targets = torch.tensor(dataset.targets)

        return DataLoader(dataset=dataset, batch_size=self.batch_size)

    def get_single_class(self, class_id: int, is_train: bool = True) -> DataLoader:
        dataset = datasets.CIFAR100(
            root=self.dataset_path,
            train=is_train,
            transform=self.transform,
            download=turn_on_download,
        )
        dataset.targets = torch.tensor(dataset.targets)
        forget_indices = (dataset.targets == class_id).nonzero(as_tuple=True)[0]
        # logger.info(f'{self.dataset_name} | class_id {class_id} get single_class | found {forget_indices.numel()} instances')
        forget_set = Subset(dataset, forget_indices)
        return DataLoader(dataset=forget_set, batch_size=self.batch_size)

    def get_retain_set(self, is_train: bool = True) -> DataLoader:

        dataset = datasets.CIFAR100(
            root=self.dataset_path,
            train=is_train,
            transform=self.transform,
            download=turn_on_download,
        )
        dataset.targets = torch.tensor(dataset.targets)
        forget_indices = (dataset.targets != self.forget_class).nonzero(as_tuple=True)[
            0
        ]
        forget_set = Subset(dataset, forget_indices)
        return DataLoader(dataset=forget_set, batch_size=self.batch_size)


class MIMU_svhn(UnlearningDataset):
    def __init__(self, forget_class=0, batch_size=64, dataset_path=absolute_path):
        super().__init__(
            dataset_name=SupportedDatasets.SVHN.value,
            forget_class=forget_class,
            batch_size=batch_size,
            dataset_path=dataset_path,
            num_classes=10,
        )
        self.transform = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def get_train_loader(self, unit_batch_size: bool = False) -> DataLoader:
        dataset = datasets.SVHN(
            root=self.dataset_path,
            split="train",
            transform=self.transform,
            download=turn_on_download,
        )
        dataset.labels = torch.tensor(dataset.labels)
        return DataLoader(
            dataset=dataset, batch_size=1 if unit_batch_size else self.batch_size
        )

    def get_val_loader(self) -> DataLoader:
        dataset = datasets.SVHN(
            root=self.dataset_path,
            split="test",
            transform=self.transform,
            download=turn_on_download,
        )
        dataset.labels = torch.tensor(dataset.labels)
        return DataLoader(dataset=dataset, batch_size=self.batch_size)

    def get_single_class(self, class_id: int, is_train: bool = True) -> DataLoader:
        dataset = datasets.SVHN(
            root=self.dataset_path,
            split="train" if is_train else "test",
            transform=self.transform,
            download=turn_on_download,
        )
        dataset.labels = torch.tensor(dataset.labels)
        forget_indices = (dataset.labels == class_id).nonzero(as_tuple=True)[0]
        logger.info(
            f"{self.dataset_name} | class_id {class_id} get single_class | found {forget_indices.numel()} instances"
        )
        forget_set = Subset(dataset, forget_indices)
        return DataLoader(dataset=forget_set, batch_size=self.batch_size)

    def get_retain_set(self, is_train: bool = True) -> DataLoader:
        dataset = datasets.SVHN(
            root=self.dataset_path,
            split="train" if is_train else "test",
            transform=self.transform,
            download=turn_on_download,
        )
        dataset.labels = torch.tensor(dataset.labels)
        forget_indices = (dataset.labels != self.forget_class).nonzero(as_tuple=True)[0]
        forget_set = Subset(dataset, forget_indices)
        return DataLoader(dataset=forget_set, batch_size=self.batch_size)


class ImageNetDataset(Dataset):
    def __init__(
        self,
        is_train: bool = True,
        transform=None,
        dataset_path="~/Datasets/ImageNet-small",
    ):
        self.is_train = is_train
        self.dataset_path = os.path.expanduser(dataset_path)
        self.files = self.get_files()
        self.class_to_idx = {k: i for i, k in enumerate(IMAGENET2012_CLASSES.keys())}
        self.labels = self.get_all_labels()

        self.transform = transform or Compose(
            [
                Resize((128, 128)),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def get_all_labels(self):
        lablels = []
        for f in self.files:
            # sample path /Datasets/ImageNet-small/val_images/ILSVRC2012_val_00017551_n02106166.JPEG
            image_class = f.split("/")[-1].split(".")[0].split("_")[-1]
            label = self.class_to_idx.get(image_class, -1)
            lablels.append(label)
        return torch.Tensor(lablels)

    def get_files(self):
        files = []
        if self.is_train:
            num_train_splits = 4
            for s in range(num_train_splits):
                pattern = self.dataset_path + f"/train_images_{s}/*.JPEG"
                files += glob(pattern)
        else:
            pattern = self.dataset_path + f"/val_images/*.JPEG"
            files += glob(pattern)
        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        index = index % self.__len__()
        path = self.files[index]
        image = Image.open(path).convert("RGB")

        # sample path /Datasets/ImageNet-small/val_images/ILSVRC2012_val_00017551_n02106166.JPEG
        image_class = path.split("/")[-1].split(".")[0].split("_")[-1]
        label = self.class_to_idx.get(image_class, -1)

        if label == -1:
            raise ValueError(f"Unknown class '{image_class}' in file {path}")

        image = self.transform(image)
        return image, label


class MIMU_imagenet_small(UnlearningDataset):
    def __init__(
        self,
        forget_class=0,
        batch_size=64,
        dataset_path="~/Datasets/ImageNet-small",
        num_workers: int = 64,
    ):
        self.num_workers = num_workers
        super().__init__(
            dataset_name=SupportedDatasets.IMAGENET_SMALL.value,
            forget_class=forget_class,
            batch_size=batch_size,
            dataset_path=os.path.expanduser(dataset_path),
            num_classes=1000,
        )

        assert os.path.exists(
            self.dataset_path
        ), f"Data not found, need to run data scripts to download and unzip this dataset."

    def get_train_loader(self, unit_batch_size: bool = False) -> DataLoader:
        dataset = ImageNetDataset(is_train=True, dataset_path=self.dataset_path)
        return DataLoader(
            dataset=dataset,
            batch_size=1 if unit_batch_size else self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def get_val_loader(self) -> DataLoader:
        dataset = ImageNetDataset(is_train=False, dataset_path=self.dataset_path)
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def get_single_class(self, class_id: int, is_train: bool = True) -> DataLoader:
        dataset = ImageNetDataset(is_train=is_train, dataset_path=self.dataset_path)

        forget_indices = (dataset.labels == class_id).nonzero(as_tuple=True)[0]
        logger.info(
            f"{self.dataset_name} | class_id {class_id} get single_class | found {forget_indices.numel()} instances"
        )
        forget_set = Subset(dataset, forget_indices)
        return DataLoader(
            dataset=forget_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def get_retain_set(self, is_train: bool = False) -> DataLoader:
        if is_train:
            logger.info(
                "ImageNet retain set getter is_train is set to True, this will take long."
            )

        dataset = ImageNetDataset(is_train=is_train, dataset_path=self.dataset_path)

        forget_indices = (dataset.labels != self.forget_class).nonzero(as_tuple=True)[0]
        forget_set = Subset(dataset, forget_indices)
        return DataLoader(
            dataset=forget_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def get_forget_set(self, is_train=False):
        if is_train:
            logger.info(
                "ImageNet forget set getter has is_train set to True, this will take long."
            )
        return self.get_single_class(class_id=self.forget_class, is_train=is_train)


class MIMU_pokemon(UnlearningDataset):
    transform = Compose(
        [Resize((128, 128)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def __init__(
        self,
        forget_class=0,
        batch_size=64,
        dataset_path="~/Datasets/Pokemon-classification",
    ):

        super().__init__(
            dataset_name=SupportedDatasets.POKEMON_CLASSIFICATION.value,
            forget_class=forget_class,
            batch_size=batch_size,
            dataset_path=os.path.expanduser(dataset_path),
            num_classes=110,
        )

        assert os.path.exists(
            self.dataset_path
        ), f"Data not found, need to run data scripts to download and unzip this dataset."

    @staticmethod
    def batch_transform(x):
        x["image"] = [MIMU_pokemon.transform(img.convert("RGB")) for img in x["image"]]
        return x

    @staticmethod
    def collate_fn(batch):
        images = [item["image"] for item in batch]
        labels = [item["label"] for item in batch]

        images = torch.stack(images)
        labels = torch.tensor(labels)
        return images, labels

    def get_dataset(self, split: str):
        return (
            load_dataset(self.dataset_path, split=split, num_proc=num_cpu)
            .with_format("torch")
            .with_transform(MIMU_pokemon.batch_transform)
        )

    def get_train_loader(self, unit_batch_size: bool = False) -> DataLoader:
        dataset = self.get_dataset(split="train")
        return DataLoader(
            dataset=dataset,
            batch_size=1 if unit_batch_size else self.batch_size,
            collate_fn=MIMU_pokemon.collate_fn,
            shuffle=True,
        )

    def get_val_loader(self) -> DataLoader:
        dataset_1 = self.get_dataset(split="validation")
        dataset_2 = self.get_dataset(split="test")

        return DataLoader(
            dataset=ConcatDataset([dataset_1, dataset_2]),
            batch_size=self.batch_size,
            collate_fn=MIMU_pokemon.collate_fn,
            shuffle=True,
        )

    def get_single_class(self, class_id: int, is_train: bool = True) -> DataLoader:
        # NOTE is_train=True is hard coded to avoid empty classes
        is_train = True
        if is_train:
            dataset = self.get_dataset(split="train").filter(
                lambda x: x["label"] == class_id, num_proc=num_cpu
            )
        else:
            dataset_1 = self.get_dataset(split="validation").filter(
                lambda x: x["label"] == class_id, num_proc=num_cpu
            )
            dataset_2 = self.get_dataset(split="test").filter(
                lambda x: x["label"] == class_id, num_proc=num_cpu
            )
            dataset = ConcatDataset([dataset_1, dataset_2])

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=MIMU_pokemon.collate_fn,
            shuffle=True,
        )

    def get_retain_set(self, is_train: bool = True) -> DataLoader:
        # NOTE is_train=True is hard coded to avoid empty classes
        is_train = True
        if is_train:
            dataset = self.get_dataset(split="train").filter(
                lambda x: x["label"] != self.forget_class, num_proc=num_cpu
            )
        else:
            dataset_1 = self.get_dataset(split="validation").filter(
                lambda x: x["label"] != self.forget_class, num_proc=num_cpu
            )
            dataset_2 = self.get_dataset(split="test").filter(
                lambda x: x["label"] != self.forget_class, num_proc=num_cpu
            )
            dataset = ConcatDataset([dataset_1, dataset_2])
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=MIMU_pokemon.collate_fn,
            shuffle=True,
        )


class MIMU_plant(UnlearningDataset):
    transform = Compose(
        [Resize((128, 128)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def __init__(
        self,
        forget_class=0,
        batch_size=64,
        dataset_path="~/Datasets/Plant-classification",
        data_cardinality_limit: int = 1024,
    ):

        super().__init__(
            dataset_name=SupportedDatasets.PLANT_CLASSIFICATION.value,
            forget_class=forget_class,
            batch_size=batch_size,
            dataset_path=os.path.expanduser(dataset_path),
            num_classes=64,
        )

        assert os.path.exists(
            self.dataset_path
        ), f"Data not found, need to run data scripts to download and unzip this dataset."

        self.data_cardinality_limit = data_cardinality_limit
        self.string_labels = self.get_subclass_string_names()
        self.str_int_mapping = dict(
            zip(self.string_labels, range(len(self.string_labels)))
        )
        self.int_str_mapping = dict(enumerate(self.string_labels))

    def get_subclass_string_names(self):
        directory = Path(self.dataset_path) / "train"
        subfolders = sorted([p.name for p in directory.iterdir() if p.is_dir()])
        return subfolders

    @staticmethod
    def batch_transform(x):
        x["image"] = [MIMU_plant.transform(img.convert("RGB")) for img in x["image"]]
        return x

    @staticmethod
    def collate_fn(batch):
        # logger.info(batch[0].keys())
        # logger.info(f'len {len(batch)}')
        images = [item["image"] for item in batch]
        labels = [item["label"] for item in batch]

        images = torch.stack(images)
        labels = torch.tensor(labels)
        return images, labels

    @staticmethod
    def constant_collate_fn(batch, constant):
        images = [item["image"] for item in batch]
        labels = [constant for item in batch]

        images = torch.stack(images)
        labels = torch.tensor(labels)
        return images, labels

    def get_dataset(
        self,
        is_train: bool = True,
        enforce_limit: bool = False,
        subclass_folder: Optional[str] = None,
        custom_limit: Optional[int] = None,
    ):
        if custom_limit is None:
            custom_limit = self.data_cardinality_limit
        if is_train:
            suffix = (
                "/train" if subclass_folder is None else f"/train/{subclass_folder}"
            )
            p = Path(self.dataset_path + suffix)
            assert p.exists()
            # logger.info(f"path: {p}")
            loader = (
                load_dataset(
                    self.dataset_path + suffix,
                    split=(f"train[:{custom_limit}]" if enforce_limit else "train"),
                    num_proc=num_cpu,
                )
                .with_format("torch")
                .with_transform(MIMU_plant.batch_transform)
            )
        else:
            suffix = "/test" if subclass_folder is None else f"/test/{subclass_folder}"
            loader = (
                load_dataset(
                    self.dataset_path + suffix,
                    split=f"train[:{custom_limit}]",
                    num_proc=num_cpu,
                )
                .with_format("torch")
                .with_transform(MIMU_plant.batch_transform)
            )
        return loader

    def get_train_loader(self, unit_batch_size: bool = False) -> DataLoader:
        dataset = self.get_dataset(is_train=True)
        return DataLoader(
            dataset=dataset,
            batch_size=1 if unit_batch_size else self.batch_size,
            collate_fn=MIMU_plant.collate_fn,
            shuffle=True,
        )

    def get_val_loader(self) -> DataLoader:
        dataset = self.get_dataset(is_train=False)

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=MIMU_plant.collate_fn,
            shuffle=True,
        )

    def get_single_class(
        self, class_id: int = 0, is_train: bool = True, custom_limit=16
    ) -> DataLoader:

        # dataset = self.get_dataset(is_train=is_train, enforce_limit=True).filter(
        #     lambda x: x["label"] == class_id, num_proc=num_cpu
        # )
        # logger.info(self.int_str_mapping[class_id])
        dataset = self.get_dataset(
            is_train=is_train,
            enforce_limit=True,
            subclass_folder=self.int_str_mapping[class_id],
            custom_limit=custom_limit,
        )
        # logger.info(dataset)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=(
                lambda b: MIMU_plant.constant_collate_fn(batch=b, constant=class_id)
            ),
            shuffle=True,
        )

    def get_retain_set(self, is_train: bool = False, custom_limit=16) -> DataLoader:
        # dataset = self.get_dataset(is_train=is_train, enforce_limit=True).filter(
        #     lambda x: x["label"] != self.forget_class, num_proc=num_cpu
        # )

        ds_array = []
        for i in range(self.num_classes):
            # for i in range(2): # debug only
            if i == self.forget_class:
                continue
            ds = self.get_dataset(
                is_train=is_train,
                enforce_limit=True,
                subclass_folder=self.int_str_mapping[i],
                custom_limit=custom_limit,
            )
            ds = ds.add_column("label", [i for _ in range(custom_limit)])

            ds_array.append(ds)

            # logger.info(ds)

        dataset = ConcatDataset(ds_array)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=MIMU_plant.collate_fn,
            shuffle=True,
        )


if __name__ == "__main__":
    # load_dotenv()
    # hf_token = os.getenv("HF_TOKEN")
    # if not hf_token:
    #     raise AssertionError("Remember to place HF_TOKEN in .env file")
    pass

    # plants = MIMU_plant()
    # ds = plants.get_val_loader()
    # b = next(iter(ds))
    # ds = plants.get_single_class(0)
    # imagenet = ImageNetDataset()
    # loader = DataLoader(imagenet, 64)
    # batch = next(iter(loader))
    # a = perf_counter()
    # imgnet = MIMU_imagenet_small()
    # b = perf_counter()
    # print(f"init MIMU_imagenet_small data class in {b-a} sec ")
    # cls = imgnet.get_train_loader()
    # c = perf_counter()
    # print(f"get train loader MIMU_imagenet_small data class in {c-b} sec ")
    # print(next(iter(cls))[0].shape)
    # run this to download torchvision dataset for the first use case
    cifar1000 = MIMU_cifar100()
    cifar1000.get_train_loader()
    # cifar1000.get_val_loader()

    # code.interact(local=locals())
