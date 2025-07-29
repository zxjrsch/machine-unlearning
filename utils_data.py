import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor

from datasets import load_dataset


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
        self, dataset_name: str, forget_class: int, batch_size: int, dataset_path: str
    ) -> None:
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.forget_class = forget_class

    @abstractmethod
    def get_train_loader(self) -> DataLoader:
        raise NotImplementedError()

    @abstractmethod
    def get_val_loader(self) -> DataLoader:
        raise NotImplementedError()

    @abstractmethod
    def get_single_class(self, class_id: int, is_train: bool) -> DataLoader:
        raise NotImplementedError()

    @abstractmethod
    def get_retain_set(self, is_train: bool) -> DataLoader:
        raise NotImplementedError()

    def get_forget_set(self, is_train: bool) -> DataLoader:
        return self.get_single_class(class_id=self.forget_class, is_train=is_train)


def get_unlearning_dataset(
    dataset: SupportedDatasets,
    forget_class: int,
    batch_size: int,
    dataset_path: Optional[str | Path] = None,
) -> UnlearningDataset:

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
            return MIMU_cifar100(forget_class=forget_class, batch_size=batch_size)
        else:
            return MIMU_cifar100(
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

    if dataset == SupportedDatasets.MNIST:
        if dataset_path is None:
            return MIMU_mnist(forget_class=forget_class, batch_size=batch_size)
        else:
            return MIMU_mnist(
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


class MIMU_mnist(UnlearningDataset):
    def __init__(self, forget_class, batch_size, dataset_path="datasets"):
        super().__init__(
            dataset_name=SupportedDatasets.MNIST,
            forget_class=forget_class,
            batch_size=batch_size,
            dataset_path=dataset_path,
        )
        self.transform = Compose([ToTensor()])

    def get_train_loader(self) -> DataLoader:
        dataset = datasets.MNIST(
            root=self.dataset_path, train=True, transform=self.transform, download=True
        )
        return DataLoader(dataset=dataset, batch_size=self.batch_size)

    def get_val_loader(self) -> DataLoader:
        dataset = datasets.MNIST(
            root=self.dataset_path, train=False, transform=self.transform, download=True
        )
        return DataLoader(dataset=dataset, batch_size=self.batch_size)

    def get_retain_set(self, is_train: bool) -> DataLoader:
        dataset = datasets.MNIST(
            root=self.dataset_path,
            train=is_train,
            transform=self.transform,
            download=True,
        )
        retain_indices = (dataset.targets != self.forget_class).nonzero(as_tuple=True)[
            0
        ]
        retain_set = Subset(dataset, retain_indices)
        return DataLoader(dataset=retain_set, batch_size=self.batch_size)

    def get_single_class(self, class_id: int, is_train: bool) -> DataLoader:
        dataset = datasets.MNIST(
            root=self.dataset_path,
            train=is_train,
            transform=self.transform,
            download=True,
        )
        desired_indices = (dataset.targets == class_id).nonzero(as_tuple=True)[0]
        desired_data = Subset(dataset, desired_indices)
        return DataLoader(dataset=desired_data, batch_size=self.batch_size)


class MIMU_cifar10(UnlearningDataset):
    def __init__(self, forget_class=9, batch_size=64, dataset_path="datasets") -> None:
        super().__init__(
            dataset_name=SupportedDatasets.CIFAR10.value,
            forget_class=forget_class,
            batch_size=batch_size,
            dataset_path=dataset_path,
        )
        self.transform = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def get_train_loader(self):

        dataset = datasets.CIFAR10(
            root=self.dataset_path,
            train=True,
            transform=self.transform,
            download=True,
        )

        dataset.targets = torch.tensor(dataset.targets)

        return DataLoader(dataset=dataset, batch_size=self.batch_size)

    def get_val_loader(self) -> DataLoader:

        dataset = datasets.CIFAR10(
            root=self.dataset_path,
            train=False,
            transform=self.transform,
            download=True,
        )
        dataset.targets = torch.tensor(dataset.targets)

        return DataLoader(dataset=dataset, batch_size=self.batch_size)

    def get_single_class(self, class_id: int, is_train: bool) -> DataLoader:

        dataset = datasets.CIFAR10(
            root=self.dataset_path,
            train=is_train,
            transform=self.transform,
            download=True,
        )
        dataset.targets = torch.tensor(dataset.targets)
        forget_indices = (dataset.targets == class_id).nonzero(as_tuple=True)[0]
        forget_set = Subset(dataset, forget_indices)
        return DataLoader(dataset=forget_set, batch_size=self.batch_size)

    def get_retain_set(self, is_train: bool) -> DataLoader:

        dataset = datasets.CIFAR10(
            root=self.dataset_path,
            train=is_train,
            transform=self.transform,
            download=True,
        )
        dataset.targets = torch.tensor(dataset.targets)
        forget_indices = (dataset.targets != self.forget_class).nonzero(as_tuple=True)[
            0
        ]
        forget_set = Subset(dataset, forget_indices)
        return DataLoader(dataset=forget_set, batch_size=self.batch_size)


class MIMU_cifar100(UnlearningDataset):
    def __init__(self, forget_class=9, batch_size=64, dataset_path="datasets"):
        super().__init__(
            dataset_name=SupportedDatasets.CIFAR100.value,
            forget_class=forget_class,
            batch_size=batch_size,
            dataset_path=dataset_path,
        )
        self.transform = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def get_train_loader(self):

        dataset = datasets.CIFAR100(
            root=self.dataset_path,
            train=True,
            transform=self.transform,
            download=True,
        )

        dataset.targets = torch.tensor(dataset.targets)

        return DataLoader(dataset=dataset, batch_size=self.batch_size)

    def get_val_loader(self) -> DataLoader:

        dataset = datasets.CIFAR100(
            root=self.dataset_path,
            train=False,
            transform=self.transform,
            download=True,
        )
        dataset.targets = torch.tensor(dataset.targets)

        return DataLoader(dataset=dataset, batch_size=self.batch_size)

    def get_single_class(self, class_id: int, is_train: bool) -> DataLoader:
        dataset = datasets.CIFAR100(
            root=self.dataset_path,
            train=is_train,
            transform=self.transform,
            download=True,
        )
        dataset.targets = torch.tensor(dataset.targets)
        forget_indices = (dataset.targets == class_id).nonzero(as_tuple=True)[0]
        forget_set = Subset(dataset, forget_indices)
        return DataLoader(dataset=forget_set, batch_size=self.batch_size)

    def get_retain_set(self, is_train: bool) -> DataLoader:

        dataset = datasets.CIFAR100(
            root=self.dataset_path,
            train=is_train,
            transform=self.transform,
            download=True,
        )
        dataset.targets = torch.tensor(dataset.targets)
        forget_indices = (dataset.targets != self.forget_class).nonzero(as_tuple=True)[
            0
        ]
        forget_set = Subset(dataset, forget_indices)
        return DataLoader(dataset=forget_set, batch_size=self.batch_size)


class MIMU_svhn(UnlearningDataset):
    def __init__(self, forget_class=0, batch_size=64, dataset_path="datasets"):
        super().__init__(
            dataset_name=SupportedDatasets.SVHN.value,
            forget_class=forget_class,
            batch_size=batch_size,
            dataset_path=dataset_path,
        )
        self.transform = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def get_train_loader(self) -> DataLoader:
        dataset = datasets.SVHN(
            root=self.dataset_path,
            split="train",
            transform=self.transform,
            download=True,
        )
        return DataLoader(dataset=dataset, batch_size=self.batch_size)

    def get_val_loader(self) -> DataLoader:
        dataset = datasets.SVHN(
            root=self.dataset_path,
            split="test",
            transform=self.transform,
            download=True,
        )
        return DataLoader(dataset=dataset, batch_size=self.batch_size)

    def get_single_class(self, class_id: int, is_train: bool) -> DataLoader:
        dataset = datasets.SVHN(
            root=self.dataset_path,
            split="train" if is_train else "test",
            transform=self.transform,
            download=True,
        )
        forget_indices = (dataset.labels == class_id).nonzero(as_tuple=True)[0]
        forget_set = Subset(dataset, forget_indices)
        return DataLoader(dataset=forget_set, batch_size=self.batch_size)

    def get_retain_set(self, is_train: bool) -> DataLoader:
        dataset = datasets.SVHN(
            root=self.dataset_path,
            split="train" if is_train else "test",
            transform=self.transform,
            download=True,
        )
        forget_indices = (dataset.labels != self.forget_class).nonzero(as_tuple=True)[0]
        forget_set = Subset(dataset, forget_indices)
        return DataLoader(dataset=forget_set, batch_size=self.batch_size)


class MIMU_imagenet_small(UnlearningDataset):
    def __init__(
        self, forget_class=0, batch_size=64, dataset_path="~/Datasets/ImageNet-small"
    ):

        super().__init__(
            dataset_name=SupportedDatasets.IMAGENET_SMALL.value,
            forget_class=forget_class,
            batch_size=batch_size,
            dataset_path=dataset_path,
        )

        assert os.path.exists(
            self.dataset_path
        ), f"Data not found, need to run data scripts to download and unzip this dataset."


class MIMU_pokemon(UnlearningDataset):
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
            dataset_path=dataset_path,
        )

        assert os.path.exists(
            self.dataset_path
        ), f"Data not found, need to run data scripts to download and unzip this dataset."

        self.transform = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def batch_transform(self, x):
        x["image"] = [self.transform(img.convert("RGB")) for img in x["image"]]
        return x

    def get_loader(self):
        return (
            load_dataset(self.dataset_path)
            .with_format("torch")
            .with_transform(self.batch_transform)
        )


class MIMU_plant(UnlearningDataset):
    def __init__(
        self,
        forget_class=0,
        batch_size=64,
        dataset_path="~/Datasets/Plant-classification",
    ):

        super().__init__(
            dataset_name=SupportedDatasets.PLANT_CLASSIFICATION.value,
            forget_class=forget_class,
            batch_size=batch_size,
            dataset_path=dataset_path,
        )

        assert os.path.exists(
            self.dataset_path
        ), f"Data not found, need to run data scripts to download and unzip this dataset."
