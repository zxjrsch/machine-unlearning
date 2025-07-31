import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor, Resize

from datasets import load_dataset

from imagenet_classes import IMAGENET2012_CLASSES


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
    def get_single_class(self, class_id: int, is_train: bool = False) -> DataLoader:
        raise NotImplementedError()

    @abstractmethod
    def get_retain_set(self, is_train: bool = False) -> DataLoader:
        raise NotImplementedError()

    def get_forget_set(self, is_train: bool = False) -> DataLoader:
        return self.get_single_class(class_id=self.forget_class, is_train=is_train)


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
        return 150
    else:
        return AssertionError(f"Dataset {dataset} not supported. Please add in utils_data.py")


class MIMU_mnist(UnlearningDataset):
    def __init__(self, forget_class, batch_size, dataset_path="datasets"):
        super().__init__(
            dataset_name=SupportedDatasets.MNIST.value,
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

    def get_retain_set(self, is_train: bool = False) -> DataLoader:
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

    def get_single_class(self, class_id: int, is_train: bool = False) -> DataLoader:
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

    def get_single_class(self, class_id: int, is_train: bool = False) -> DataLoader:

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

    def get_retain_set(self, is_train: bool = False) -> DataLoader:

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

    def get_single_class(self, class_id: int, is_train: bool = False) -> DataLoader:
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

    def get_retain_set(self, is_train: bool = False) -> DataLoader:

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
        dataset.labels = torch.tensor(dataset.labels)
        return DataLoader(dataset=dataset, batch_size=self.batch_size)

    def get_val_loader(self) -> DataLoader:
        dataset = datasets.SVHN(
            root=self.dataset_path,
            split="test",
            transform=self.transform,
            download=True,
        )
        dataset.labels = torch.tensor(dataset.labels)
        return DataLoader(dataset=dataset, batch_size=self.batch_size)

    def get_single_class(self, class_id: int, is_train: bool = False) -> DataLoader:
        dataset = datasets.SVHN(
            root=self.dataset_path,
            split="train" if is_train else "test",
            transform=self.transform,
            download=True,
        )
        dataset.labels = torch.tensor(dataset.labels)

        forget_indices = (dataset.labels == class_id).nonzero(as_tuple=True)[0]
        forget_set = Subset(dataset, forget_indices)
        return DataLoader(dataset=forget_set, batch_size=self.batch_size)

    def get_retain_set(self, is_train: bool = False) -> DataLoader:
        dataset = datasets.SVHN(
            root=self.dataset_path,
            split="train" if is_train else "test",
            transform=self.transform,
            download=True,
        )
        dataset.labels = torch.tensor(dataset.labels)
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
            dataset_path=os.path.expanduser(dataset_path),
        )

        assert os.path.exists(
            self.dataset_path
        ), f"Data not found, need to run data scripts to download and unzip this dataset."


class MIMU_pokemon(UnlearningDataset):
    transform = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
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
        images = [item['image'] for item in batch]
        labels = [item['label'] for item in batch]

        images = torch.stack(images)
        labels = torch.tensor(labels)
        return images, labels

    def get_dataset(self, split: str):
        return (
            load_dataset(self.dataset_path, split=split)
            .with_format("torch")
            .with_transform(MIMU_pokemon.batch_transform)
        )
    
    def get_train_loader(self) -> DataLoader:
        dataset = self.get_dataset(split='train')
        return DataLoader(dataset=dataset, batch_size=self.batch_size, collate_fn=MIMU_pokemon.collate_fn, shuffle=True)


    def get_val_loader(self) -> DataLoader:
        dataset_1 = self.get_dataset(split='validation')
        dataset_2 = self.get_dataset(split='test')
        print(len(dataset_1), len(dataset_2))

        return DataLoader(dataset=ConcatDataset([dataset_1, dataset_2]), batch_size=self.batch_size, collate_fn=MIMU_pokemon.collate_fn, shuffle=True)


    def get_single_class(self, class_id: int) -> DataLoader:
        dataset = self.get_dataset(split='train').filter(lambda x: x['label'] == class_id)
        print(f'class id: {len(dataset)} | {class_id}')
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            collate_fn=MIMU_pokemon.collate_fn,
            shuffle=True
        )

    def get_retain_set(self, is_train: bool = False) -> DataLoader:
        raise NotImplementedError()

    def get_forget_set(self, is_train: bool = False) -> DataLoader:
        return self.get_single_class(class_id=self.forget_class, is_train=is_train)



class MIMU_plant(UnlearningDataset):
    transform = Compose(
        [Resize((256, 256)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
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
            dataset_path=os.path.expanduser(dataset_path),
        )

        assert os.path.exists(
            self.dataset_path
        ), f"Data not found, need to run data scripts to download and unzip this dataset."

    @staticmethod
    def batch_transform(x):
        x["image"] = [MIMU_plant.transform(img.convert("RGB")) for img in x["image"]]
        return x
    
    @staticmethod
    def collate_fn(batch):
        images = [item['image'] for item in batch]
        labels = [item['label'] for item in batch]

        images = torch.stack(images)
        labels = torch.tensor(labels)
        return images, labels

    def get_dataset(self, is_train: bool= True):
        if is_train:
            return (
            load_dataset(self.dataset_path + '/train', split='train')
            .with_format("torch")
            .with_transform(MIMU_plant.batch_transform)
            ) 
        else:

            return (
            load_dataset(self.dataset_path + '/test', split='train')
            .with_format("torch")
            .with_transform(MIMU_plant.batch_transform)
            ) 
        
    def get_train_loader(self) -> DataLoader:
        dataset = self.get_dataset(is_train=True)
        return DataLoader(dataset=dataset, batch_size=self.batch_size, collate_fn=MIMU_plant.collate_fn, shuffle=True)


    def get_val_loader(self) -> DataLoader:
        dataset = self.get_dataset(is_train=False)

        return DataLoader(dataset=dataset, batch_size=self.batch_size, collate_fn=MIMU_plant.collate_fn, shuffle=True)


    def get_single_class(self, class_id: int) -> DataLoader:
        dataset = self.get_dataset(is_train=True).filter(lambda x: x['label'] == class_id)
        print(f'class id: {len(dataset)} | {class_id}')
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            collate_fn=MIMU_plant.collate_fn,
            shuffle=True
        )

    def get_retain_set(self, is_train: bool = False) -> DataLoader:
        raise NotImplementedError()

    def get_forget_set(self, is_train: bool = False) -> DataLoader:
        return self.get_single_class(class_id=self.forget_class, is_train=is_train)



if __name__ == "__main__":
    # load_dotenv()
    # hf_token = os.getenv("HF_TOKEN")
    # if not hf_token:
    #     raise AssertionError("Remember to place HF_TOKEN in .env file")
    import code

    plants = MIMU_plant()
    ds = plants.get_val_loader()
    b = next(iter(ds))

    ds = plants.get_single_class(0)

    code.interact(local=locals())
