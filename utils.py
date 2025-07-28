import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor

from datasets import load_dataset

global_config = OmegaConf.load("configs/config.yaml")

transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

from torchvision.models import resnet18


# CIFAR 10
def get_cifar10_train_loader(batch_size) -> DataLoader:

    dataset = datasets.CIFAR10(
        root=global_config.dataset_path,
        train=True,
        transform=transform,
        download=True,
    )

    dataset.targets = torch.tensor(dataset.targets)

    return DataLoader(dataset=dataset, batch_size=batch_size)


def get_cifar10_val_loader(batch_size) -> DataLoader:

    dataset = datasets.CIFAR10(
        root=global_config.dataset_path,
        train=False,
        transform=transform,
        download=True,
    )
    dataset.targets = torch.tensor(dataset.targets)

    return DataLoader(dataset=dataset, batch_size=batch_size)


def get_cifar10_forget_set(
    forget_class: int = 9, batch_size: int = 8, is_train: bool = False
) -> DataLoader:
    assert 0 <= forget_class <= 9
    dataset = datasets.CIFAR10(
        root=global_config.dataset_path,
        train=is_train,
        transform=transform,
        download=True,
    )
    dataset.targets = torch.tensor(dataset.targets)
    forget_indices = (dataset.targets == forget_class).nonzero(as_tuple=True)[0]
    forget_set = Subset(dataset, forget_indices)
    return DataLoader(dataset=forget_set, batch_size=batch_size)


def get_cifar10_retain_set(
    forget_class: int = 9, batch_size: int = 8, is_train: bool = False
) -> DataLoader:
    assert 0 <= forget_class <= 9
    dataset = datasets.CIFAR10(
        root=global_config.dataset_path,
        train=is_train,
        transform=transform,
        download=True,
    )
    dataset.targets = torch.tensor(dataset.targets)
    forget_indices = (dataset.targets != forget_class).nonzero(as_tuple=True)[0]
    forget_set = Subset(dataset, forget_indices)
    return DataLoader(dataset=forget_set, batch_size=batch_size)


# CIFAR 100
def get_cifar100_train_loader(batch_size) -> DataLoader:

    dataset = datasets.CIFAR100(
        root=global_config.dataset_path,
        train=True,
        transform=transform,
        download=True,
    )

    dataset.targets = torch.tensor(dataset.targets)

    return DataLoader(dataset=dataset, batch_size=batch_size)


def get_cifar100_val_loader(batch_size) -> DataLoader:

    dataset = datasets.CIFAR100(
        root=global_config.dataset_path,
        train=False,
        transform=transform,
        download=True,
    )
    dataset.targets = torch.tensor(dataset.targets)

    return DataLoader(dataset=dataset, batch_size=batch_size)


def get_cifar100_forget_set(
    forget_class: int = 9, batch_size: int = 8, is_train: bool = False
) -> DataLoader:
    assert 0 <= forget_class <= 99
    dataset = datasets.CIFAR100(
        root=global_config.dataset_path,
        train=is_train,
        transform=transform,
        download=True,
    )
    dataset.targets = torch.tensor(dataset.targets)
    forget_indices = (dataset.targets == forget_class).nonzero(as_tuple=True)[0]
    forget_set = Subset(dataset, forget_indices)
    return DataLoader(dataset=forget_set, batch_size=batch_size)


def get_cifar100_retain_set(
    forget_class: int = 9, batch_size: int = 8, is_train: bool = False
) -> DataLoader:
    assert 0 <= forget_class <= 99
    dataset = datasets.CIFAR100(
        root=global_config.dataset_path,
        train=is_train,
        transform=transform,
        download=True,
    )
    dataset.targets = torch.tensor(dataset.targets)
    forget_indices = (dataset.targets != forget_class).nonzero(as_tuple=True)[0]
    forget_set = Subset(dataset, forget_indices)
    return DataLoader(dataset=forget_set, batch_size=batch_size)


# SVHN
def get_svhn_train_loader(batch_size) -> DataLoader:

    dataset = datasets.SVHN(
        root=global_config.dataset_path,
        split="train",
        transform=transform,
        download=True,
    )
    return DataLoader(dataset=dataset, batch_size=batch_size)


def get_svhn_val_loader(batch_size) -> DataLoader:

    dataset = datasets.SVHN(
        root=global_config.dataset_path,
        split="test",
        transform=transform,
        download=True,
    )
    return DataLoader(dataset=dataset, batch_size=batch_size)


if __name__ == "__main__":
    f = lambda x: next(iter(x))
    # cifar10_forget_set = get_cifar10_forget_set()
    # cifar10_retain_set = get_cifar10_retain_set()

    # cifar100_forget_set = get_cifar100_forget_set()
    # cifar100_retain_set = get_cifar100_retain_set()

    # svhn_train_loader = get_svhn_train_loader(batch_size=64)
    # svhn_val_loader = get_svhn_val_loader(batch_size=64)

    def custom_transform(x):
        x["image"] = [transform(img.convert("RGB")) for img in x["image"]]
        return x

    dataloader = (
        load_dataset("datasets/pokemon-classification/data", split="test")
        .with_format("torch")
        .with_transform(custom_transform)
    )

    import code

    code.interact(local=locals())
