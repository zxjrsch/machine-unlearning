import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor

global_config = OmegaConf.load("configs/config.yaml")

transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


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


if __name__ == "__main__":
    f = lambda x: next(iter(x))
    cifar10_forget_set = get_cifar10_forget_set()
    cifar10_retain_set = get_cifar10_retain_set()

    cifar100_forget_set = get_cifar100_forget_set()
    cifar100_retain_set = get_cifar100_retain_set()

    # code.interact(local=locals())
