from loguru import logger

from utils_data import *


def get_forget_test_data(dataset: SupportedDatasets):
    data = get_unlearning_dataset(dataset=dataset, batch_size=10)
    forget_loader = data.get_forget_set()
    forget_batch = next(iter(forget_loader))
    logger.info(
        f"input batch shape: {forget_batch[0].shape} | target shape: {forget_batch[1].shape}"
    )
    return forget_batch, data.forget_class


def test_mnist():
    forget_batch, forget_class = get_forget_test_data(SupportedDatasets.MNIST)
    assert torch.all(forget_batch[1] == forget_class * torch.ones_like(forget_batch[1]))


def test_cifar10():
    forget_batch, forget_class = get_forget_test_data(SupportedDatasets.CIFAR10)
    assert torch.all(forget_batch[1] == forget_class * torch.ones_like(forget_batch[1]))


def test_cifar100():
    forget_batch, forget_class = get_forget_test_data(SupportedDatasets.CIFAR100)
    assert torch.all(forget_batch[1] == forget_class * torch.ones_like(forget_batch[1]))


def test_svhn():
    forget_batch, forget_class = get_forget_test_data(SupportedDatasets.SVHN)
    assert torch.all(forget_batch[1] == forget_class * torch.ones_like(forget_batch[1]))


def test_imagenet1k():
    forget_batch, forget_class = get_forget_test_data(SupportedDatasets.IMAGENET_SMALL)
    assert torch.all(forget_batch[1] == forget_class * torch.ones_like(forget_batch[1]))


def test_plant():
    forget_batch, forget_class = get_forget_test_data(
        SupportedDatasets.PLANT_CLASSIFICATION
    )
    assert torch.all(forget_batch[1] == forget_class * torch.ones_like(forget_batch[1]))


def test_pokemon():
    forget_batch, forget_class = get_forget_test_data(
        SupportedDatasets.POKEMON_CLASSIFICATION
    )
    assert torch.all(forget_batch[1] == forget_class * torch.ones_like(forget_batch[1]))


def test_pokemon_loader():
    pokemon_ds = MIMU_pokemon(batch_size=1)
    pokemon_dl = pokemon_ds.get_single_class(class_id=0)
    # batch = next(iter(pokemon_dl))
    # logger.info(f'Pokemon single class loader batch shape: {batch[0].shape} | {1}')
    # logger.info(batch)


def test_mnist_count_numel():
    mnist_ds = MIMU_mnist(batch_size=1, forget_class=0)
    mnist_dl = mnist_ds.get_single_class(
        class_id=0
    )  # looger currently commented out in get_single_class method
