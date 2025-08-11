from time import perf_counter

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


def test_performance():
    mimu_datasets = [
        MIMU_mnist,
        MIMU_cifar10,
        MIMU_cifar100,
        MIMU_svhn,
        MIMU_imagenet_small,
    ]
    for d in mimu_datasets:
        ds = d()
        t0 = perf_counter()
        l = ds.get_single_class(class_id=0, is_train=True)
        ds.get_retain_set()
        t1 = perf_counter()
        logger.info(f"{ds.dataset_name} get_single_class time {t1-t0}")


def test_pokemon_performance():
    ds = MIMU_pokemon()
    test_ds = ds.get_dataset(split="test")
    class_str = test_ds.features["label"].names
    logger.info(class_str)
    unique = len(set(class_str))
    logger.info(f"unique classes = {unique}")
    # val_loader = ds.get_val_loader()

    # t0 = perf_counter()

    # single_class = load_dataset(
    # ds.dataset_path + f"/train/{class_str[0]}",
    # split=(
    #     f"train[:{5}]"
    # ),
    # # num_proc=num_cpu,
    # )
    # t1 = perf_counter()
    # logger.info(f'{ds.dataset_name} get_single_class time {t1-t0}')
    # logger.info(single_class)


def test_representatives():
    # mimu_datasets = [MIMU_mnist, MIMU_cifar10, MIMU_cifar100, MIMU_svhn, MIMU_imagenet_small, MIMU_pokemon, MIMU_plant]
    # for ds in mimu_datasets:
    #     d: UnlearningDataset = ds()
    #     t0 = perf_counter()
    #     reps = d.get_representatives()
    #     t1 = perf_counter()

    #     logger.info(f'{d.dataset_name} | num of reps {len(reps)} | {t1-t0}')
    d = MIMU_plant()
    t0 = perf_counter()
    # d.get_single_class(class_id=0)
    retain = d.get_retain_set()
    next(iter(retain))

    # o = d.get_subclass_string_names()
    # logger.info(o)
    t1 = perf_counter()
    logger.info(f"{d.dataset_name} | {t1-t0}")
