from utils_data import *
from loguru import logger

def get_forget_test_data(dataset: SupportedDatasets):
    data = get_unlearning_dataset(dataset=dataset, batch_size=10)
    forget_loader = data.get_forget_set()
    forget_batch = next(iter(forget_loader))
    logger.info(f'input batch shape: {forget_batch[0].shape} | target shape: {forget_batch[1].shape}')
    return forget_batch, data.forget_class

def test_mnist():
    forget_batch, forget_class = get_forget_test_data(SupportedDatasets.MNIST)
    assert torch.all(forget_batch[1] == forget_class*torch.ones_like(forget_batch[1]))

def test_cifar10():
    forget_batch, forget_class = get_forget_test_data(SupportedDatasets.CIFAR10)
    assert torch.all(forget_batch[1] == forget_class*torch.ones_like(forget_batch[1]))

def test_cifar100():
    forget_batch, forget_class = get_forget_test_data(SupportedDatasets.CIFAR100)
    assert torch.all(forget_batch[1] == forget_class*torch.ones_like(forget_batch[1]))

def test_svhn():
    forget_batch, forget_class = get_forget_test_data(SupportedDatasets.SVHN)
    assert torch.all(forget_batch[1] == forget_class*torch.ones_like(forget_batch[1]))