from trainer import *
from loguru import logger
from itertools import product
from time import perf_counter
import glob 


def test_gcn():

    model_architectures = [SupportedVisionModels.HookedMNISTClassifier, SupportedVisionModels.HookedResnet]
    supported_datasets = [
        # SupportedDatasets.MNIST, 
        SupportedDatasets.CIFAR10, 
        # SupportedDatasets.CIFAR100, 
        # SupportedDatasets.SVHN, 
        # SupportedDatasets.IMAGENET_SMALL, 
        # SupportedDatasets.PLANT_CLASSIFICATION, 
        # SupportedDatasets.POKEMON_CLASSIFICATION
    ]
    for (ma, ds)  in product(model_architectures, supported_datasets):
        logger.info(f'========== Traning GCN for {ma.value} on {ds.value} ==========')
        config = GCNTrainerConfig(
            vision_model_architecture=ma,
            vision_dataset=ds,
            vision_model_path=sorted(glob.glob(f'checkpoints/{ma.value}_{ds.value}/*.pt'))[-1]
        )
        trainer = GCNTrainer(config)
        a = perf_counter()
        path = trainer.train()
        b = perf_counter()
        logger.info(f'Training GCN {ma.value} on {ds.value} took {round(b-a)} seconds, checkpoint saved to {path}.')
        