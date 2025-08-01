import glob
from itertools import product
from time import perf_counter

from loguru import logger

from trainer import *


def test_vision_trainer():

    model_architectures = [
        SupportedVisionModels.HookedMLPClassifier,
        SupportedVisionModels.HookedResnet,
    ]
    supported_datasets = [
        # SupportedDatasets.MNIST,
        SupportedDatasets.CIFAR10,
        # SupportedDatasets.CIFAR100,
        # SupportedDatasets.SVHN,
        # SupportedDatasets.IMAGENET_SMALL,
        # SupportedDatasets.PLANT_CLASSIFICATION,
        # SupportedDatasets.POKEMON_CLASSIFICATION
    ]
    for ma, ds in product(model_architectures, supported_datasets):
        logger.info(f"========== running {ma.value} on {ds.value} ==========")
        config = VisionModelTrainerConfig(
            architecture=ma, vision_dataset=ds, steps=1, logging_steps=1
        )
        trainer = VisionModelTrainer(config)
        a = perf_counter()
        path = trainer.train()
        b = perf_counter()
        logger.info(
            f"Training {ma.value} on {ds.value} took {round(b-a)} seconds, checkpoint saved to {path}."
        )


def test_sft_trainer():
    model_architectures = [
        SupportedVisionModels.HookedMLPClassifier,
        SupportedVisionModels.HookedResnet,
    ]
    supported_datasets = [
        # SupportedDatasets.MNIST,
        SupportedDatasets.CIFAR10,
        # SupportedDatasets.CIFAR100,
        # SupportedDatasets.SVHN,
        # SupportedDatasets.IMAGENET_SMALL,
        # SupportedDatasets.PLANT_CLASSIFICATION,
        # SupportedDatasets.POKEMON_CLASSIFICATION
    ]
    for ma, ds in product(model_architectures, supported_datasets):
        logger.info(f"========== running {ma.value} on {ds.value} ==========")
        config = SFTConfig(
            architecture=ma,
            vision_dataset=ds,
            steps=1,
            logging_steps=1,
            finetuning_mode=SFTModes.Randomize_Forget_And_Finetune_Retain,
            original_model_path=sorted(
                glob.glob(f"checkpoints/{ma.value}_{ds.value}/*.pt")
            )[-1],
        )
        trainer = SFTTrainer(config)
        a = perf_counter()
        trainer.finetune()
        b = perf_counter()
        logger.info(f"Training {ma.value} on {ds.value} took {round(b-a)} seconds.")
