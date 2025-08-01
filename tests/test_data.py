import glob
from itertools import product
from time import perf_counter

from loguru import logger

from data import GraphGenerator
from model import SupportedDatasets, SupportedVisionModels


def test_graph_generator():
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
        logger.info(
            f"========== Generating graph data for {ma.value} on {ds.value} =========="
        )
        generator = GraphGenerator(
            vision_model_type=ma,
            unlearning_dataset=ds,
            checkpoint_path=sorted(
                glob.glob(f"checkpoints/{ma.value}_{ds.value}/*.pt")
            )[-1],
        )

        a = perf_counter()
        generator.genereate_graphs()
        b = perf_counter()
        logger.info(
            f"Generating graph data for {ma.value} on {ds.value} took {round(b-a)} seconds."
        )
