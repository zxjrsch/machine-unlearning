from data import GraphGenerator
from model import SupportedDatasets, SupportedVisionModels
from itertools import product
from loguru import logger
from time import perf_counter
import glob

def test_graph_generator():
    model_architectures = [SupportedVisionModels.HookedMLPClassifier, SupportedVisionModels.HookedResnet]
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
        logger.info(f'========== Generating graph data for {ma.value} on {ds.value} ==========')
        generator = GraphGenerator(
            vision_model_type=ma,
            unlearning_dataset=ds,
            checkpoint_path=sorted(glob.glob(f'checkpoints/{ma.value}_{ds.value}/*.pt'))[-1],
        )


        a = perf_counter()
        generator.save_forward_backward_features()
        b = perf_counter()
        logger.info(f'Generating graph data for {ma.value} on {ds.value} took {round(b-a)} seconds.')


