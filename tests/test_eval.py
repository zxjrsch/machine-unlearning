from eval import * 
from itertools import product

def test_eval():
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

        config = EvalConfig(
            vision_model=ma,
            vision_model_path=sorted(glob(f'checkpoints/{ma.value}_{ds.value}/*.pt'))[-1],
            vision_dataset=ds
        )
        eval = Eval(config)
        logger.info(eval.get_gcn_path())