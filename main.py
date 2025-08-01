from itertools import product

from loguru import logger

from pipeline import PipelineConfig, Pipeline
from model import SupportedDatasets, SupportedVisionModels

from omegaconf import OmegaConf


def main():
    global_config = OmegaConf.load("configs/config.yaml")

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
        logger.info(f'========== Running pipeline for {ma.value} on {ds.value} ==========')

        config = PipelineConfig(
            model_architecture=ma,
            vision_dataset=ds,
            vision_model_epochs=2,
            vision_model_max_steps_per_epoch=1, # adjust to something larger, like 256 
            vision_model_logging_steps=10,
            vision_model_batch_size=64,
            vision_model_learning_rate=1e-3,
            device=global_config['device'],
            forget_class=0,
            graph_dataset_size=2048,
            graph_batch_size=64,
            gcn_train_steps=1,  # adjust to something larger, like 130
            gcn_learning_rate=1e-2,
            gcn_logging_steps=10,
            sft_steps=1, # adjust to something larger, like 50
            eval_batch_size=256,
            eval_draw_plots= True,
            eval_draw_category_probabilities=True, 
            topK_list=[8000],
            kappa_list=[7000],

            # # these following optionals can be genereated by the pipeline 
            # # when it is run in full but can also be passed in 
            # trained_vision_model_path: Optional[Path] = None
            # graph_dir: Optional[Path] = None
            # gcn_path: Optional[Path] = None
        )
        pipeline = Pipeline(config)
        metrics_dict_array = pipeline.run()

if __name__ == "__main__":
    main()
