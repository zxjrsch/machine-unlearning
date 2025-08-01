from itertools import product

from loguru import logger
from omegaconf import OmegaConf

from model import SupportedDatasets, SupportedVisionModels
from pipeline import Pipeline, PipelineConfig


def main():
    global_config = OmegaConf.load("/home/claire/mimu/configs/config.yaml")
    model_architectures = [
        SupportedVisionModels.HookedResnet,
        SupportedVisionModels.HookedMLPClassifier,
    ]
    supported_datasets = [
        SupportedDatasets.CIFAR10,
        SupportedDatasets.SVHN,
        SupportedDatasets.MNIST,
        SupportedDatasets.CIFAR100,
        # SupportedDatasets.IMAGENET_SMALL,
        # SupportedDatasets.PLANT_CLASSIFICATION,
        SupportedDatasets.POKEMON_CLASSIFICATION,
    ]
    for ds, ma in product(supported_datasets, model_architectures):
        logger.info(
            f"========== Running pipeline for {ma.value} on {ds.value} =========="
        )

        config = PipelineConfig(
            model_architecture=ma,
            vision_dataset=ds,
            vision_model_epochs=2,
            vision_model_max_steps_per_epoch=64,  # adjust to something larger, like 256
            vision_model_logging_steps=10,
            vision_model_batch_size=64,
            vision_model_learning_rate=1e-3,
            device=global_config["device"],
            forget_class=0,
            graph_dataset_size=2048,
            graph_batch_size=64,
            use_sinkhorn_sampler=True,
            gcn_train_steps=30,  # adjust to something larger, like 130
            gcn_learning_rate=1e-2,
            gcn_logging_steps=10,
            sft_steps=32,  # adjust to something larger, like 50
            eval_batch_size=256,
            eval_draw_plots=True,
            eval_draw_category_probabilities=True,
            topK_list=[8000],
            kappa_list=[1000, 2000, 3000, 4000, 5000, 6000, 7000],
            # # these following optionals can be genereated by the pipeline
            # # when it is run in full but can also be passed in
            # trained_vision_model_path: Optional[Path] = None
            # graph_dir: Optional[Path] = None
            # gcn_path: Optional[Path] = None
        )
        pipeline = Pipeline(config)
        pipeline.run()


if __name__ == "__main__":
    main()
