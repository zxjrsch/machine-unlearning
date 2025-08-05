from itertools import product
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf

import ray
from model import SupportedDatasets, SupportedVisionModels
from pipeline import Pipeline, PipelineConfig
from trainer import GCNPriorDistribution


def main():

    workding_dir = Path.cwd()
    ray_dir = workding_dir / "ray"
    ray.init(_temp_dir=str(ray_dir), runtime_env={"working_dir": str(workding_dir)})

    global_config = OmegaConf.load(workding_dir / "configs/config.yaml")
    model_architectures = [
        SupportedVisionModels.HookedMLPClassifier,
        SupportedVisionModels.HookedResnet,
    ]
    supported_datasets = [
        SupportedDatasets.SVHN,
        SupportedDatasets.CIFAR100,
        SupportedDatasets.POKEMON_CLASSIFICATION,
        SupportedDatasets.MNIST,
        SupportedDatasets.CIFAR10,
        # SupportedDatasets.IMAGENET_SMALL,
        # SupportedDatasets.PLANT_CLASSIFICATION,
    ]
    for ds, ma in product(supported_datasets, model_architectures):
        logger.info(
            f"========== Running pipeline for {ma.value} on {ds.value} =========="
        )

        config = PipelineConfig(
            model_architecture=ma,
            vision_dataset=ds,
            vision_model_epochs=2,
            vision_model_max_steps_per_epoch=2,  # adjust to something larger, like 256
            vision_model_logging_steps=1,
            vision_model_batch_size=64,
            vision_model_learning_rate=1e-3,
            use_distributed_training=True,
            num_workers=2,  # num gpus,
            device=global_config["device"],
            forget_class=0,
            graph_dataset_size=2048,
            graph_batch_size=64,
            use_sinkhorn_sampler=True,
            gcn_prior_distribution=GCNPriorDistribution.WEIGHT,
            gcn_train_steps=1,  # adjust to something larger, like 130
            gcn_learning_rate=1e-2,
            gcn_logging_steps=10,
            sft_steps=32,  # adjust to something larger, like 50
            eval_batch_size=256,
            eval_draw_plots=True,
            eval_draw_category_probabilities=True,
            topK_list=[8000],
            kappa_list=[5000, 6000, 7000],
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
