from itertools import product
from multiprocessing import Process
from pathlib import Path

import trackio
from loguru import logger
from omegaconf import OmegaConf

import ray
from model import SupportedDatasets, SupportedVisionModels
from pipeline import Pipeline, PipelineConfig
from reporter import *
from trainer import GCNPriorDistribution, SFTModes


def main():
    working_dir = Path.cwd()
    ray_dir = working_dir / "ray"
    ray.init(
        _temp_dir=str(ray_dir),
        runtime_env={
            "working_dir": str(working_dir),
            # "env_vars": {
            #     "PYTHONPATH": '/home/claire/mimu/.venv/bin/python',
            #     # "VIRTUAL_ENV": "/home/claire/mimu/.venv"
            # },
        },
        # num_gpus=2,
    )

    global_config = OmegaConf.load(working_dir / "configs/config.yaml")
    model_architectures = [
        SupportedVisionModels.HookedResnet,
        SupportedVisionModels.HookedMLPClassifier,
    ]
    # we are dropping SupportedDatasets.POKEMON_CLASSIFICATION dataset for now
    # due to data non-uniformity
    supported_datasets = [
        SupportedDatasets.CIFAR100,
        SupportedDatasets.IMAGENET_SMALL,
        SupportedDatasets.SVHN,
        SupportedDatasets.MNIST,
        SupportedDatasets.CIFAR10,
        SupportedDatasets.PLANT_CLASSIFICATION,
    ]
    for ds, ma in product(supported_datasets, model_architectures):
        logger.info(
            f" | -------- Running pipeline for {ma.value} on {ds.value}  -------- "
        )

        config = PipelineConfig(
            model_architecture=ma,
            vision_dataset=ds,
            vision_model_epochs=1,
            vision_model_max_steps_per_epoch=1
            * 16,  # adjust to something larger, like 256
            vision_model_logging_steps=8,  # 1024,
            vision_model_batch_size=4,  # 512,  # 256,
            vision_model_learning_rate=1e-3,
            vision_model_checkpoint_dir=Path.cwd() / "vision_checkpoints",
            plot_vision_model_train_statistics=True,
            num_workers=2,  # num gpus,
            device=global_config["device"],
            forget_class=0,
            graph_dataset_size=2048,
            graph_dataset_dir=Path.cwd() / "graphs",
            gcn_checkpoint_dir=Path.cwd() / "gcn_checkpoints",
            graph_batch_size=64,
            use_sinkhorn_sampler=True,
            use_set_difference_masking_strategy=False,
            gcn_prior_distribution=GCNPriorDistribution.WEIGHT,
            gcn_train_steps=1,  # adjust to something larger, like 130
            gcn_learning_rate=1e-2,
            gcn_weight_decay=5e-4,
            gcn_logging_steps=1,
            sft_mode=SFTModes.Randomize_Forget,
            sft_steps=2,  # adjust to something larger, like 50
            eval_batch_size=256,
            eval_draw_plots=True,
            eval_draw_category_probabilities=True,
            eval_metrics_base_path=Path.cwd() / "metrics_and_plots",
            topK_list=[8000],
            kappa_list=[7000],
            working_dir=Path.cwd(),
            # # these following optionals can be genereated by the pipeline
            # # when it is run in full but can also be passed in
            # trained_vision_model_path: Optional[Path] = None
            # graph_dir: Optional[Path] = None
            # gcn_path: Optional[Path] = None
        )
        pipeline = Pipeline(config)
        pipeline.run()
        # pipeline.run(
        #     trained_vision_model_path="vision_checkpoints/HookedResnet_PLANT_CLASSIFICATION_d6f/model.pt",
        #     graph_dir="graphs/HookedResnet_PLANT_CLASSIFICATION",
        # )
        # try:
        #     pipeline.run()
        # except Exception as e:
        #     logger.info(f"Error encountered for {ma.value} on {ds.value}")
        #     logger.info(e)


def view_training():
    p = Process(target=trackio.show)
    p.start()
    return p


def plot():
    # assume frozen topK
    config = ReporterConfig()
    reporter = Reporter(config)
    reporter.plot()


def genereate_tables(topK=8000, kappa=7000):
    config = LaTeXTableGeneratorConfig()
    table_generator = LaTeXTableGenerator(config)
    table_generator.generate_tables(topK=topK, kappa=kappa, genereate_csv=False)


if __name__ == "__main__":
    # p = view_training()
    # main()
    # p.terminate()
    # p.join()
    plot()

    genereate_tables(topK=8000, kappa=7000)
