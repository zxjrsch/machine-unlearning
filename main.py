from itertools import product
from multiprocessing import Process
from pathlib import Path

import trackio
from art import text2art
from loguru import logger
from omegaconf import OmegaConf

import ray
from model import SupportedDatasets, SupportedVisionModels
from pipeline import Pipeline, PipelineConfig
from reporter import *
from trainer import GCNPriorDistribution, SFTModes

resnet_topK = 62_000
mlp_topK = 8_000

resnet_kappa_array = [7_000, 39_000, 45_000, 52_000, 59_000]
mlp_kappa_array = [1000, 4_900, 5_700, 6_000, 7_000]

model_architectures = [
    SupportedVisionModels.HookedResnet,
    SupportedVisionModels.HookedMLPClassifier,
]

supported_datasets = [
    SupportedDatasets.PLANT_CLASSIFICATION,
    SupportedDatasets.CIFAR100,
    SupportedDatasets.IMAGENET_SMALL,
    SupportedDatasets.SVHN,
    SupportedDatasets.MNIST,
    SupportedDatasets.CIFAR10,
]


def get_epochs(dataset: SupportedDatasets):
    if dataset in [
        SupportedDatasets.IMAGENET_SMALL,
        SupportedDatasets.PLANT_CLASSIFICATION,
        SupportedDatasets.CIFAR100,
    ]:
        return 8
    else:
        return 2


def view_training():
    p = Process(target=trackio.show)
    p.start()
    return p


def plot():
    config = ReporterConfig()
    reporter = Reporter(config)
    reporter.plot()


def generate_tables():
    config = LaTeXTableGeneratorConfig()
    table_generator = LaTeXTableGenerator(config)

    # ResNet
    for kappa in resnet_kappa_array:
        table_generator.generate_tables(
            vision_models=[SupportedVisionModels.HookedResnet],
            datasets=supported_datasets,
            topK=resnet_topK,
            kappa=kappa,
            genereate_csv=False,
        )
    # MLP
    for kappa in mlp_kappa_array:
        table_generator.generate_tables(
            vision_models=[SupportedVisionModels.HookedMLPClassifier],
            datasets=supported_datasets,
            topK=mlp_topK,
            kappa=kappa,
            genereate_csv=False,
        )


def inclusion_exclusion(
    run_experiments: bool = True,
    start_dashboard: bool = True,
    plot_graphs: bool = True,
    produce_tables: bool = True,
):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            crashed: bool = False
            if start_dashboard:
                dashboard = view_training()
                if not run_experiments:
                    logger.info(
                        f"Starting dashboard in independent process with PID {dashboard.pid}."
                    )

            if run_experiments:
                try:
                    fn()
                except Exception as e:
                    logger.info(text2art("\n Uh   oh,   something   broke!"))
                    logger.exception(e)
                    crashed = True

            if start_dashboard and run_experiments:
                dashboard.terminate()
                dashboard.join()
                exit(1)

            if plot_graphs and not crashed:
                plot()

            if produce_tables and not crashed:
                generate_tables()

        return wrapper

    return decorator


@inclusion_exclusion(
    run_experiments=True, start_dashboard=True, plot_graphs=True, produce_tables=True
)
def main():
    working_dir = Path.cwd()
    ray_dir = working_dir / "ray"
    ray.init(
        _temp_dir=str(ray_dir),
        runtime_env={
            "working_dir": str(working_dir),
        },
    )

    global_config = OmegaConf.load(working_dir / "configs/config.yaml")

    for ds, ma in product(supported_datasets, model_architectures):
        logger.info(
            f" | -------- Running pipeline for {ma.value} on {ds.value}  -------- "
        )
        config = PipelineConfig(
            model_architecture=ma,
            vision_dataset=ds,
            vision_model_epochs=get_epochs(ds),
            vision_model_max_steps_per_epoch=1024 * 16,
            vision_model_logging_steps=1024,
            vision_model_batch_size=512,
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
            gcn_train_steps=32,
            gcn_learning_rate=1e-2,
            gcn_weight_decay=5e-4,
            gcn_logging_steps=16,
            sft_mode=SFTModes.Randomize_Forget,
            sft_steps=32,
            eval_batch_size=256,
            eval_draw_plots=True,
            eval_draw_category_probabilities=True,
            eval_metrics_base_path=Path.cwd() / "metrics_and_plots",
            topK_list=[
                resnet_topK if ma == SupportedVisionModels.HookedResnet else mlp_topK
            ],
            kappa_list=(
                resnet_kappa_array
                if ma == SupportedVisionModels.HookedResnet
                else mlp_kappa_array
            ),
            working_dir=Path.cwd(),
        )
        pipeline = Pipeline(config)

        # run from scratch
        pipeline.run()

        # # If vision model has been previously trained, or if the model is trained and graph has been generated
        # pipeline.run(
        #     trained_vision_model_path="vision_checkpoints/HookedResnet_PLANT_CLASSIFICATION_d6f/model.pt",
        #     graph_dir="graphs/HookedResnet_PLANT_CLASSIFICATION",
        # )

    logger.info(f" | -------- Finished running main() pipeline  -------- ")


if __name__ == "__main__":
    main()
