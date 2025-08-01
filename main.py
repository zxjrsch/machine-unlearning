from itertools import product

from loguru import logger

from data import GraphGenerator
from eval import Eval, EvalConfig
from model import SupportedVisionModels
from reporter import Reporter, ReporterConfig
from trainer import (GCNTrainer, GCNTrainerConfig, VisionModelTrainer,
                     VisionModelTrainerConfig)
from utils_data import SupportedDatasets
from glob import glob

def eval_single_model_dataset_pair(model_architecture, vision_dataset):
    pass

def train_mnist_classifier():
    config = VisionModelTrainerConfig()
    trainer = VisionModelTrainer(config)
    checkpoint_path = trainer.train()
    return checkpoint_path


def generate_graph(checkpoint_path):
    graph_data_cardinaility = 2048
    vision_model_type = SupportedVisionModels.HookedMLPClassifier
    SupportedDatasets.MNIST
    dg = GraphGenerator(
        vision_model_type=vision_model_type,
        checkpoint_path=checkpoint_path,
        graph_data_cardinaility=graph_data_cardinaility,
    )

    # # If you wish to obtain single training data points in the form of PyG Data
    # data = dg.get_data()
    # logger.info(data)

    # this saves data in batches
    dg.genereate_graphs()


def train_gcn(src_checkpoint, topK=2500):
    config = GCNTrainerConfig(src_checkpoint=src_checkpoint, mask_K=topK)
    trainer = GCNTrainer(config=config)
    ckpt_path = trainer.train()
    return ckpt_path

    # # test masking
    # current_model_dim = 8192
    # trainer.mask_single_layer(mask=torch.rand(current_model_dim))

    # data_loader = GraphDataLoader()
    # for i in range(5):
    #     data_loader.next()


def evaluation(gcn_path, classifier_path, topK=2500, kappa=2000):
    # for real usecase
    config = EvalConfig(
        gcn_path=gcn_path, classifier_path=classifier_path, topK=topK, kappa=kappa
    )
    eval = Eval(config)
    return eval.eval()


def viz():
    config = ReporterConfig()
    reporter = Reporter(config)
    reporter.plot()


def main():

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

        config = EvalConfig(
            vision_model=ma,
            vision_model_path=sorted(glob(f'checkpoints/{ma.value}_{ds.value}/*.pt'))[-1],
            vision_dataset=ds
        )
        eval = Eval(config)
        # logger.info(eval.get_gcn_path())
        # reps = eval.get_vision_class_representatives()
        # logger.info(len(reps))
        # logger.info(type(reps))
        # logger.info(type(reps[0]))
        # logger.info(reps[0][0].shape)
        # logger.info('=======')
        # logger.info(reps[0][1])
        eval.eval()



    logger.info(f"-------- Training Classifier -------")
    # checkpoint_path = Path(sorted(glob.glob(os.path.join('checkpoints/mnist_classifier', '*.pt')))[0])
    classifier_checkpoint_path = train_mnist_classifier()

    logger.info(f"-------- Generating GCN Training Data -------")
    generate_graph(checkpoint_path=classifier_checkpoint_path)

    # full sweep
    topK_array = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    kappa_array = list(range(100, 7000 + 1, 1000))

    # partial sweep to save time
    topK_array = [8000]
    kappa_array = [7000]

    experiments = list(product(topK_array, kappa_array))

    i = 0
    for topK, kappa in experiments:
        i += 1
        logger.info(
            f"------- Starting exp {i} of {len(experiments)} | top-{topK} kappa-{kappa}  ------"
        )
        gcn_checkpoint_path = train_gcn(classifier_checkpoint_path, topK=topK)
        metrics = evaluation(
            gcn_path=gcn_checkpoint_path,
            classifier_path=classifier_checkpoint_path,
            topK=topK,
            kappa=kappa,
        )
        logger.info(f"------- Completed top-{topK} kappa-{kappa} experiment ------")

    viz()
    logger.info(f"-------- Experiment Complete -------")


if __name__ == "__main__":
    main()
