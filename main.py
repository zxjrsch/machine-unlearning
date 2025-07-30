from itertools import product

from loguru import logger

from data import GraphGenerator
from eval import Eval, EvalConfig
from model import SupportedVisionModels
from reporter import Reporter, ReporterConfig
from trainer import GCNTrainer, GCNTrainerConfig, Trainer, TrainerConfig
from utils_data import SupportedDatasets


def train_mnist_classifier():
    config = TrainerConfig()
    trainer = Trainer(config)
    checkpoint_path = trainer.train()
    return checkpoint_path


def generate_graph(checkpoint_path):
    graph_data_cardinaility = 2048
    vision_model_type = SupportedVisionModels.HookedMNISTClassifier
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
    dg.save_forward_backward_features()


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
