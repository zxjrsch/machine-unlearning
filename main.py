import glob
import os
from pathlib import Path

import torch
from loguru import logger

from data import GraphGenerator
from eval import Eval, EvalConfig
from model import HookedMNISTClassifier
from reporter import Reporter, ReporterConfig
from trainer import (GCNTrainer, GCNTrainerConfig, GraphDataLoader, Trainer,
                     TrainerConfig)


def train_mnist_classifier():
    config = TrainerConfig()
    trainer = Trainer(config)
    checkpoint_path = trainer.train()
    return checkpoint_path

def generate_graph(checkpoint_path):

    dg = GraphGenerator(model=HookedMNISTClassifier(), 
                        checkpoint_path=checkpoint_path)

    # # If you wish to obtain single training data points in the form of PyG Data
    # data = dg.get_data()
    # logger.info(data)

    # this saves data in batches
    dg.save_forward_backward_features(limit=2048)

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
    config = EvalConfig(gcn_path=gcn_path, classifier_path=classifier_path, topK=topK, kappa=kappa)
    eval = Eval(config)
    return eval.eval()

def viz():
    config = ReporterConfig()
    reporter = Reporter(config)
    reporter.plot()

def main():

    logger.info(f'-------- Training Classifier -------')
    # checkpoint_path = Path(sorted(glob.glob(os.path.join('checkpoints/mnist_classifier', '*.pt')))[0])
    classifier_checkpoint_path = train_mnist_classifier()

    logger.info(f'-------- Generating GCN Training Data -------')
    generate_graph(checkpoint_path=classifier_checkpoint_path)

    i = 0
    # full sweep
    topK_array = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    kappa_array = [900, 1900, 2900, 3900, 4900, 5900, 6900, 7900]

    # # partial sweep to save time
    # topK_array = [5000, 6000]
    # kappa_array = [4000, 5000]

    for topK, kappa in zip(topK_array, kappa_array):
        i += 1
        logger.info(f'------- Starting exp {i} of {len(topK_array)} | top-{topK} kappa-{kappa}  ------')
        gcn_checkpoint_path = train_gcn(classifier_checkpoint_path, topK=topK)
        metrics = evaluation(gcn_path=gcn_checkpoint_path, classifier_path=classifier_checkpoint_path, topK=topK, kappa=kappa)
        logger.info(f'------- Completed top-{topK} experiment ------')

    viz()
    logger.info(f'-------- Experiment Complete -------')


if __name__ == "__main__":
    main()

