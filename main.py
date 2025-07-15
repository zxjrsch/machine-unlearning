import glob
import os
from pathlib import Path

import torch
from loguru import logger

from data import GraphGenerator
from model import HookedMNISTClassifier
from trainer import (GCNTrainer, GCNTrainerConfig, GraphDataLoader, Trainer,
                     TrainerConfig)

from eval import Eval, EvalConfig

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

def train_gcn(src_checkpoint):
    config = GCNTrainerConfig(src_checkpoint=src_checkpoint, mask_K=2500)
    trainer = GCNTrainer(config=config)
    ckpt_path = trainer.train()
    return ckpt_path

    # # test masking
    # current_model_dim = 8192
    # trainer.mask_single_layer(mask=torch.rand(current_model_dim))

    # data_loader = GraphDataLoader()
    # for i in range(5):
    #     data_loader.next()

def eval(gcn_path, classifier_path):
    # for real usecase 
    config = EvalConfig(gcn_path=gcn_path, classifier_path=classifier_path)
    eval = Eval(config)
    eval.eval()

def main():

    # checkpoint_path = Path(sorted(glob.glob(os.path.join('checkpoints/mnist_classifier', '*.pt')))[0])
    classifier_checkpoint_path = train_mnist_classifier()
    generate_graph(checkpoint_path=classifier_checkpoint_path)
    gcn_checkpoint_path = train_gcn(classifier_checkpoint_path)
    eval(gcn_path=gcn_checkpoint_path, classifier_path=classifier_checkpoint_path)


if __name__ == "__main__":
    main()

