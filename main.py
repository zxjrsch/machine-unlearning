import glob
import os
from pathlib import Path

import torch
from loguru import logger

from data import GraphGenerator
from model import HookedMNISTClassifier
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

    data = dg.get_data()
    logger.info(data)

    # this saves data in batches
    dg.save_forward_backward_features(limit=8)

def train_gcn(src_checkpoint):
    config = GCNTrainerConfig(src_checkpoint=src_checkpoint)
    trainer = GCNTrainer(config=config)
    trainer.train()

    # test masking
    # current_model_dim = 365184
    # trainer.mask_model(mask=torch.rand(current_model_dim))

    # data_loader = GraphDataLoader()
    # for i in range(5):
    #     data_loader.next()



def main():
    checkpoint_path = train_mnist_classifier()

    # checkpoint_path = Path(sorted(glob.glob(os.path.join('checkpoints', '*.pt')))[0])

    generate_graph(checkpoint_path=checkpoint_path)

    train_gcn(checkpoint_path)


if __name__ == "__main__":
    main()

