from pathlib import Path

from loguru import logger

from data import GraphGenerator
from model import HookedMNISTClassifier
from trainer import Trainer, TrainerConfig


def train_mnist_classifier():
    config = TrainerConfig()
    trainer = Trainer(config)
    checkpoint_path = trainer.train()
    return checkpoint_path

def generate_graph(checkpoint_path):

    dg = GraphGenerator(model=HookedMNISTClassifier(), 
                        checkpoint_path=checkpoint_path)

    # this saves data in batches
    dg.save_forward_backward_features()

def main():
    checkpoint_path = train_mnist_classifier()
    generate_graph(checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    main()

