from data import GraphGenerator
from model import HookedMNISTClassifier
from trainer import Trainer, TrainerConfig
from loguru import logger
from pathlib import Path


def train_mnist_classifier():
    config = TrainerConfig()
    trainer = Trainer(config)
    checkpoint_path = trainer.train()
    return checkpoint_path

def generate_graph(checkpoint_path):

    dg = GraphGenerator(model=HookedMNISTClassifier(), 
                        checkpoint_path=checkpoint_path)
    
    data = dg.get_forward_backward_features()
    logger.info(data)

def main():
    checkpoint_path = train_mnist_classifier()
    generate_graph(checkpoint_path=checkpoint_path)



if __name__ == "__main__":
    main()

