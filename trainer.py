from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Type, Union

import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from model import HookedMNISTClassifier


@dataclass
class TrainerConfig:
    architecture: Type[nn.Module] = HookedMNISTClassifier
    batch_size: int = 256
    checkpoint_dir: Union[Path, str] = Path('./checkpoints')
    data_path: Union[Path, str] = Path('./datasets')
    model_name: str = 'mnist_classifier'
    epochs: int = 1
    lr: float = 1e-3
    logging_steps: int = 10
    wandb = None


class Trainer:
    def __init__(self, config: TrainerConfig) -> None:
        self.config = config
        self.model: nn.Module = torch.compile(config.architecture())
        self.criterion = nn.CrossEntropyLoss()
        self.train_data = self.get_train_dataloader()
        self.val_data = self.get_val_dataloader()

        # TODO validate path existence

    def get_train_dataloader(self) -> DataLoader:
        train_data = MNIST(root=self.config.data_path, train=True, transform=ToTensor(), download=True)
        train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True, drop_last=True, pin_memory=True)

        logger.info(f'Batch size: {self.config.batch_size} | # Train batches: {len(train_loader)}')

        return train_loader
    
    def get_val_dataloader(self) -> DataLoader:

        val_data = MNIST(root=self.config.data_path, train=False, transform=ToTensor(), download=True)
        val_loader = DataLoader(val_data, batch_size=self.config.batch_size, shuffle=True, drop_last=True, pin_memory=True)

        logger.info(f'Batch size: {self.config.batch_size} | # Validation batches: {len(val_loader)}')

        return val_loader

    
    def train(self, device='cuda:1') -> Path:
        """Returns checkpoint path."""
        adam_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)

        for i in range(self.config.epochs):

            self.model.train()
            self.model = self.model.to(device)
            for j, (input, target) in enumerate(self.train_data):
                input = input.to(device)
                target = target.to(device)

                preds = self.model(input)
                train_loss: torch.Tensor = self.criterion(preds, target)

                train_loss.backward()
                adam_optimizer.step()
                adam_optimizer.zero_grad()

                if j % self.config.logging_steps == 0:
                    logger.info(f'Epoch {i+1} | Step {j} | Loss {round(train_loss.item(), 2)}')
                    self.val()
                    self.model.train()

        checkpoint_path: Path = self.checkpoint()
        logger.info(f'Checkpoints saved to {checkpoint_path}')
        logger.info('Training complete.')
        return checkpoint_path

    def val(self, device='cuda:1') -> None:
        self.model = self.model.to(device)
        self.model.eval()
        test_loss, num_batches = 0, len(self.val_data)
        score, total = 0, len(self.val_data.dataset)
        with torch.no_grad():
            for i, (input, target) in enumerate(self.val_data):
                input = input.to(device)
                target = target.to(device)
                
                preds: torch.Tensor = self.model(input)
                test_loss += self.criterion(preds, target).item()
                score += (preds.argmax(1) == target).type(torch.float).sum().item()

            test_loss /= num_batches
            score /= total

        logger.info(f'Test loss {round(test_loss, 5)} | Score {round(100*score, 1)} %')

    def checkpoint(self) -> Path:
        """Returns checkpoint path."""
        d: Union[str, Path] = self.config.checkpoint_dir
        if type(d) is str:
            d = Path(d)
        d.mkdir(exist_ok=True)

        now = datetime.now().strftime("%Y_%m_%d_%H_%M")
        file_name = self.config.model_name + '_' + now + '.pt'
        
        checkpoint_path = d / file_name
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path