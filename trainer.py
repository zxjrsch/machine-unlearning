import glob
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Type, Union

import torch
from loguru import logger
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from data import GCNBatch, GraphGenerator
from model import HookedMNISTClassifier, MaskingGCN


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

    
    def train(self, device='cuda:1', step_limit: Union[int, None]=3) -> Path:
        """Returns checkpoint path."""
        adam_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)

        for i in range(self.config.epochs):

            self.model.train()
            self.model = self.model.to(device)
            enumerator = list(enumerate(self.train_data))
            if step_limit is not None:
                enumerator = enumerator[:step_limit]
            for j, (input, target) in enumerator:
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
    
@dataclass
class GCNTrainerConfig:
    src_checkpoint: Path 
    lr=0.01
    weight_decay=5e-4
    steps=3

class GraphDataLoader:
    def __init__(self, graph_data_dir: Path = Path('datasets/Graphs')):
        self.graph_data_dir = graph_data_dir
        self.edge_matrix = self.load_edge_matrix()
        self.batch_paths = sorted(glob.glob(os.path.join(self.graph_data_dir, 'batch_*.pt')))
        self.next_batch_path = 0

    def load_edge_matrix(self) -> Tensor:
        file_path = self.graph_data_dir / 'graph_edge_matrix.pt'
        return torch.load(file_path)
    
    def load_weight_feature(self) -> Tensor:
        file_path = self.graph_data_dir / 'flattened_model_weights.pt'
        return torch.load(file_path)

    def next(self) -> Tuple[GCNBatch, Tensor]:
        """Returns batched feature tensor, and edge matrix."""
        i = self.next_batch_path % len(self.batch_paths)
        file_path = self.batch_paths[i]
        gcn_batch = torch.load(file_path, weights_only=False)
        logger.info(f'Getting batch {self.next_batch_path} === {i} mod {len(self.batch_paths)}')
        self.next_batch_path += 1
        return gcn_batch, self.edge_matrix

class GCNTrainer:

    def __init__(self, config: GCNTrainerConfig) -> None:
        self.config = config
        self.src_model: HookedMNISTClassifier = self.load_src_model()
        self.weight_vector = self.vectorize_model()
        self.src_model_layer_dims = self.get_model_signature()
        self.src_model_layer_shapes = self.src_model.dim_array + [self.src_model.out_dim]
        self.src_model_dim = self.weight_vector.numel()
        self.gcn = MaskingGCN()
        self.graph_data_loader = GraphDataLoader()
        self.device = 'cuda:1'
    
    def load_src_model(self) -> nn.Module:
        model: HookedMNISTClassifier = HookedMNISTClassifier()
        model = torch.compile(model)
        model.load_state_dict(torch.load(self.config.src_checkpoint, weights_only=True))
        return model

    def vectorize_model(self) -> Tensor:
        trainable_layers = [torch.flatten(p) for p in self.src_model.parameters() if p.requires_grad]
        return torch.cat(trainable_layers)

    def get_model_signature(self) -> List[int]:
        return [p.numel() for p in self.src_model.parameters() if p.requires_grad]

    def mask_model(self, mask: Tensor) -> nn.Module:
        # assumes HookedMNISTClassifier used so parameters are matrices
        model_vect = torch.mul(mask, self.weight_vector)

        i, layers = 0, []
        for k in range(len(self.src_model_layer_dims)):
            n, m = self.src_model_layer_shapes[k:k+2]
            assert m*n == self.src_model_layer_dims[k]
            j = i + self.src_model_layer_dims[k]
            layer_vect = model_vect[i: j]
            # logger.info(f'{layer_vect.shape}, {m} x {n} = {m*n}')
            layers.append(layer_vect.unflatten(dim=0, sizes=(m, n)))
            i = j

        state_dict = self.src_model.state_dict()
        i = 0
        for key in state_dict.keys():
            state_dict[key] = layers[i]
            i += 1

        model: HookedMNISTClassifier = torch.compile(HookedMNISTClassifier())
        return model.load_state_dict(state_dict)

    def train(self) -> None:
        adam_optimizer = torch.optim.Adam(self.gcn.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.gcn = self.gcn.to(self.device)
        self.gcn.train()

        for s in range(self.config.steps):
            gcn_batch, edge_matrix = self.graph_data_loader.next()
            feature_batch = gcn_batch.feature_batch.to(self.device)
            input_batch = gcn_batch.input_batch.to(self.device)
            target_batch = gcn_batch.target_batch.to(self.device)

            edge_matrix = edge_matrix.to(self.device)

            preds = self.gcn(x=feature_batch, edge_index=edge_matrix)

            # train_loss: Tensor = None
            # train_loss.backward()
            # adam_optimizer.step()
            # adam_optimizer.zero_grad()

            # logger.info(f'Step {s} | loss {train_loss.item()}')

            # sanity check 
            logger.info(preds.shape)
            logger.info(torch.sum(preds, dim=1).detach())
        logger.info('Training complete.')

    def loss_fn(self):
        pass