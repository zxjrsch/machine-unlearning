import glob
import os
from dataclasses import dataclass
from pathlib import Path
from random import randint
from typing import Any, Tuple, Union

import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import OmegaConf
from torch import nn
from torch_geometric.data import Data
from torchinfo import summary

from data import GraphGenerator
from model import HookedMNISTClassifier, MaskingGCN
from trainer import GCNTrainer, GCNTrainerConfig

global_config = OmegaConf.load("configs/config.yaml")

@dataclass
class EvalConfig:
    gcn_checkpoint: Path = Path(sorted(glob.glob(os.path.join('checkpoints/gcn/', '*.pt')))[0])
    classifier_checkpoint: Path = Path(sorted(glob.glob(os.path.join('checkpoints/mnist_classifier/', '*.pt')))[0])
    eval_datatset_dir: Path = Path('./eval')
    mask_layer: int = -2
    forget_digit: int = 9


class Eval:
    def __init__(self, config: EvalConfig, device=global_config.device) -> None:
        self.masking_gcn = self.load_model(config.gcn_checkpoint, MaskingGCN)
        self.classifier_model = self.load_model(config.classifier_checkpoint, HookedMNISTClassifier)
        self.device = device
        self.graph_generator = GraphGenerator(
            model = HookedMNISTClassifier(),
            checkpoint_path=config.classifier_checkpoint, 
            graph_dataset_dir=config.eval_datatset_dir,
            forget_digit=config.forget_digit,
            device=device,
            mask_layer=config.mask_layer
        )
        self.class_dataloader = self.graph_generator.data_loader
        self.trainer_config = GCNTrainerConfig(src_checkpoint=config.classifier_checkpoint, mask_K=2500)
        self.trainer = GCNTrainer(self.trainer_config)


    def generate_graph(self, idx=None) -> Tuple[Any, Data]:
        i = randint(0, 20000) if idx is None else idx
        return self.graph_generator.get_data(idx=i)
    
    def eval_single_datapoint(self) -> Tuple[int, int]:
        self.masking_gcn = self.masking_gcn.to(self.device)
        data: Data = None
        input, target, data = self.generate_graph()
        input = input.to(self.device)
        x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)
        with torch.no_grad():
            emperical_Q_logits = self.masking_gcn(x=x, edge_index = edge_index)
            mask = self.trainer.gumbel_top_k_sampling_v2(logits=emperical_Q_logits, k=self.trainer_config.mask_K).to(self.device)
            masked_model = self.trainer.mask_single_layer(mask=mask).to(self.device)
            probs = F.softmax(masked_model(input), dim=-1)
            pred = probs.argmax(1)
            return pred, target
        
        
    def eval(self, rounds = 100) -> float:
        num_correct = 0
        for _ in range(rounds):
            pred, target = self.eval_single_datapoint()
            if pred == target:
                num_correct += 1
        return num_correct / rounds * 100


    def load_model(self, path: Path, architecture: Union[HookedMNISTClassifier, MaskingGCN]) -> nn.Module:
        model: nn.Module = architecture()
        if architecture == HookedMNISTClassifier:
            model = torch.compile(model)
        model.load_state_dict(torch.load(path))
        return model
    
    def inspect_models(self) -> None:
        summary(self.masking_gcn)
        summary(self.classifier_model)
    
if __name__ == '__main__':
    config = EvalConfig()
    eval = Eval(config=config)
    # eval.inspect_models()
    percent = eval.eval()
    logger.info(f'percent {percent}')