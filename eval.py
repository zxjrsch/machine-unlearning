import copy
import glob
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import torch
from loguru import logger
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from data import GraphGenerator
from model import HookedMNISTClassifier, MaskingGCN
from trainer import gumbel_top_k_sampling_v2

global_config = OmegaConf.load("configs/config.yaml")

@dataclass
class EvalConfig:
    gcn_path: Path # =  Path(sorted(glob.glob(os.path.join('checkpoints/gcn/', '*.pt')))[0]) 
    classifier_path: Path # = Path(sorted(glob.glob(os.path.join('checkpoints/mnist_classifier/', '*.pt')))[0])
    data_path: Path = Path('datasets')
    graph_data_path: Path = Path('eval/Graphs')
    metrics_path: Path = Path('eval/Metrics')
    forget_digit = 9
    batch_size = 16
    device = global_config.device
    mask_layer = -2
    topK = 2500

class Eval:
    def __init__(self, config: EvalConfig, draw_eval_plots: bool=True):
        self.config = config
        self.gcn = self.load_gcn()
        self.classifier = self.load_classifier()
        self.graph_generator = self.load_graph_generator()
        self.draw_eval_plots = draw_eval_plots

    def load_graph_generator(self) -> GraphGenerator:
        return GraphGenerator(
            model=self.classifier,
            graph_dataset_dir=self.config.graph_data_path,
            process_save_batch_size = 1,     # we want per-sample grad
            forget_digit=self.config.forget_digit,
            mask_layer=self.config.mask_layer,
            device=self.config.device, 
            save_redundant_features=False
        )

    def load_gcn(self) -> MaskingGCN:
        model = MaskingGCN()
        model.load_state_dict(torch.load(self.config.gcn_path))
        return model 
    
    def load_classifier(self) -> HookedMNISTClassifier:
        model = HookedMNISTClassifier()
        model = torch.compile(model)
        model.load_state_dict(torch.load(self.config.classifier_path))
        return model

    def mnist_class_representatives(self) -> List[Tuple[Tensor, Tensor]]:
        dataset = MNIST(root=self.config.data_path, train=False, transform=ToTensor(), download=True)
        dataset = iter(DataLoader(dataset=dataset, batch_size=1))
        representatives = []
        for d in range(10):
            data = next(dataset)
            while data[1].item() != d:
                data = next(dataset)
            representatives.append(data)
        return representatives
    
    def mnist_forget_set(self) -> DataLoader:
        dataset = MNIST(root=self.config.data_path, train=False, transform=ToTensor(), download=True)
        forget_indices = (dataset.targets == self.config.forget_digit).nonzero(as_tuple=True)[0]
        forget_set = Subset(dataset, forget_indices)
        return DataLoader(dataset=forget_set, batch_size=self.config.batch_size)        
        
    def mnist_retain_set(self) -> DataLoader:
        dataset = MNIST(root=self.config.data_path, train=False, transform=ToTensor(), download=True)
        retain_indices = (dataset.targets != self.config.forget_digit).nonzero(as_tuple=True)[0]
        retain_set = Subset(dataset, retain_indices)
        return DataLoader(dataset=retain_set, batch_size=self.config.batch_size)       
    
    def compute_representative_masks(self) -> List[Tuple[Tensor, int]]:
        """Gets masks for all the classes."""
        reps = self.mnist_class_representatives()

        # a list for all digits
        node_features = self.graph_generator.get_representative_features(reps)
        edge_matrix = self.graph_generator.edge_matrix.to(self.config.device)
        
        self.gcn.eval()
        self.gcn = self.gcn.to(self.config.device)

        mask_label_list = []
        with torch.no_grad():
            for features, cls in node_features:
                Q_logits = self.gcn(x=features, edge_index=edge_matrix)
                mask = gumbel_top_k_sampling_v2(logits=Q_logits, k=self.config.topK, temperature=1, eps=1e-10)
                mask_label_list.append((mask, cls.item()))

        return mask_label_list
    
    def get_model_mask(self) -> Tensor:
        mask_label_list = self.compute_representative_masks()

        forget_mask = None
        retain_mask = None

        for mask, digit in mask_label_list:
            if digit == self.config.forget_digit:
                forget_mask = mask
            else:
                retain_mask = mask if retain_mask is None else retain_mask + mask

        mask = torch.clamp(forget_mask - retain_mask, min=0)

        # return 1 for weight values which are are important to predict retain set but not forget data
        return (mask == 0).float()
    
    def model_mask_forget_class(self) -> HookedMNISTClassifier:
        forget_class_mask = None
        for mask, digit in self.compute_representative_masks():
            if digit == self.config.forget_digit:
                forget_class_mask = mask
                break
        return self.get_masked_model(custom_mask=forget_class_mask)

    def get_masked_model(self, custom_mask: Union[Tensor, None]=None) -> HookedMNISTClassifier:
        """Single layer masking."""
        mask: Tensor = self.get_model_mask()
        mask = mask.to(self.config.device)
        weight_vector = self.graph_generator.weight_feature.to(self.config.device)
        layer_vect = torch.mul(mask, weight_vector)

        m,n = [p for p in self.classifier.parameters() if p.requires_grad][self.config.mask_layer].shape
        layer_matrix = layer_vect.unflatten(dim=0, sizes=(m,n))

        state_dict = copy.deepcopy(self.classifier.state_dict())
        key = list(state_dict.keys())[self.config.mask_layer]
        state_dict[key] = layer_matrix

        model: HookedMNISTClassifier = torch.compile(HookedMNISTClassifier())
        model.load_state_dict(state_dict)
        return model
    
    def inference(self, model: HookedMNISTClassifier, data_loader: DataLoader, is_forget_set: bool = True, description: str = None) -> Any:
        model = model.to(self.config.device)
        model.eval()

        test_loss, num_batches = 0, len(data_loader)
        score, total = 0, len(data_loader.dataset)
        with torch.no_grad():
            for i, (input, target) in enumerate(data_loader):
                input = input.to(self.config.device)
                target = target.to(self.config.device)
                
                preds: torch.Tensor = model(input)
                test_loss += self.graph_generator.loss_fn(preds, target).item()
                score += (preds.argmax(1) == target).type(torch.float).sum().item()

            test_loss /= num_batches
            test_loss = round(test_loss, 5)
            score /= total
            score = round(100*score, 1) # percentage

        eval_set = 'forget set' if is_forget_set else 'retain set'
        logger.info(f'Exp: {description} | Test loss on {eval_set} {test_loss} | Score {score} %')

        return {
            'experiment': description,
            'eval data': eval_set,
            'test loss': test_loss,
            'score': score
        }
    
    def draw_visualization(self, loss: List[float], score: List[float], experiment: str = '') -> None:
        categories = ['Before Masking', 'After Masking']

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

        # Plot loss
        axs[0].bar(categories, loss, color='blue')
        axs[0].set_title('Test Loss')
        axs[0].set_ylabel('Loss')

        # Plot score
        axs[1].bar(categories, score, color='blue')
        axs[1].set_title('Score')
        axs[1].set_ylabel('Score')
        plt.suptitle(f'{experiment} (top-{self.config.topK})')
        plt.tight_layout()

        self.config.metrics_path.mkdir(parents=True, exist_ok=True)
        save_path = self.config.metrics_path / f'{experiment}_top_{self.config.topK}.png'
        plt.savefig(save_path)
        plt.show()

    def eval_unlearning(self) -> Dict:
        """Evaluate model before and after MIMU masking on forget set."""

        logger.info('..... eval unlearning .....')
        before_masking_eval_metrics = self.inference(
            description='no masking on forget set',
            model=self.classifier,
            data_loader=self.mnist_forget_set(),
            is_forget_set=True
        )

        after_masking_eval_metrics = self.inference(
            description='mimu on forget set',
            model=self.get_masked_model(),
            data_loader=self.mnist_forget_set(),
            is_forget_set=True
        )

        if self.draw_eval_plots:

            self.draw_visualization(
                loss=[before_masking_eval_metrics['test loss'], after_masking_eval_metrics['test loss']],
                score=[before_masking_eval_metrics['score'], after_masking_eval_metrics['score']],
                experiment='eval_unlearning_on_forget_set'
            )

        return {
            'experiment': 'eval_unlearning',
            # 'top-K value': self.config.topK,
            'eval data': 'forget_set',
            'before masking loss': before_masking_eval_metrics['test loss'],
            'after masking loss': after_masking_eval_metrics['test loss'],
            'before masking score': before_masking_eval_metrics['score'],
            'after masking score': after_masking_eval_metrics['score']
        }

    def eval_performance_degradation(self) -> Dict:
        """Eval model before and after MIMU masking on retain set."""

        logger.info('..... eval performance degradation .....')

        before_masking_eval_metrics = self.inference(
            description='no masking on retain set',
            model=self.classifier,
            data_loader=self.mnist_retain_set(),
            is_forget_set=False
        )

        after_masking_eval_metrics = self.inference(
            description='mimu on retain set',
            model=self.get_masked_model(),
            data_loader=self.mnist_retain_set(),
            is_forget_set=False
        )

        if self.draw_eval_plots:

            self.draw_visualization(
                loss=[before_masking_eval_metrics['test loss'], after_masking_eval_metrics['test loss']],
                score=[before_masking_eval_metrics['score'], after_masking_eval_metrics['score']],
                experiment='eval_performance_degradation_on_retain_set'
            )

        return {
            'experiment': 'eval_performance_degradation',
            # 'top-K value': self.config.topK,
            'eval data': 'retain_set',
            'before masking loss': before_masking_eval_metrics['test loss'],
            'after masking loss': after_masking_eval_metrics['test loss'],
            'before masking score': before_masking_eval_metrics['score'],
            'after masking score': after_masking_eval_metrics['score']
        }

    def eval_mask_efficacy(self) -> Dict:
        """Eval whether mask identified weights important for predicting desired forget class."""

        logger.info('..... eval mask efficacy .....')

        before_masking_eval_metrics = self.inference(
            description='no masking on forget set',
            model=self.classifier,
            data_loader=self.mnist_forget_set(),
            is_forget_set=True
        )

        after_masking_eval_metrics = self.inference(
            description='pure class mask on forget set',
            model=self.model_mask_forget_class(),
            data_loader=self.mnist_forget_set(),
            is_forget_set=True
        )

        if self.draw_eval_plots:

            self.draw_visualization(
                loss=[before_masking_eval_metrics['test loss'], after_masking_eval_metrics['test loss']],
                score=[before_masking_eval_metrics['score'], after_masking_eval_metrics['score']],
                experiment='eval_performance_degradation_on_foget_set'
            )

        return {
            'experiment': 'eval_performance_degradation',
            # 'top-K value': self.config.topK,
            'eval data': 'forget_set',
            'before masking loss': before_masking_eval_metrics['test loss'],
            'after masking loss': after_masking_eval_metrics['test loss'],
            'before masking score': before_masking_eval_metrics['score'],
            'after masking score': after_masking_eval_metrics['score']
        }
    
    def eval(self, save_metrics: bool = True) -> Dict:
        unlearning_metrics = self.eval_unlearning()
        performance_degradation_metrics = self.eval_performance_degradation()
        mask_efficacy_metrics = self.eval_mask_efficacy()
        metrics = {
            'id': str(uuid.uuid1()),
            'forget_digit': self.config.forget_digit,
            'num_graph_vertices': self.graph_generator.num_vertices,
            'num_graph_edges': self.graph_generator.edge_matrix.shape[1],
            'maked_layer': self.config.mask_layer,
            'top_k_value': self.config.topK,
            'unlearning_metrics': unlearning_metrics,
            'performance_degradation_metrics': performance_degradation_metrics,
            'mask_efficacy_metrics': mask_efficacy_metrics
        }
        if save_metrics:
            self.config.metrics_path.mkdir(exist_ok=True, parents=True)
            file_path = self.config.metrics_path / f'top-{self.config.topK}.jsonl'
            with open(file_path, 'a') as f:
                f.write(json.dumps(metrics) + '\n')

        return metrics

if __name__ == '__main__':
    config = EvalConfig()
    eval = Eval(config)
    metrics = eval.eval()
