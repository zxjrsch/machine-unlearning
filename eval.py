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
import torch.nn.functional as F
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
    metrics_path: Path = Path('eval/Metrics and Plots')
    forget_digit = 9
    batch_size = 16
    device = global_config.device
    mask_layer = -2
    topK:int = 2500
    plot_category_probabilities: bool = True

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

        # weights important for forget set prediction but not important for retain classes correspond to 1s
        mask = torch.clamp(forget_mask - retain_mask, min=0)

        # negation operation return True for weight values which are are important to predict RETAIN set but not forget data
        mask = mask == 0
        num_keep_weights = mask.sum().item()
        percent = round(num_keep_weights/self.graph_generator.num_vertices * 100, 2)

        logger.info(f'top-{self.config.topK} | target layer {self.config.mask_layer} | Retaining {num_keep_weights} ({percent} %)')

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
        if custom_mask is None:
            mask: Tensor = self.get_model_mask()
        else:
            mask = custom_mask
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

    def get_randomly_masked_model(self) -> HookedMNISTClassifier:
        mask = torch.zeros(self.graph_generator.num_vertices, dtype=torch.float32)
        num_1s = self.graph_generator.num_vertices - self.config.topK
        assert num_1s > 0
        indices = torch.randperm(self.graph_generator.num_vertices)[:num_1s]
        mask[indices] = 1
        return self.get_masked_model(custom_mask=mask)
    
    def inference(self, model: HookedMNISTClassifier, data_loader: DataLoader, is_forget_set: bool = True, description: str = None) -> Any:
        model = model.to(self.config.device)
        model.eval()

        test_loss, num_batches = 0, len(data_loader)
        score, total = 0, len(data_loader.dataset)
        with torch.no_grad():
            mean_prob = None
            plotted = False # generate one plot per inference
            for i, (input, target) in enumerate(data_loader):
                input = input.to(self.config.device)
                target = target.to(self.config.device)
                
                preds: torch.Tensor = model(input)
                classifier_probs: List[float] = F.softmax(preds, dim=-1)

                if self.config.plot_category_probabilities and not plotted and is_forget_set:
                    # the mean probability is only meaningful over a single class, which is why we check is_forget_set
                    probs: List[float] = classifier_probs.mean(dim=0).tolist()
                    mean_prob = probs[self.config.forget_digit] 
                    fig, ax = plt.subplots()
                    plt.figure(figsize=(10, 6))
                    plt.bar(range(10), probs, tick_label=[str(i) for i in range(10)])
                    save_path = self.config.metrics_path / f'probabilities/{description}/'
                    save_path.mkdir(parents=True, exist_ok=True)
                    save_path = save_path / f'top_k_{self.config.topK}.png'

                    plt.xlabel('Class')
                    plt.ylabel('Probability')
                    plt.title(f'Classifier Probabilities ({description})')
                    plt.xticks(range(10))
                    plt.ylim(0, 1)
                    plt.savefig(save_path)
                    plotted = True

                
                test_loss += self.graph_generator.loss_fn(preds, target).item()
                score += (classifier_probs.argmax(1) == target).type(torch.float).sum().item()

                logger.info(f'@@@@@ Predictions {classifier_probs.tolist()[0]} | Argmax: {preds.argmax(1)[0]} | Forget digit {self.config.forget_digit} @@@@@')

            test_loss /= num_batches
            test_loss = round(test_loss, 5)
            score /= total
            score = round(100*score, 1) # percentage

        eval_set = 'forget set' if is_forget_set else 'retain set'
        logger.info(f'Exp: {description} | test_loss on {eval_set} {test_loss} | Score {score} %')

        metrics =  {
            'experiment': description,
            'eval_data': eval_set,
            'test_loss': test_loss,
            'score': score,
        }

        if is_forget_set:
            metrics['mean_classifier_probability_on_forget_digit'] = mean_prob

        return metrics
    
    def draw_visualization(self, loss: List[float], score: List[float], experiment: str = '') -> None:
        categories = ['Before Masking', 'After Masking', 'Random']

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

        # Plot loss
        axs[0].bar(categories, loss, color='blue')
        axs[0].set_title('test_loss')
        axs[0].set_ylabel('Loss')

        # Plot score
        axs[1].bar(categories, score, color='blue')
        axs[1].set_title('Score')
        axs[1].set_ylabel('Score')

        plt.suptitle(f'{experiment} (top-{self.config.topK})')
        plt.tight_layout()

        self.config.metrics_path.mkdir(parents=True, exist_ok=True)
        save_path = self.config.metrics_path / f'{experiment}/'
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / f'top_{self.config.topK}.png'
        plt.savefig(save_path)

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

        random_baseline_eval_metrics = self.inference(
            description='random on forget set',
            model=self.get_randomly_masked_model(),
            data_loader=self.mnist_forget_set(),
            is_forget_set=True
        )


        if self.draw_eval_plots:

            self.draw_visualization(
                loss=[before_masking_eval_metrics['test_loss'], after_masking_eval_metrics['test_loss'], random_baseline_eval_metrics['test_loss']],
                score=[before_masking_eval_metrics['score'], after_masking_eval_metrics['score'], random_baseline_eval_metrics['score']],
                experiment='eval_unlearning_on_forget_set'
            )

        return {
            'experiment': 'eval_unlearning',
            # 'top-K value': self.config.topK,
            'eval_data': 'forget_set',

            'before_masking_loss': before_masking_eval_metrics['test_loss'],
            'before_masking_score': before_masking_eval_metrics['score'],
            'before_mask_probability': before_masking_eval_metrics['mean_classifier_probability_on_forget_digit'],

            'after_masking_loss': after_masking_eval_metrics['test_loss'],
            'after_masking_score': after_masking_eval_metrics['score'],
            'after_mask_probability': after_masking_eval_metrics['mean_classifier_probability_on_forget_digit'],

            'random_masking_loss': random_baseline_eval_metrics['test_loss'],
            'random_masking_score': random_baseline_eval_metrics['score'],
            'random_masking_probability': random_baseline_eval_metrics['mean_classifier_probability_on_forget_digit'],


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

        random_baseline_eval_metrics = self.inference(
            description='random on retain set',
            model=self.get_randomly_masked_model(),
            data_loader=self.mnist_retain_set(),
            is_forget_set=False
        )

        if self.draw_eval_plots:

            self.draw_visualization(
                loss=[before_masking_eval_metrics['test_loss'], after_masking_eval_metrics['test_loss'], random_baseline_eval_metrics['test_loss']],
                score=[before_masking_eval_metrics['score'], after_masking_eval_metrics['score'], random_baseline_eval_metrics['score']],
                experiment='eval_performance_degradation_on_retain_set'
            )

        return {
            'experiment': 'eval_performance_degradation',
            # 'top-K value': self.config.topK,
            'eval_data': 'retain_set',

            'before_masking_loss': before_masking_eval_metrics['test_loss'],
            'before_masking_score': before_masking_eval_metrics['score'],

            'after_masking_loss': after_masking_eval_metrics['test_loss'],
            'after_masking_score': after_masking_eval_metrics['score'],

            'random_masking_loss': random_baseline_eval_metrics['test_loss'],
            'random_masking_score': random_baseline_eval_metrics['score'],
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

        random_baseline_eval_metrics = self.inference(
            description='random forget set',
            model=self.get_randomly_masked_model(),
            data_loader=self.mnist_forget_set(),
            is_forget_set=True
        )

        if self.draw_eval_plots:

            self.draw_visualization(
                loss=[before_masking_eval_metrics['test_loss'], after_masking_eval_metrics['test_loss'], random_baseline_eval_metrics['test_loss']],
                score=[before_masking_eval_metrics['score'], after_masking_eval_metrics['score'], random_baseline_eval_metrics['score']],
                experiment='eval_mask_efficacy_on_foget_set'
            )

        return {
            'experiment': 'eval_performance_degradation',
            # 'top-K value': self.config.topK,
            'eval_data': 'forget_set',

            'before_masking_loss': before_masking_eval_metrics['test_loss'],
            'before_masking_score': before_masking_eval_metrics['score'],
            
            'after_masking_loss': after_masking_eval_metrics['test_loss'],
            'after_masking_score': after_masking_eval_metrics['score'],

            'random_masking_loss': random_baseline_eval_metrics['test_loss'],
            'random_masking_score': random_baseline_eval_metrics['score'],
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
            file_path = self.config.metrics_path / 'metrics'
            file_path.mkdir(exist_ok=True, parents=True)
            file_path = file_path / f'top-{self.config.topK}.json'
            with open(file_path, 'a') as f:
                f.write(json.dumps(metrics) + '\n')

        return metrics

if __name__ == '__main__':
    config = EvalConfig()
    eval = Eval(config)
    metrics = eval.eval()
