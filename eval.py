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
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from data import GraphGenerator
from model import HookedMNISTClassifier, MaskingGCN
from trainer import (UnlearningSFT, UnlearningSFTConfig,
                     gumbel_top_k_sampling_v2)

global_config = OmegaConf.load("configs/config.yaml")

@dataclass
class EvalConfig:
    gcn_path: Path  #=  Path(sorted(glob.glob(os.path.join('checkpoints/gcn/', '*.pt')))[0]) 
    classifier_path: Path  #= Path(sorted(glob.glob(os.path.join('checkpoints/mnist_classifier/', '*.pt')))[0])
    data_path: Path = Path('datasets')
    graph_data_path: Path = Path('eval/Graphs')
    metrics_path: Path = Path('eval/Metrics and Plots')
    forget_digit: int = 9
    batch_size: int = 256
    device: str = global_config.device
    mask_layer = -2
    topK:int = 7000
    kappa: int = 5000
    plot_category_probabilities: bool = True
    use_set_difference_masking_strategy: bool = False

class Eval:
    def __init__(self, config: EvalConfig, draw_eval_plots: bool=True):
        self.config = config
        self.gcn = self.load_gcn()
        self.classifier = self.load_classifier()
        self.graph_generator = self.load_graph_generator()
        self.draw_eval_plots = draw_eval_plots
        self.use_set_difference_masking_strategy = config.use_set_difference_masking_strategy
        # sft unlearning baseline
        self.finetuning_unlearning_model = self.train_sft_model()   

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

    def mnist_single_class(self, digit: int) -> DataLoader:
        dataset = MNIST(root=self.config.data_path, train=False, transform=ToTensor(), download=True)
        desired_indices = (dataset.targets == digit).nonzero(as_tuple=True)[0]
        desired_data = Subset(dataset, desired_indices)
        return DataLoader(dataset=desired_data, batch_size=self.config.batch_size)      
    
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
        """
        Set different masking strategy computes the set A of important weights for forget digit prediction, 
        and the set B of important weights for the retain digits, then takes the set difference C = A - B which 
        is considered to be the weights solely responsible for forget digit, and when masking out set C, the model 
        degradation on retain is minimal. Thus the cardinality of #C <= top-K value.

        If is_set_difference_strategy == False then we generate the mask based on feasibility scores of weights.
        """
        mask_label_list = self.compute_representative_masks()

        forget_mask = None
        retain_masks_sum = None

        for mask, digit in mask_label_list:
            if digit == self.config.forget_digit:
                forget_mask = mask
            else:
                retain_masks_sum = mask if retain_masks_sum is None else retain_masks_sum + mask

        if self.use_set_difference_masking_strategy:

            # weights important for forget set prediction but not important for retain classes correspond to 1s
            mask = torch.clamp(forget_mask - retain_masks_sum, min=0)

            # negation operation return True for weight values which are are important to predict RETAIN set but not forget data
            mask = mask == 0
            num_keep_weights = mask.sum().item()
            percent = round(num_keep_weights/self.graph_generator.num_vertices * 100, 2)

            logger.info(f'top-{self.config.topK} | target layer {self.config.mask_layer} | Retaining {num_keep_weights} ({percent} %)')

            return (mask == 0).float()
        
        else:

            # To handle target layer specified as a negative value such as -2
            num_layers = sum([1 for layer in self.classifier.parameters() if layer.requires_grad])
            target_layer = torch.tensor(self.config.mask_layer % num_layers + 1)  # index from 1 to avoid taking log(0)
            
            num_retain_classes = len(mask_label_list) - 1 
            assert num_retain_classes == 9
            scores = retain_masks_sum / num_retain_classes - forget_mask

            feasibility_ranking = torch.log(target_layer) / torch.log(1+scores) 
            masked_weight_indices = feasibility_ranking.topk(self.config.kappa).indices # zero weights with top-K feasibility
            mask = torch.ones_like(feasibility_ranking, dtype=torch.float)
            mask[masked_weight_indices] = 0

            # tune kappa to desired loss
            percent = round(1 - self.config.kappa/self.graph_generator.num_vertices * 100, 2)
            logger.info(f'top-{self.config.topK} | target layer {self.config.mask_layer} | retaining (weights - kappa) = {self.graph_generator.num_vertices - self.config.kappa} ({percent} %) parameters.')

            return mask

    
    def model_mask_forget_class(self) -> HookedMNISTClassifier:
        forget_class_mask = None
        for mask, digit in self.compute_representative_masks():
            if digit == self.config.forget_digit:
                forget_class_mask = mask
                break

        topK_indices = forget_class_mask.topk(self.config.topK).indices
        mask = torch.zeros_like(forget_class_mask, dtype=torch.float)
        mask[topK_indices] = 1
        return self.get_masked_model(custom_mask=mask)

    def get_masked_model(self, custom_mask: Union[Tensor, None]=None) -> HookedMNISTClassifier:
        """Single layer masking."""
        if custom_mask is None:
            mask: Tensor = self.get_model_mask()
        else:
            mask = custom_mask
        mask = mask.to(self.config.device)
        weight_vector = self.graph_generator.weight_feature.to(self.config.device)
        layer_vect = torch.mul(mask, weight_vector)
        
        retain_rate = 1 - self.config.kappa / self.graph_generator.num_vertices
        layer_vect /= retain_rate

        m, n = [p for p in self.classifier.parameters() if p.requires_grad][self.config.mask_layer].shape
        layer_matrix = layer_vect.unflatten(dim=0, sizes=(m,n))

        state_dict = copy.deepcopy(self.classifier.state_dict())
        key = list(state_dict.keys())[self.config.mask_layer]
        state_dict[key] = layer_matrix

        model: HookedMNISTClassifier = torch.compile(HookedMNISTClassifier())
        model.load_state_dict(state_dict)
        return model

    def get_randomly_masked_model(self) -> HookedMNISTClassifier:
        mask = torch.zeros(self.graph_generator.num_vertices, dtype=torch.float32)
        num_1s = self.graph_generator.num_vertices - self.config.kappa
        assert num_1s > 0
        indices = torch.randperm(self.graph_generator.num_vertices)[:num_1s]
        mask[indices] = 1
        return self.get_masked_model(custom_mask=mask)
    
    def train_sft_model(self) -> HookedMNISTClassifier:
        sft_config = UnlearningSFTConfig(
            original_model_path=self.config.classifier_path,
            save_dir = Path('checkpoints/finetuned_mnist_classifier'),
            data_dir=self.config.data_path,
            finetune_layer=self.config.mask_layer,
            lr=1e-2,
            device=self.config.device,
            steps=50,
            batch_size=4,
            logging_steps=10,
            forget_digit=self.config.forget_digit
        )
        trainer = UnlearningSFT(sft_config)
        model = trainer.finetune(save_checkpoint=False)
        return model
    
    def inference(self, model: HookedMNISTClassifier, data_loader: DataLoader, is_forget_set: bool = True, description: str = None) -> Dict:
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
                
                preds: Tensor = model(input)
                classifier_probs: Tensor = F.softmax(preds, dim=-1)

                if self.config.plot_category_probabilities and not plotted and is_forget_set:
                    # the mean probability is only meaningful over a single class, which is why we check is_forget_set
                    probs: List[float] = classifier_probs.mean(dim=0).tolist()
                    mean_prob = probs[self.config.forget_digit] 

                    plt.clf()
                    fig, ax = plt.subplots()
                    plt.figure(figsize=(10, 6))
                    plt.bar(range(10), probs, tick_label=[str(i) for i in range(10)])
                    save_path = self.config.metrics_path / f'probabilities/{description}/'
                    save_path.mkdir(parents=True, exist_ok=True)
                    save_path = save_path / f'top_k_{self.config.topK}_kappa_{self.config.kappa}.png'

                    plt.xlabel('Class')
                    plt.ylabel('Probability')
                    plt.title(f'Classifier Probabilities ({description})')
                    plt.xticks(range(10))
                    plt.ylim(0, 1)
                    plt.savefig(save_path)
                    plotted = True

                test_loss += self.graph_generator.loss_fn(preds, target).item()
                score += (classifier_probs.argmax(1) == target).type(torch.float).sum().item()

                # logger.info(f'@@@@@ Predictions {classifier_probs.tolist()[0]} | Argmax: {preds.argmax(1)[0]} | Forget digit {self.config.forget_digit} @@@@@')

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
        plt.clf()

        # apend new baseline label below 
        categories = ['Original', 'MIMU', 'Random', 'SFT']

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

        # Plot loss
        axs[0].bar(categories, loss, color='blue')
        axs[0].set_title('test_loss')
        axs[0].set_ylabel('Loss')

        # Plot score
        axs[1].bar(categories, score, color='blue')
        axs[1].set_title('Score')
        axs[1].set_ylabel('Score')

        plt.suptitle(f'{experiment} (top-{self.config.topK} kappa-{self.config.kappa})')
        plt.tight_layout()

        self.config.metrics_path.mkdir(parents=True, exist_ok=True)
        save_path = self.config.metrics_path / f'{experiment}/'
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / f'top_{self.config.topK}_kappa_{self.config.kappa}.png'
        plt.savefig(save_path)
        plt.close()

    def get_class_probability(self, digit: int, model: nn.Module) -> List[float]:
        data_loader = self.mnist_single_class(digit=digit)
        model = model.to(self.config.device)
        model.eval()
        with torch.no_grad():
            input, target = next(iter(data_loader))
            input = input.to(self.config.device)
            target = target.to(self.config.device)
            logits: torch.Tensor = model(input)
            classifier_probabilities = F.softmax(logits, dim=-1).mean(dim=0).tolist()
            return classifier_probabilities
        
    def draw_class_probability(self, model: nn.Module, description: str) -> Dict:
        plt.clf()
        fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True, constrained_layout=True)

        fig.suptitle(f'Unlearning digit {self.config.forget_digit}', fontsize=16, y=1.05)

        x = list(range(10))
        distributions = {}
        for digit, ax in enumerate(axes.flatten()):
            distribution = self.get_class_probability(digit=digit, model=model)
            distributions[digit] = distribution
            colors = plt.cm.viridis(torch.linspace(0, 1, len(x)).numpy())
            ax.bar(x, distribution, align='center', alpha=0.8, color=colors)
            ax.set_title(f'Digit {digit}')
            ax.set_xticks(x)
            ax.set_xticklabels(x)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

        fig.supxlabel('Classes', fontsize=14)
        fig.supylabel('Classifier Probability', fontsize=14)

        save_path = self.config.metrics_path / f'probabilities_all_classes/{description}/'
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / f'{description}_top_k_{self.config.topK}_kappa_{self.config.kappa}.png'

        plt.savefig(save_path)
        plt.close()
        return distributions

    def draw_weight_distribution(self, model: nn.Module, model_name: str, is_abs_val: bool = True) -> None:

        layer_vector: Tensor = [p.flatten() for p in model.parameters()][self.config.mask_layer]
        if is_abs_val:
            layer_vector = layer_vector.abs()
        layer_vector = layer_vector.tolist()
        assert len(layer_vector) == self.graph_generator.num_vertices
        plt.clf()
        plt.hist(layer_vector, bins=15, edgecolor='black')
        plt.xlabel('Weight')
        plt.ylabel('Frequency')
        plt.title(f'{model_name} parameter frequency (layer: {self.config.mask_layer})')

        save_path = self.config.metrics_path / f'weight_histograms/{model_name}/'
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / f'{model_name}_top_k_{self.config.topK}_kappa_{self.config.kappa}.png'
        plt.savefig(save_path)
        plt.close()

    def eval_class_probability(self) -> Dict:
        # record dictionaries to json later when needed
        self.draw_class_probability(model=self.classifier, description='classifier')
        self.draw_class_probability(model=self.get_masked_model(), description='mimu')
        self.draw_class_probability(model=self.get_randomly_masked_model(), description='random')
        self.draw_class_probability(model=self.finetuning_unlearning_model, description='sft')

    def eval_weight_distributions(self) -> None:
        self.draw_weight_distribution(model=self.classifier, model_name='classifier')
        self.draw_weight_distribution(model=self.get_masked_model(), model_name='mimu')
        self.draw_weight_distribution(model=self.get_randomly_masked_model(), model_name='random')
        self.draw_weight_distribution(model=self.finetuning_unlearning_model, model_name='sft')

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
            description='random on forget set (unlearning)',
            model=self.get_randomly_masked_model(),
            data_loader=self.mnist_forget_set(),
            is_forget_set=True
        )


        sft_baseline_eval_metrics = self.inference(
            description='SFT baseline on forget set',
            model=self.finetuning_unlearning_model,
            data_loader=self.mnist_forget_set(),
            is_forget_set=True
        )

        if self.draw_eval_plots:

            self.draw_visualization(
                loss=[before_masking_eval_metrics['test_loss'], 
                      after_masking_eval_metrics['test_loss'], 
                      random_baseline_eval_metrics['test_loss'], 
                      sft_baseline_eval_metrics['test_loss'],
                ],
                score=[
                    before_masking_eval_metrics['score'], 
                    after_masking_eval_metrics['score'], 
                    random_baseline_eval_metrics['score'],
                    sft_baseline_eval_metrics['score']
                ],
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

            # random baseline
            'random_masking_loss': random_baseline_eval_metrics['test_loss'],
            'random_masking_score': random_baseline_eval_metrics['score'],
            'random_masking_probability': random_baseline_eval_metrics['mean_classifier_probability_on_forget_digit'],

            # sft baseline
            'sft_baseline_loss': sft_baseline_eval_metrics['test_loss'],
            'sft_baseline_score': sft_baseline_eval_metrics['score'],
            'sft_baseline_probability': sft_baseline_eval_metrics['mean_classifier_probability_on_forget_digit']

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
            description='random retain set (degradation)',
            model=self.get_randomly_masked_model(),
            data_loader=self.mnist_retain_set(),
            is_forget_set=False
        )

        sft_baseline_eval_metrics = self.inference(
            description='SFT baseline on retain set',
            model=self.finetuning_unlearning_model,
            data_loader=self.mnist_retain_set(),
            is_forget_set=False
        )

        if self.draw_eval_plots:

            self.draw_visualization(
                loss=[before_masking_eval_metrics['test_loss'], 
                      after_masking_eval_metrics['test_loss'], 
                      random_baseline_eval_metrics['test_loss'], 
                      sft_baseline_eval_metrics['test_loss'],
                ],
                score=[
                    before_masking_eval_metrics['score'], 
                    after_masking_eval_metrics['score'], 
                    random_baseline_eval_metrics['score'],
                    sft_baseline_eval_metrics['score']
                ],
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

            # random baseline 
            'random_masking_loss': random_baseline_eval_metrics['test_loss'],
            'random_masking_score': random_baseline_eval_metrics['score'],

            # sft baseline
            'sft_baseline_loss': sft_baseline_eval_metrics['test_loss'],
            'sft_baseline_score': sft_baseline_eval_metrics['score'],
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
            description='random forget set (efficacy)',
            model=self.get_randomly_masked_model(),
            data_loader=self.mnist_forget_set(),
            is_forget_set=True
        )

        sft_baseline_eval_metrics = self.inference(
            description='SFT baseline on forget set',
            model=self.finetuning_unlearning_model,
            data_loader=self.mnist_forget_set(),
            is_forget_set=False
        )

        if self.draw_eval_plots:

            self.draw_visualization(
                loss=[before_masking_eval_metrics['test_loss'], 
                      after_masking_eval_metrics['test_loss'], 
                      random_baseline_eval_metrics['test_loss'], 
                      sft_baseline_eval_metrics['test_loss'],
                ],
                score=[
                    before_masking_eval_metrics['score'], 
                    after_masking_eval_metrics['score'], 
                    random_baseline_eval_metrics['score'],
                    sft_baseline_eval_metrics['score']
                ],
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

            'sft_baseline_loss': sft_baseline_eval_metrics['test_loss'],
            'sft_baseline_score': sft_baseline_eval_metrics['score'],
        }

    
    def eval(self, save_metrics: bool = True) -> Dict:
        unlearning_metrics = self.eval_unlearning()
        performance_degradation_metrics = self.eval_performance_degradation()
        mask_efficacy_metrics = self.eval_mask_efficacy()

        self.eval_weight_distributions()
        self.eval_class_probability()

        metrics = {
            'id': str(uuid.uuid1()),
            'forget_digit': self.config.forget_digit,
            'num_graph_vertices': self.graph_generator.num_vertices,
            'num_graph_edges': self.graph_generator.edge_matrix.shape[1],
            'maked_layer': self.config.mask_layer,
            'top_k_value': self.config.topK,
            'kappa': self.config.kappa,
            'unlearning_metrics': unlearning_metrics,
            'performance_degradation_metrics': performance_degradation_metrics,
            'mask_efficacy_metrics': mask_efficacy_metrics
        }
        if save_metrics:
            self.config.metrics_path.mkdir(exist_ok=True, parents=True)
            file_path = self.config.metrics_path / 'metrics'
            file_path.mkdir(exist_ok=True, parents=True)

            if self.use_set_difference_masking_strategy:
                file_path = file_path / f'top-{self.config.topK}.json'
            else:
                # use feasibility ranking (topK, kappa)
                file_path = file_path / f'top-{self.config.topK}-kappa-{self.config.kappa}.json'
                

            with open(file_path, 'a') as f:
                f.write(json.dumps(metrics) + '\n')

        return metrics

if __name__ == '__main__':
    config = EvalConfig()
    eval = Eval(config)
    metrics = eval.eval()


