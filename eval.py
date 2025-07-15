import copy
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

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
    forget_digit = 9
    batch_size = 16
    device = global_config.device
    mask_layer = -2
    topK = 2500

class Eval:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.gcn = self.load_gcn()
        self.classifier = self.load_classifier()
        self.graph_generator = self.load_graph_generator()

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
    
    def inference(self, model: HookedMNISTClassifier, data_loader: DataLoader, is_forget_set: bool = True) -> None:
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
            score /= total

        eval_set = 'forget set' if is_forget_set else 'retain set'
        logger.info(f'Test loss on {eval_set} {round(test_loss, 5)} | Score {round(100*score, 1)} %')

    
    def eval(self) -> None:

        # masked model 
        self.inference(
            model=self.get_masked_model(),
            data_loader=self.mnist_forget_set(),
            is_forget_set=True
        )
        self.inference(
            model=self.get_masked_model(),
            data_loader=self.mnist_retain_set(),
            is_forget_set=False
        )


         # original model 
        self.inference(
            model=self.classifier,
            data_loader=self.mnist_forget_set(),
            is_forget_set=True
        )
        self.inference(
            model=self.classifier,
            data_loader=self.mnist_retain_set(),
            is_forget_set=False
        )

        # # study mask efficacy
        # self.inference(
        #     model=self.model_mask_forget_class(),
        #     data_loader=self.mnist_forget_set(),
        #     is_forget_set=True
        # )
        
        # self.inference(
        #     model=self.classifier,
        #     data_loader=self.mnist_forget_set(),
        #     is_forget_set=True
        # )

if __name__ == '__main__':
    config = EvalConfig()
    eval = Eval(config)
    eval.eval()
