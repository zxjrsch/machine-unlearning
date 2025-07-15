import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Tuple, Union

import torch
from loguru import logger
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torchinfo import summary
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from model import Activations, Gradients, HookedMNISTClassifier, Weights

global_config = OmegaConf.load("configs/config.yaml")

class ModelInspector:
    def __init__(self, model: nn.Module) -> None:
        self.model = model 

    def inspect(self):
        summary(self.model)

    def get_layer_parameters(self) -> Iterator[Parameter]:
        for _, param in self.model.named_parameters():
            yield param

    def get_layer_types(self) -> List[Tuple[str, str]]:
        """
        Inspect the layers returned by nn.Module.name_parameters()
        Sample output [('Linear', 'bias')] shows the bias term of a linear layer.
        """
        layer_types = []
        for layer, _ in self.model.named_parameters():
            module_name = '.'.join(layer.split('.')[:-1])
            module = self.model.get_submodule(module_name)
            layer_types.append((module.__class__.__name__, layer.split('.')[-1]))
        return layer_types
    
    def get_layer_signatures(self) -> List[str]:
        """
        Inspect layer signature of nn.Module.name_parameters()
        Sample output: ['Linear(in_features=784, out_features=512, bias=False)']
        """
        layer_signature = []
        for layer, _ in self.model.named_parameters():
            module_name = '.'.join(layer.split('.')[:-1])
            module = self.model.get_submodule(module_name)
            layer_signature.append(module.__repr__())
        return layer_signature
    
    def get_layer_shapes(self) -> List[Tuple[str, torch.Size]]:
        """
        Size of the nn.Module.name_parameters() tensors.
        Sample output: [('weight', torch.Size([512, 784]))]
        """
        layer_shapes = []
        for layer, param in self.model.named_parameters():
            layer_shapes.append((layer.split('.')[-1], param.shape))
        return layer_shapes
    
    def get_layer_parameter_count(self) -> List[int]:
        """
        Named parameter count by layer named_parameters().
        """
        layer_parameter_count = []
        for _, param in self.model.named_parameters():
            if param.requires_grad:
                layer_parameter_count.append(param.numel())
        return layer_parameter_count
    
    def count_trainable_parameters(self) -> int:
        N = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        assert N == sum(self.get_layer_parameter_count())
        return N
        

    def get_layer_full_names(self) -> List[str]:
        """
        Inspect full name of layers.
        ['_orig_mod.hidden_layers.hidden layer 1.0.weight']
        """
        object_names = []
        for layer, _ in self.model.named_parameters():
            object_names.append(layer)
        return object_names

@dataclass
class GCNBatch:
    feature_batch: Tensor
    input_batch: Tensor
    target_batch: Tensor    

    
class GraphGenerator(ModelInspector):
    def __init__(self, 
                model: HookedMNISTClassifier, 
                checkpoint_path: Union[Path, None]=None, 
                graph_dataset_dir: Path = Path('./datasets/Graphs'),
                process_save_batch_size: int = 64,
                forget_digit: int = 9, 
                device: str = global_config.device,
                mask_layer: Union[int, None] = -2,   # specify one layer to mask, if None then all layers selected
                save_redundant_features: bool = True
    ) -> None:
    
        
        super().__init__(model)

        if checkpoint_path is not None:
            self.model = torch.compile(self.model)
            self.model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            logger.info(f'Loaded pretrained model from {checkpoint_path}')

        graph_dataset_dir.mkdir(exist_ok=True, parents=True)
        self.graph_dataset_dir = graph_dataset_dir
        self.mask_layer = self.validate_layer(mask_layer)

        self._save_redundant_features = save_redundant_features

        self.forget_digit = forget_digit
        self.process_save_batch_size = process_save_batch_size
        self.device = device
        self.data_loader: DataLoader = self.get_dataloader()
        self.num_vertices = self.get_num_vertices()
        self.weight_feature = self.get_weight_feature() 
        self.edge_matrix = self.get_edge_matrix()
        self.loss_fn = nn.CrossEntropyLoss()
        self.analyze()

    def get_dataloader(self, data_path: str = './datasets') -> DataLoader:
        dataset = MNIST(root=data_path, train=True, transform=ToTensor(), download=True)

        # Important: we want per sample grad, not batch-averaged grad, hence batch_size=1 for feature extraction
        return DataLoader(dataset=dataset, batch_size=1)

    def reset_forget_digit(self, digit: Union[int, None], data_path='./datasets') -> DataLoader:
        if digit is None:
            self.forget_digit = None
            self.data_loader = self.get_dataloader(data_path=data_path)
        else:
            assert 0 <= digit <= 9
            self.forget_digit = digit
            self.data_loader = self.get_single_class_dataloader(data_path=data_path)

        logger.info(f'Resetting graph generator dataloader to digit: {digit}')
        return self.data_loader


    def get_single_class_dataloader(self, data_path: str = './datasets') -> DataLoader:
        dataset = MNIST(root=data_path, train=True, transform=ToTensor(), download=True)
        subset_indices = (dataset.targets == self.forget_digit).nonzero(as_tuple=True)[0]
        subset = torch.utils.data.Subset(dataset, subset_indices)

        # Important: we want per sample grad, not batch-averaged grad, hence batch_size=1 for feature extraction
        return DataLoader(dataset=subset, batch_size=1)

    def analyze(self) -> None:
        om =  math.floor(math.log(self.num_vertices, 10))
        logger.info(f'Graph contains {self.num_vertices} | order magnitude 10^{om} vertices.')
        om =  math.floor(math.log(self.edge_matrix.shape[1], 10))
        logger.info(f'Graph contains {self.edge_matrix.shape[1]} | order magnitude 10^{om} edges.')

        weight_feature_dim = self.weight_feature.shape
        assert weight_feature_dim == torch.Size((self.num_vertices, ))

    def validate_layer(self, mask_layer) -> Union[None, int]:
        if mask_layer is not None:
            try:
                [p for p in self.model.parameters() if p.requires_grad][mask_layer]
            except Exception:
                logger.info(f'Layer {mask_layer} is invalid.')
                exit()
        return mask_layer

    def get_num_vertices(self):
        if self.mask_layer is None:
            return self.count_trainable_parameters()
        return self.get_layer_parameter_count()[self.mask_layer]


    def get_weight_feature(self, save: bool=False) -> Tensor:

        trainable_layers = [torch.flatten(p) for p in self.model.parameters() if p.requires_grad]

        if self.mask_layer is None:
            # get all layers
            flattened_weights = torch.cat(trainable_layers)
        else:
            # single layer
            flattened_weights = trainable_layers[self.mask_layer]

        if save:
            file_name: Path = self.graph_dataset_dir / 'flattened_model_weights.pt'
            torch.save(flattened_weights, file_name)
            logger.info(f'Weights saved at {file_name}')
        
        return flattened_weights
    
    def get_edge_matrix(self) -> Tensor:
        
        if self.mask_layer is None:
            return self.get_edge_matrix_full_model()
        
        return self.get_edge_matrix_by_layer()
    
    def get_edge_matrix_by_layer(self) -> Tensor:

        # assume layer is matrix, as in our experiments
        layer = [p for p in self.model.parameters() if p.requires_grad][self.mask_layer]
        m, n = layer.shape
        enumeration_matrx = torch.arange(start=0, end=m*n, step=1, dtype=torch.long).unflatten(dim=0, sizes=(m, n))

        edge_matrix_list = []

        for c in range(n):
            col = enumeration_matrx[:, c]
            O_edges = torch.cartesian_prod(col, col)    # out edges emanate out of an activation 
            masks = O_edges[:, 0] == O_edges[:, 1]
            O_edges = O_edges[~masks]
            edge_matrix_list.append(O_edges.T)

        for r in range(m):
            row = enumeration_matrx[r, :]
            I_edges = torch.cartesian_prod(row, row)
            masks = I_edges[:, 0] == I_edges[:, 1]
            I_edges = I_edges[~masks]
            edge_matrix_list.append(I_edges.T)

        edge_matrix = torch.cat(edge_matrix_list, dim=1)

        if self._save_redundant_features:
            file_name: Path = self.graph_dataset_dir / 'graph_edge_matrix.pt'
            torch.save(edge_matrix, file_name)
            logger.info(f'Edge matrix saved at {file_name}')

        return edge_matrix


    def get_edge_matrix_full_model(self) -> Tensor:
        dims = self.model.dim_array + [self.model.out_dim]
        assert len(dims) >= 3  # at least a pair of matrices is needed, specified by 3 numbers (a x b) * (b x c)
        assert sum(dims[i] * dims[i+1] for i in range(len(dims)-1)) == self.num_vertices

        v = torch.arange(start=0, end=self.num_vertices, step=1, dtype=torch.long)
        c, weight_enumeration_matrices = 0, []
        for k in range(len(dims)-1):
            m, n = dims[k+1], dims[k]
            enumeration_matrix = v[c: c+m*n].unflatten(dim=0, sizes=(m, n))
            weight_enumeration_matrices.append(enumeration_matrix)
            c += m*n

        edge_matrices_list = []
        for j in range(len(weight_enumeration_matrices)-1):
            A = weight_enumeration_matrices[j+1]
            B = weight_enumeration_matrices[j]

            assert A.shape[1] == B.shape[0]
            p, q, r = A.shape[0], A.shape[1], B.shape[1]
            for activation_idx in range(q):

                # cross I.O. edges
                forward_edges = torch.cartesian_prod(A[:, activation_idx], B[activation_idx, :]).T
                edge_matrices_list.append(forward_edges)
                del forward_edges
                torch.cuda.empty_cache()

                backward_edges = torch.cartesian_prod(B[activation_idx, :], A[:, activation_idx]).T
                edge_matrices_list.append(backward_edges)
                del backward_edges
                torch.cuda.empty_cache()

                # In edges 
                I_edges = torch.cartesian_prod(B[activation_idx, :], B[activation_idx, :])
                masks = I_edges[:, 0] == I_edges[:, 1]
                I_edges = I_edges[~masks]
                edge_matrices_list.append(I_edges.T)
                del I_edges, masks
                torch.cuda.empty_cache()

                # Out edges 
                O_edges = torch.cartesian_prod(A[:, activation_idx], A[:, activation_idx])
                masks = O_edges[:, 0] == O_edges[:, 1]
                O_edges = O_edges[~masks]
                edge_matrices_list.append(O_edges.T)
                del O_edges, masks
                torch.cuda.empty_cache()

        # edge case 
        # final layer (the classifier) has only edges of In-type
        B: Tensor = weight_enumeration_matrices[-1]
        q, r = B.shape[0], B.shape[1]
        for activation_idx in range(q):
            I_edges = torch.cartesian_prod(B[activation_idx, :], B[activation_idx, :])
            masks = I_edges[:, 0] == I_edges[:, 1]
            I_edges = I_edges[~masks]
            edge_matrices_list.append(I_edges.T)
            del I_edges, masks
            torch.cuda.empty_cache()

        edge_matrix = torch.cat(edge_matrices_list, dim=1)
        torch.cuda.empty_cache()

        if self._save_redundant_features:
            file_name: Path = self.graph_dataset_dir / 'graph_edge_matrix.pt'
            torch.save(edge_matrix, file_name)
            logger.info(f'Edge matrix saved at {file_name}')

        return edge_matrix

                
    def flatten_and_zero_grad(self) -> Tensor:
        gradients = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                g = param.grad 
                assert g is not None
                gradients.append(torch.flatten(g))
                param.grad = None
        if self.mask_layer is not None:
            # get grad for layer 
            gradients = gradients[self.mask_layer]
        else:
            # get grad for entire model
            gradients =  torch.cat(gradients)
        assert gradients.numel() == self.num_vertices
        return gradients
    
    def flatten_in_out_activation(self, activations: List[Activations]) -> Tuple[Tensor, Tensor]:
        """Due to the need for per-sample gradient, this function currently only support single data batches."""
        if self.mask_layer is None:
            return self.flatten_in_out_activation_full_model(activations)
        
        return self.flatten_in_out_activation_single_layer(activations)
    
    def flatten_in_out_activation_single_layer(self, activations: List[Activations]) -> Tuple[Tensor, Tensor]:

        # layers weights are matrices in our experiments
        m, n = [p for p in self.model.parameters() if p.requires_grad][self.mask_layer].shape
        activation: Activations = activations[self.mask_layer]
        in_activation, out_activation = activation.input_activation[0].squeeze(), activation.output_activation[0].squeeze()

        assert in_activation.shape[-1] == n
        assert out_activation.shape[-1] == m

        in_feature = in_activation.expand(m, -1)
        out_feature = out_activation.unsqueeze(1).expand(-1, n)

        # # sanity check 
        # logger.info(f'{in_feature.shape} | {out_feature.shape} | {m} x {n}')

        assert in_feature.shape == torch.Size((m, n))
        assert out_feature.shape == torch.Size((m, n))

        # # sanity check 
        # logger.info(f'{in_feature.flatten().shape}, {out_feature.flatten().shape}')

        return in_feature.flatten(), out_feature.flatten()

    def flatten_in_out_activation_full_model(self, activations: List[Activations]) -> Tuple[Tensor, Tensor]:
        dims = self.model.dim_array + [self.model.out_dim]

        in_activation_features = []
        out_activation_features = []
        
        for j in range(len(activations)):
            m, n = dims[j], dims[j+1]

            # indexing specific to HookedMNISTClassifier models
            input_activation = activations[j].input_activation[0].squeeze()
            output_activation = activations[j].output_activation[0].squeeze()
            
            assert input_activation.shape[0] == m
            assert output_activation.shape[0] == n

            in_feature = input_activation.expand(n, -1)
            out_feature = output_activation.unsqueeze(1).expand(-1, m)

            assert in_feature.shape == torch.Size((n, m))
            assert out_feature.shape == torch.Size((n, m))

            in_activation_features.append(in_feature.flatten())
            out_activation_features.append(out_feature.flatten())
        
        in_features_vector = torch.cat(in_activation_features)
        out_features_vector = torch.cat(out_activation_features)

        # # tracer
        # logger.info(f'{in_feature.flatten().shape} | {out_feature.flatten().shape} | {m*n}')
        # logger.info(f'{in_features_vector.shape}')
        # logger.info(f'{out_features_vector.shape}')

        return in_features_vector, out_features_vector
    
    def save_forward_backward_features(self, limit: Union[int, None] = 256) -> None:
        """Saves grads and activations."""

        assert self.data_loader # data loader must be initialized before getting gradient features
        self.model: HookedMNISTClassifier = self.model.to(self.device)
        self.weight_feature = self.weight_feature.to(self.device)

        self.model.capture_mode(is_on=True)
        self.model.eval()

        eumerator = list(enumerate(self.data_loader))
        if limit is not None:
            eumerator = eumerator[:limit]

        feature_batch, batch_idx, total_batches = [], 0, math.ceil(len(eumerator) / self.process_save_batch_size)
        input_batch, target_batch = [], []
        for i, (input, target) in eumerator:
            input_batch.append(input)
            target_batch.append(target)
            input, target = input.to(self.device), target.to(self.device)
            # assert torch.all(target == self.forget_digit)   # sanity check
            preds = self.model(input)
            loss: Tensor = self.loss_fn(preds, target)
            loss.backward()

            in_feature, out_feature = self.flatten_in_out_activation(self.model.activations)
            gradients: Tensor = self.flatten_and_zero_grad()

            # dim = [num_vertices, num_features] 
            graph_feature_matrix = torch.column_stack((in_feature, out_feature, self.weight_feature, gradients))
            
            # # tracer
            # logger.info(f'{in_feature.shape} | {out_feature.shape} | {self.weight_feature.shape} | {gradients.shape}')
            # logger.info(graph_feature_matrix.shape)

            feature_batch.append(graph_feature_matrix)

            if len(feature_batch) == self.process_save_batch_size:
                batch_idx += 1

                gcn_batch = GCNBatch(
                    feature_batch = torch.stack(feature_batch, dim=0),
                    input_batch = torch.stack(input_batch, dim=0).squeeze(),
                    target_batch = torch.stack(target_batch, dim=0).squeeze()
                )

                if batch_idx % 10 == 0:
                    logger.info(f'Saving batch {batch_idx} of {total_batches}')
                file_name = self.graph_dataset_dir / f'batch_{batch_idx}.pt'

        
                torch.save(gcn_batch, file_name)
                feature_batch = []
                input_batch = []
                target_batch = []
                torch.cuda.empty_cache()

            self.model.reset_activations()
            torch.cuda.empty_cache()

        if len(feature_batch) > 0:
            batch_idx += 1
            gcn_batch = GCNBatch(
                feature_batch = torch.stack(feature_batch, dim=0),
                input_batch = torch.stack(input_batch, dim=0).squeeze(),
                target_batch = torch.stack(target_batch, dim=0).squeeze()
            )
            logger.info(f'Saving batch {batch_idx} of {total_batches}')
            file_name = self.graph_dataset_dir / f'batch_{batch_idx}.pt'
            torch.save(gcn_batch, file_name)

            del feature_batch, input_batch, target_batch
            self.model.reset_activations()
            torch.cuda.empty_cache()

    def get_data(self, idx: int=0) -> Tuple[Any, Any, Data]:
        """Generate target and PyG graph data"""

        assert self.data_loader # data loader must be initialized before getting gradient features
        self.model: HookedMNISTClassifier = self.model.to(self.device)
        self.weight_feature = self.weight_feature.to(self.device)

        self.model.capture_mode(is_on=True)
        self.model.eval()


        for i, (input, target) in enumerate(self.data_loader):
            if i == idx:
                input, target = input.to(self.device), target.to(self.device)
                # assert torch.all(target == self.forget_digit)   # sanity check
                preds = self.model(input)
                loss: Tensor = self.loss_fn(preds, target)
                loss.backward()

                in_feature, out_feature = self.flatten_in_out_activation(self.model.activations)
                gradients: Tensor = self.flatten_and_zero_grad()

                # dim = [num_vertices, num_features]
                graph_feature_matrix = torch.column_stack((in_feature, out_feature, self.weight_feature, gradients))
                
                # # tracer
                # logger.info(f'{in_feature.shape} | {out_feature.shape} | {self.weight_feature.shape} | {gradients.shape}')
                # logger.info(graph_feature_matrix.shape)

                self.model.reset_activations()
                torch.cuda.empty_cache()

                return input, target, Data(x=graph_feature_matrix, edge_index=self.edge_matrix)

    def get_representative_features(self, representatives: List[Tuple[Tensor, Tensor]]) -> List[Tuple[Tensor, Tensor]]:
        """The input list consists of (image, label) and output list contains (feature, label). 
        This method is currently used for evaluation pipeline.
        """
        """Saves grads and activations."""

        self.model: HookedMNISTClassifier = self.model.to(self.device)
        self.weight_feature = self.weight_feature.to(self.device)

        self.model.capture_mode(is_on=True)
        self.model.eval()

        graph_features: List[Tuple[Tensor, Tensor]] = []
        for input, target in representatives:

            self.model.zero_grad(set_to_none=True)

            input, target = input.to(self.device), target.to(self.device)
            preds = self.model(input)
            loss: Tensor = self.loss_fn(preds, target)
            loss.backward()

            in_feature, out_feature = self.flatten_in_out_activation(self.model.activations)
            gradients: Tensor = self.flatten_and_zero_grad()

            # dim = [num_vertices, num_features] 
            graph_feature_matrix = torch.column_stack((in_feature, out_feature, self.weight_feature, gradients))
            
            # # tracer
            # logger.info(f'{in_feature.shape} | {out_feature.shape} | {self.weight_feature.shape} | {gradients.shape}')
            # logger.info(graph_feature_matrix.shape)
            
            data = (graph_feature_matrix, target)
            graph_features.append(data)

            # logger.info(f'Generated graph feature for representatitive {target.detach()}')
        return graph_features