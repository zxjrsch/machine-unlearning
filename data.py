import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple, Union

import torch
from loguru import logger
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torchinfo import summary

from model import (Activations, HookedModel, SupportedVisionModels,
                   vision_model_loader)
from utils_data import (SupportedDatasets, UnlearningDataset,
                        get_unlearning_dataset)

global_config = OmegaConf.load("configs/config.yaml")


class ModelInspector:
    def __init__(self, model: nn.Module) -> None:
        self.model: nn.Module = model

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
            module_name = ".".join(layer.split(".")[:-1])
            module = self.model.get_submodule(module_name)
            layer_types.append((module.__class__.__name__, layer.split(".")[-1]))
        return layer_types

    def get_layer_signatures(self) -> List[str]:
        """
        Inspect layer signature of nn.Module.name_parameters()
        Sample output: ['Linear(in_features=784, out_features=512, bias=False)']
        """
        layer_signature = []
        for layer, _ in self.model.named_parameters():
            module_name = ".".join(layer.split(".")[:-1])
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
            layer_shapes.append((layer.split(".")[-1], param.shape))
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
    def __init__(
        self,
        vision_model_type: SupportedVisionModels,
        unlearning_dataset: SupportedDatasets,
        checkpoint_path: Path,
        graph_dataset_dir: Path = Path("./datasets/Graphs"),
        graph_data_cardinaility: Optional[int] = 1024,
        process_save_batch_size: int = 64,
        forget_class: int = 9,
        device: str = global_config["device"],
        mask_layer: Union[int, None] = -2,
        save_redundant_features: bool = True,
    ) -> None:
        """
        If mask_layer is int, it specifies one layer to mask, however if
        mask_layer=None then all layers are selected. None is not generally supported
        for this unlearning codebase due to the high computation cost of training such GCN.
        """

        model = vision_model_loader(
            model_type=vision_model_type,
            load_pretrained_from_path=checkpoint_path,
            dataset=unlearning_dataset,
        )
        super().__init__(model=model)

        self.unlearning_dataset: UnlearningDataset = get_unlearning_dataset(
            dataset=unlearning_dataset,
            forget_class=forget_class,
            batch_size=process_save_batch_size,
        )

        suffix = f'{self.model.model_string}_{self.unlearning_dataset.dataset_name}'
        graph_dataset_dir = (
            graph_dataset_dir
            / f"{suffix}"
        )

        graph_dataset_dir.mkdir(exist_ok=True, parents=True)
        with open(graph_dataset_dir/ f'{suffix}.txt', 'w') as f:
            f.write(str(suffix))
        self.graph_dataset_dir = graph_dataset_dir
        self.mask_layer = self.validate_layer(mask_layer)

        self.save_redundant_features = save_redundant_features
        self.graph_data_cardinaility = graph_data_cardinaility

        self.forget_class = forget_class
        self.process_save_batch_size = process_save_batch_size
        self.device = device
        self.data_loader: DataLoader = self.get_dataloader()
        self.num_vertices = self.get_num_vertices()
        self.weight_feature = self.get_weight_feature()
        self.edge_matrix = self.get_edge_matrix()

        # currently cross entropy loss is suitable across all image classification taskss
        self.loss_fn = nn.CrossEntropyLoss()
        self.analyze()

    def get_dataloader(self, is_train: bool = False) -> DataLoader:
        old_batch_size = self.unlearning_dataset.batch_size
        # Important: we want per sample grad, not batch-averaged grad, hence batch_size=1 for feature extraction
        self.unlearning_dataset.batch_size = 1
        if is_train:
            loader = self.unlearning_dataset.get_train_loader()
        else:
            loader = self.unlearning_dataset.get_val_loader()
        self.unlearning_dataset.batch_size = old_batch_size
        return loader

    def analyze(self) -> None:
        om = math.floor(math.log(self.num_vertices, 10))
        logger.info(
            f"Graph contains {self.num_vertices} | order magnitude 10^{om} vertices."
        )
        om = math.floor(math.log(self.edge_matrix.shape[1], 10))
        logger.info(
            f"Graph contains {self.edge_matrix.shape[1]} | order magnitude 10^{om} edges."
        )

        weight_feature_dim = self.weight_feature.shape
        assert weight_feature_dim == torch.Size((self.num_vertices,))

    def validate_layer(self, mask_layer) -> Union[None, int]:
        if mask_layer is not None:
            try:
                [p for p in self.model.parameters() if p.requires_grad][mask_layer]
            except Exception:
                logger.info(f"Layer {mask_layer} is invalid.")
                exit()
        return mask_layer

    def get_num_vertices(self):
        if self.mask_layer is None:
            return self.count_trainable_parameters()
        return self.get_layer_parameter_count()[self.mask_layer]

    def get_weight_feature(self, save: bool = False) -> Tensor:
        trainable_layers = [
            torch.flatten(p) for p in self.model.parameters() if p.requires_grad
        ]

        if self.mask_layer is None:
            # get all layers
            flattened_weights = torch.cat(trainable_layers)
        else:
            # single layer
            flattened_weights = trainable_layers[self.mask_layer]

        if save:
            file_name: Path = self.graph_dataset_dir / "flattened_model_weights.pt"
            torch.save(flattened_weights, file_name)
            logger.info(f"Weights saved at {file_name}")

        return flattened_weights

    def get_edge_matrix(self) -> Tensor:
        if self.mask_layer is None:
            return self.get_edge_matrix_full_model()

        return self.get_edge_matrix_by_layer()

    def get_edge_matrix_by_layer(self) -> Tensor:
        # assume layer is matrix, as in our experiments
        layer = [p for p in self.model.parameters() if p.requires_grad][self.mask_layer]
        m, n = layer.shape
        enumeration_matrx = torch.arange(
            start=0, end=m * n, step=1, dtype=torch.long
        ).unflatten(dim=0, sizes=(m, n))

        edge_matrix_list = []

        for c in range(n):
            col = enumeration_matrx[:, c]
            O_edges = torch.cartesian_prod(
                col, col
            )  # out edges emanate out of an activation
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

        if self.save_redundant_features:
            file_name: Path = self.graph_dataset_dir / "graph_edge_matrix.pt"
            torch.save(edge_matrix, file_name)
            logger.info(f"Edge matrix saved at {file_name}")

        return edge_matrix

    def get_edge_matrix_full_model(self) -> Tensor:
        dims = self.model.dim_array + [self.model.out_dim]
        assert (
            len(dims) >= 3
        )  # at least a pair of matrices is needed, specified by 3 numbers (a x b) * (b x c)
        assert (
            sum(dims[i] * dims[i + 1] for i in range(len(dims) - 1))
            == self.num_vertices
        )

        v = torch.arange(start=0, end=self.num_vertices, step=1, dtype=torch.long)
        c, weight_enumeration_matrices = 0, []
        for k in range(len(dims) - 1):
            m, n = dims[k + 1], dims[k]
            enumeration_matrix = v[c : c + m * n].unflatten(dim=0, sizes=(m, n))
            weight_enumeration_matrices.append(enumeration_matrix)
            c += m * n

        edge_matrices_list = []
        for j in range(len(weight_enumeration_matrices) - 1):
            A = weight_enumeration_matrices[j + 1]
            B = weight_enumeration_matrices[j]

            assert A.shape[1] == B.shape[0]
            p, q, r = A.shape[0], A.shape[1], B.shape[1]
            for activation_idx in range(q):
                # cross I.O. edges
                forward_edges = torch.cartesian_prod(
                    A[:, activation_idx], B[activation_idx, :]
                ).T
                edge_matrices_list.append(forward_edges)
                del forward_edges
                torch.cuda.empty_cache()

                backward_edges = torch.cartesian_prod(
                    B[activation_idx, :], A[:, activation_idx]
                ).T
                edge_matrices_list.append(backward_edges)
                del backward_edges
                torch.cuda.empty_cache()

                # In edges
                I_edges = torch.cartesian_prod(
                    B[activation_idx, :], B[activation_idx, :]
                )
                masks = I_edges[:, 0] == I_edges[:, 1]
                I_edges = I_edges[~masks]
                edge_matrices_list.append(I_edges.T)
                del I_edges, masks
                torch.cuda.empty_cache()

                # Out edges
                O_edges = torch.cartesian_prod(
                    A[:, activation_idx], A[:, activation_idx]
                )
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

        if self.save_redundant_features:
            file_name: Path = self.graph_dataset_dir / "graph_edge_matrix.pt"
            torch.save(edge_matrix, file_name)
            logger.info(f"Edge matrix saved at {file_name}")

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
            gradients = torch.cat(gradients)
        assert gradients.numel() == self.num_vertices
        return gradients

    def flatten_in_out_activation(
        self, activations: List[Activations]
    ) -> Tuple[Tensor, Tensor]:
        """Due to the need for per-sample gradient, this function currently only support single data batches."""
        if self.mask_layer is None:
            return self.flatten_in_out_activation_full_model(activations)

        return self.flatten_in_out_activation_single_layer(activations)

    def flatten_in_out_activation_single_layer(
        self, activations: List[Activations]
    ) -> Tuple[Tensor, Tensor]:
        # layers weights are matrices in our experiments
        m, n = [p for p in self.model.parameters() if p.requires_grad][
            self.mask_layer
        ].shape

        activation: Activations = activations[self.mask_layer]
        in_activation, out_activation = (
            activation.input_activation[0].squeeze(),
            activation.output_activation[0].squeeze(),
        )
        # logger.info(f'(m,n)=({m}, {n}) | (in, out) activation shape {in_activation.shape[-1]}, {out_activation.shape[-1]}')
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

    def flatten_in_out_activation_full_model(
        self, activations: List[Activations]
    ) -> Tuple[Tensor, Tensor]:
        dims = self.model.dim_array + [self.model.out_dim]

        in_activation_features = []
        out_activation_features = []

        for j in range(len(activations)):
            m, n = dims[j], dims[j + 1]

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

    def save_forward_backward_features(self) -> None:
        """Saves grads and activations."""
        assert (
            self.data_loader
        )  # data loader must be initialized before getting gradient features
        self.model: HookedModel = self.model.to(self.device)
        self.weight_feature = self.weight_feature.to(self.device)

        self.model.capture_mode(is_on=True)
        self.model.eval()

        eumerator = list(enumerate(self.data_loader))
        if self.graph_data_cardinaility is not None:
            logger.info(
                f"Using graph_data_cardinaility = {self.graph_data_cardinaility}"
            )
            eumerator = eumerator[: self.graph_data_cardinaility]

        feature_batch, batch_idx, total_batches = (
            [],
            0,
            math.ceil(len(eumerator) / self.process_save_batch_size),
        )
        input_batch, target_batch = [], []
        for i, (input, target) in eumerator:
            input_batch.append(input)
            target_batch.append(target)
            input, target = input.to(self.device), target.to(self.device)
            # assert torch.all(target == self.forget_class)   # sanity check
            preds = self.model(input)
            loss: Tensor = self.loss_fn(preds, target)
            loss.backward()

            in_feature, out_feature = self.flatten_in_out_activation(
                self.model.activations
            )
            gradients: Tensor = self.flatten_and_zero_grad()

            # dim = [num_vertices, num_features]
            in_feature = self.mean_var_nomralize(in_feature)
            out_feature = self.mean_var_nomralize(out_feature)
            self.mean_var_nomralize(self.weight_feature)
            self.mean_var_nomralize(gradients)

            graph_feature_matrix = torch.column_stack(
                (in_feature, out_feature, self.weight_feature, gradients)
            )
            # logger.info(f'Vector norm {torch.linalg.vector_norm(graph_feature_matrix[0])}, {torch.linalg.vector_norm(graph_feature_matrix[1])}, {torch.linalg.vector_norm(graph_feature_matrix[2])}, {torch.linalg.vector_norm(graph_feature_matrix[3])}')

            # # tracer
            # logger.info(f'{in_feature.shape} | {out_feature.shape} | {self.weight_feature.shape} | {gradients.shape}')
            # logger.info(graph_feature_matrix.shape)

            feature_batch.append(graph_feature_matrix)

            if len(feature_batch) == self.process_save_batch_size:
                batch_idx += 1

                gcn_batch = GCNBatch(
                    feature_batch=torch.stack(feature_batch, dim=0),
                    input_batch=torch.stack(input_batch, dim=0).squeeze(),
                    target_batch=torch.stack(target_batch, dim=0).squeeze(),
                )

                if batch_idx % 10 == 0:
                    logger.info(f"Saving batch {batch_idx} of {total_batches}")
                file_name = self.graph_dataset_dir / f"batch_{batch_idx}.pt"

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
                feature_batch=torch.stack(feature_batch, dim=0),
                input_batch=torch.stack(input_batch, dim=0).squeeze(),
                target_batch=torch.stack(target_batch, dim=0).squeeze(),
            )
            logger.info(f"Saving batch {batch_idx} of {total_batches}")
            file_name = self.graph_dataset_dir / f"batch_{batch_idx}.pt"
            torch.save(gcn_batch, file_name)

            del feature_batch, input_batch, target_batch
            self.model.reset_activations()
            torch.cuda.empty_cache()

    def mean_var_nomralize(self, feature_vector: Tensor) -> Tensor:
        return (feature_vector - feature_vector.mean()) / (feature_vector.std() + 1e-8)

    def get_data(self, idx: int = 0) -> Tuple[Any, Any, Data]:
        """Generate target and PyG graph data"""

        assert (
            self.data_loader
        )  # data loader must be initialized before getting gradient features
        self.model: HookedModel = self.model.to(self.device)
        self.weight_feature = self.weight_feature.to(self.device)

        self.model.capture_mode(is_on=True)
        self.model.eval()

        for i, (input, target) in enumerate(self.data_loader):
            if i == idx:
                input, target = input.to(self.device), target.to(self.device)
                # assert torch.all(target == self.forget_class)   # sanity check
                preds = self.model(input)
                loss: Tensor = self.loss_fn(preds, target)
                loss.backward()

                in_feature, out_feature = self.flatten_in_out_activation(
                    self.model.activations
                )
                gradients: Tensor = self.flatten_and_zero_grad()

                # dim = [num_vertices, num_features]
                in_feature = self.mean_var_nomralize(in_feature)
                out_feature = self.mean_var_nomralize(out_feature)
                weight_feature = self.mean_var_nomralize(self.weight_feature)
                gradient_feature = self.mean_var_nomralize(gradients)

                graph_feature_matrix = torch.column_stack(
                    (in_feature, out_feature, weight_feature, gradient_feature)
                )

                # # tracer
                # logger.info(f'{in_feature.shape} | {out_feature.shape} | {self.weight_feature.shape} | {gradients.shape}')
                # logger.info(graph_feature_matrix.shape)

                self.model.reset_activations()
                torch.cuda.empty_cache()

                return (
                    input,
                    target,
                    Data(x=graph_feature_matrix, edge_index=self.edge_matrix),
                )

    def get_representative_features(
        self, representatives: List[Tuple[Tensor, Tensor]]
    ) -> List[Tuple[Tensor, Tensor]]:
        """The input list consists of (image, label) and output list contains (feature, label).
        This method is currently used for evaluation pipeline.
        """
        """Saves grads and activations."""

        self.model: HookedModel = self.model.to(self.device)
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

            in_feature, out_feature = self.flatten_in_out_activation(
                self.model.activations
            )
            gradients: Tensor = self.flatten_and_zero_grad()

            # dim = [num_vertices, num_features]
            in_feature = self.mean_var_nomralize(in_feature)
            out_feature = self.mean_var_nomralize(out_feature)
            self.mean_var_nomralize(self.weight_feature)
            self.mean_var_nomralize(gradients)

            graph_feature_matrix = torch.column_stack(
                (in_feature, out_feature, self.weight_feature, gradients)
            )
            # # tracer
            # logger.info(f'{in_feature.shape} | {out_feature.shape} | {self.weight_feature.shape} | {gradients.shape}')
            # logger.info(graph_feature_matrix.shape)

            data = (graph_feature_matrix, target)
            graph_features.append(data)

            # logger.info(f'Generated graph feature for representatitive {target.detach()}')
        return graph_features


if __name__ == "__main__":
    import glob

    checkpoint_dir = Path("checkpoints/HookedResnet_MNIST")
    if checkpoint_dir.exists() and len(list(checkpoint_dir.iterdir())):
        resnet_files = sorted(glob.glob(str(checkpoint_dir / "*.pt")))
        generator = GraphGenerator(
            vision_model_type=SupportedVisionModels.HookedResnet,
            unlearning_dataset=SupportedDatasets.MNIST,
            checkpoint_path=resnet_files[0],
        )
        generator.save_forward_backward_features()
    else:
        raise AssertionError(f"No resnet foundin {checkpoint_dir}")
