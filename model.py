from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Tuple, Union

import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.utils.hooks import RemovableHandle
from torch_geometric.nn import GCNConv, MessagePassing
from torchinfo import summary


@dataclass
class Activations:
    module_name: str
    input_activation: torch.Tensor
    output_activation: torch.Tensor

@dataclass
class Weights:
    module_name: str
    module_weights: torch.Tensor

@dataclass
class Gradients:
    module_name: str
    gradient: torch.Tensor


class HookedMNISTClassifier(nn.Module):
    def __init__(self, 
            mnist_classes: int = 10,
            mnist_dim: int = 28*28,
            include_bias: bool = False,
            hidden_dims: List[int] = [256, 512, 64],
    ) -> None:
        
        assert len(hidden_dims) > 0 # requires at least one hidden layer 
        super().__init__()

        assert not include_bias     # current graph generation does not support bias
        
        self.out_dim = mnist_classes
        self.include_bias = include_bias 
        self.dim_array = [mnist_dim] + hidden_dims 
        self.hook_handles_activations: List[RemovableHandle] = []
        self.hook_handles_weights: List[RemovableHandle] = []
        self.hook_handles_gradients: List[RemovableHandle] = []

        self.weights: List[Weights] = []
        self.activations: List[Activations] = []
        self.gradients: List[Gradients] = []

        self.flatten = nn.Flatten()
        self.hidden_layers = nn.ModuleDict({
            f'hidden layer {i+1}': self._make_single_layer(in_dim=self.dim_array[i], out_dim=self.dim_array[i+1])
            for i in range(len(self.dim_array)-1)
        })
        self.classifier = nn.Linear(in_features=self.dim_array[-1], out_features=mnist_classes, bias=self.include_bias)


    def forward(self, x):
        x = self.flatten(x)
        for hidden_layer in self.hidden_layers.values():
            x = hidden_layer(x)
        return self.classifier(x) # un-normalized logits (i.e. not proabilities)

    def _make_single_layer(self, in_dim: int, out_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim, bias=self.include_bias), 
            nn.ReLU()
        )
    
    def _hook_factory_weight(self, module_name: str) -> Callable[[nn.Module, torch.Tensor, torch.Tensor], None]:
        def hook(module, input, output):
            w = Weights(module_name=module_name, module_weights=module)
            self.weights.append(w)
        return hook 
    
    def _hook_factory_activation(self, module_name: str) -> Callable[[nn.Module, torch.Tensor, torch.Tensor], None]:
        def hook(module, input, output):
            a = Activations(module_name=module_name, input_activation=input, output_activation=output)
            self.activations.append(a)
        return hook 
    
    def _hook_factory_gradient(self, module_name: str) -> Callable[[nn.Module, torch.Tensor, torch.Tensor], None]:
        def hook(module, in_grad, out_grad):
            g = Gradients(module_name=module_name, gradient=out_grad)
            self.gradients.append(g)
        return hook

    def register_hooks_weight(self) -> None:

        for layer_name, hidden_layer in self.hidden_layers.items():
            h = hidden_layer.register_forward_hook(self._hook_factory_weight(module_name=layer_name))
            self.hook_handles_weights.append(h)

        h = self.classifier.register_forward_hook(self._hook_factory_weight(module_name='classifier'))
        self.hook_handles_weights.append(h)

    def register_hooks_activation(self) -> None:

        for layer_name, hidden_layer in self.hidden_layers.items():
            h = hidden_layer.register_forward_hook(self._hook_factory_activation(module_name=layer_name))
            self.hook_handles_activations.append(h)

        h = self.classifier.register_forward_hook(self._hook_factory_activation(module_name='classifier'))
        self.hook_handles_activations.append(h)

    def register_hooks_gradient(self) -> None:

        for layer_name, hidden_layer in self.hidden_layers.items():
            h = hidden_layer.register_full_backward_hook(self._hook_factory_gradient(module_name=layer_name))
            self.hook_handles_gradients.append(h)

        h = self.classifier.register_full_backward_hook(self._hook_factory_gradient(module_name='classifier'))
        self.hook_handles_gradients.append(h)


    def destroy_hooks_weight(self) -> None:
        for handle in self.hook_handles_weights:
            handle.remove()

    def destroy_hooks_activation(self) -> None:
        for handle in self.hook_handles_activations:
            handle.remove()

    def destroy_hooks_gradients(self) -> None:
        for handle in self.hook_handles_gradients:
            handle.remove()

    def capture_mode(self, is_on: bool = True) -> None:
        if is_on:
            self.register_hooks_activation()
            # self.register_hooks_weight()
            self.register_hooks_gradient()
        else:
            self.destroy_hooks_activation()
            self.destroy_hooks_weight()
            self.destroy_hooks_gradients
            
            self.activations = []
            self.weights = []
            self.gradients = []


    def reset_activations(self) -> None:
        # self.gradients = []
        self.activations = []   
        # weights are not reset because they are fixed


class MaskingGCNConv(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')

        self.A = nn.Linear(in_channels, out_channels, bias=False) 
        self.B = nn.Linear(in_channels, out_channels, bias=False) 
        self.softmax = nn.Softmax()

    def forward(self, x, edge_index):
       
       # collects mean aggregated message
       avg_msg = self.propagate(edge_index=edge_index, x=x)

       # linear projection
       x = self.A(avg_msg) + self.B(x)
       return self.softmax(x)

    def message(self, x_j):
        return x_j


class MaskingGCN(nn.Module):

    def __init__(self, num_node_features=4, num_message_passing_rounds=17, hidden_dim=32, use_deg_avg=False):
        """
        Current node features:
            1) weight
            2) gradient component: d(loss)/d(weight i)
            3) input activation
            4) output activation

        GCN Message Passing Rounds: 
            Recommended value is (# layers in masked network - 1) to ensure full receptive field. 
            Default arg is 17 for initial ResNet-18 experiments.

        use_deg_avg
            Uses PyG default GCNCov instead of custom defined MaskingGCNConv
        """
        super().__init__()

        self.proj_in = MaskingGCNConv(num_node_features, hidden_dim)

        if use_deg_avg:
            # a stack of message passing layers
            self.conv = nn.ModuleDict({
                f"Convolution layer {i}" : GCNConv(hidden_dim, hidden_dim)
                for i in range(num_message_passing_rounds-1)
            })
        else:
             # a stack of message passing layers
            self.conv = nn.ModuleDict({
                f"Convolution layer {i}" : MaskingGCNConv(hidden_dim, hidden_dim)
                for i in range(num_message_passing_rounds-1)
            })

        self.proj_out = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=0)
            
    def forward(self, x, edge_index):

        x = self.proj_in(x, edge_index)
        for conv_layer in self.conv.values():
            x = self.softmax(conv_layer(x, edge_index))
    
        return self.softmax(self.proj_out(x))
    
