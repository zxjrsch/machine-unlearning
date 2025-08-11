from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List, Optional, Type, Union

import requests
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from torch_geometric.nn import GATConv, GCNConv, MessagePassing
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                          pipeline)

from utils_data import SupportedDatasets, get_vision_dataset_classes


class SupportedVisionModels(Enum):
    HookedMLPClassifier = "HookedMLPClassifier"
    HookedResnet = "HookedResnet"


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


class HookedModel(ABC):
    def __init__(self, model_string: str):
        self.hook_handles_activations: List[RemovableHandle] = []
        self.hook_handles_weights: List[RemovableHandle] = []
        self.hook_handles_gradients: List[RemovableHandle] = []

        self.weights: List[Weights] = []
        self.activations: List[Activations] = []
        self.gradients: List[Gradients] = []

        self.model_string = model_string

    @abstractmethod
    def register_hooks_weight(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def register_hooks_activation(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def register_hooks_gradient(self) -> None:
        raise NotImplementedError()

    def _hook_factory_weight(
        self, module_name: str
    ) -> Callable[[nn.Module, torch.Tensor, torch.Tensor], None]:
        def hook(module, input, output):
            w = Weights(module_name=module_name, module_weights=module)
            self.weights.append(w)

        return hook

    def _hook_factory_activation(
        self, module_name: str
    ) -> Callable[[nn.Module, torch.Tensor, torch.Tensor], None]:
        def hook(module, input, output):
            a = Activations(
                module_name=module_name,
                input_activation=input,
                output_activation=output,
            )
            self.activations.append(a)

        return hook

    def _hook_factory_gradient(
        self, module_name: str
    ) -> Callable[[nn.Module, torch.Tensor, torch.Tensor], None]:
        def hook(module, in_grad, out_grad):
            g = Gradients(module_name=module_name, gradient=out_grad)
            self.gradients.append(g)

        return hook

    def destroy_hooks_weight(self) -> None:
        for handle in self.hook_handles_weights:
            handle.remove()

    def destroy_hooks_activation(self) -> None:
        for handle in self.hook_handles_activations:
            handle.remove()

    def destroy_hooks_gradients(self) -> None:
        for handle in self.hook_handles_gradients:
            handle.remove()

    def refresh(self) -> None:
        self.destroy_hooks_activation()
        self.destroy_hooks_weight()
        self.destroy_hooks_gradients()

        self.activations = []
        self.weights = []
        self.gradients = []

    def capture_mode(self, is_on: bool = True) -> None:
        if is_on:
            self.refresh()
            self.register_hooks_activation()
            # self.register_hooks_weight()
            self.register_hooks_gradient()
        else:
            self.refresh()

    def reset_activations(self) -> None:
        # self.gradients = []
        self.activations = []
        # weights are not reset because they are fixed


class HookedMLPClassifier(nn.Module, HookedModel):
    def __init__(
        self,
        num_classes: int = 10,
        image_dim: int = 28 * 28,
        include_bias: bool = False,
        hidden_dims: List[int] = [128, 64],
    ) -> None:
        assert len(hidden_dims) > 0  # requires at least one hidden layer
        HookedModel.__init__(
            self, model_string=SupportedVisionModels.HookedMLPClassifier.value
        )
        nn.Module.__init__(self)

        assert not include_bias  # current graph generation does not support bias

        self.out_dim = num_classes
        self.include_bias = include_bias
        self.dim_array = [image_dim] + hidden_dims

        self.flatten = nn.Flatten()
        self.hidden_layers = nn.ModuleDict(
            {
                f"hidden layer {i + 1}": self._make_single_layer(
                    in_dim=self.dim_array[i], out_dim=self.dim_array[i + 1]
                )
                for i in range(len(self.dim_array) - 1)
            }
        )
        self.classifier = nn.Linear(
            in_features=self.dim_array[-1],
            out_features=num_classes,
            bias=self.include_bias,
        )

    def forward(self, x):
        x = self.flatten(x)
        for hidden_layer in self.hidden_layers.values():
            x = hidden_layer(x)
        # un-normalized logits (i.e. not proabilities)
        return self.classifier(x)

    def _make_single_layer(self, in_dim: int, out_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim, bias=self.include_bias),
            nn.ReLU(),
        )

    def register_hooks_weight(self) -> None:
        for layer_name, hidden_layer in self.hidden_layers.items():
            h = hidden_layer.register_forward_hook(
                self._hook_factory_weight(module_name=layer_name)
            )
            self.hook_handles_weights.append(h)

        h = self.classifier.register_forward_hook(
            self._hook_factory_weight(module_name="classifier")
        )
        self.hook_handles_weights.append(h)

    def register_hooks_activation(self) -> None:
        for layer_name, hidden_layer in self.hidden_layers.items():
            h = hidden_layer.register_forward_hook(
                self._hook_factory_activation(module_name=layer_name)
            )
            self.hook_handles_activations.append(h)

        h = self.classifier.register_forward_hook(
            self._hook_factory_activation(module_name="classifier")
        )
        self.hook_handles_activations.append(h)

    def register_hooks_gradient(self) -> None:
        for layer_name, hidden_layer in self.hidden_layers.items():
            h = hidden_layer.register_full_backward_hook(
                self._hook_factory_gradient(module_name=layer_name)
            )
            self.hook_handles_gradients.append(h)

        h = self.classifier.register_full_backward_hook(
            self._hook_factory_gradient(module_name="classifier")
        )
        self.hook_handles_gradients.append(h)


class MaskingGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="mean")

        self.A = nn.Linear(in_channels, out_channels, bias=False)
        self.B = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        # collects mean aggregated message
        avg_msg = self.propagate(edge_index=edge_index, x=x)

        # linear projection
        return self.A(avg_msg) + self.B(x)

    def message(self, x_j):
        return x_j


class MaskingGCN(nn.Module):
    def __init__(
        self,
        num_node_features=4,
        num_message_passing_rounds=3,
        hidden_dim=32,
        use_deg_avg=False,
        output_logits=True,
        use_relu=False,
    ):
        """
        Current node features:
            1) weight
            2) gradient component: d(loss)/d(weight i)
            3) input activation
            4) output activation

        GCN Message Passing Rounds:
            Recommended value is (# layers in masked network - 1) to ensure full receptive field.
            For example 17 is recommended for ResNet-18 experiments.

        use_deg_avg
            Uses PyG default GCNCov instead of custom defined MaskingGCNConv

        If output_logits is True, outputs of GCN are not normalized.
        """
        super().__init__()

        self.output_logits = output_logits
        self.use_relu = use_relu

        self.proj_in = MaskingGCNConv(num_node_features, hidden_dim)

        if use_deg_avg:
            # a stack of message passing layers
            self.conv = nn.ModuleDict(
                {
                    f"Convolution layer {i}": GCNConv(hidden_dim, hidden_dim)
                    for i in range(num_message_passing_rounds - 1)
                }
            )
        else:
            # a stack of message passing layers
            self.conv = nn.ModuleDict(
                {
                    f"Convolution layer {i}": MaskingGCNConv(hidden_dim, hidden_dim)
                    for i in range(num_message_passing_rounds - 1)
                }
            )

        self.proj_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = self.proj_in(x, edge_index)
        for conv_layer in self.conv.values():
            x = conv_layer(x, edge_index)
            if self.use_relu:
                x = F.relu(x)
            else:
                x = F.sigmoid(x)

        if self.output_logits:
            return self.proj_out(x).squeeze()

        # normalized probability
        return F.softmax(self.proj_out(x), dim=1).squeeze(-1)


# BETA
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        self.dropout = dropout  # drop prob = 0.6
        self.in_features = in_features  #
        self.out_features = out_features  #
        self.alpha = alpha  # LeakyReLU with negative input slope, alpha = 0.2
        self.concat = concat  # conacat = True for all layers except the output layer.

        # Xavier Initialization of Weights
        # Alternatively use weights_init to apply weights of choice
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # Linear Transformation
        h = torch.mm(input, self.W)  # matrix multiplication
        N = h.size()[0]
        # print(N)

        # Attention Mechanism
        a_input = torch.cat(
            [h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1
        ).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # Masked Attention
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


# BETA
class GraphTransformer(torch.nn.Module):
    def __init__(self, num_features=4, num_classes=10):
        self.hid = 8
        self.in_head = 8
        self.out_head = 1

        self.conv1 = GATConv(num_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(
            self.hid * self.in_head,
            num_classes,
            concat=False,
            heads=self.out_head,
            dropout=0.6,
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class HookedResnet(HookedModel, nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        num_in_channels: int = 3,
        unlearning_target_layer_dim: int = 128,
    ) -> None:
        HookedModel.__init__(
            self, model_string=SupportedVisionModels.HookedResnet.value
        )
        nn.Module.__init__(self)

        self.num_classes = num_classes
        self.unlearning_target_layer_dim = unlearning_target_layer_dim

        self.inplanes = 64
        self.dilation = 1

        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(
            num_in_channels,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2, dilate=False)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2, dilate=False)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2, dilate=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # bias are set to False in order to construct GCN graph appropriately
        self.unlearning_target_feedforward = nn.Linear(
            512 * BasicBlock.expansion, unlearning_target_layer_dim, bias=False
        )

        self.fc = nn.Linear(unlearning_target_layer_dim, num_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                nn.BatchNorm2d,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=nn.BatchNorm2d,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.unlearning_target_feedforward(x)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # --- register fwd/bwd hooks for targeted unlearning layer
    def register_hooks_weight(self) -> None:

        h = self.unlearning_target_feedforward.register_forward_hook(
            self._hook_factory_weight(module_name="unlearning_target_feedforward")
        )
        self.hook_handles_weights.append(h)

        h = self.fc.register_forward_hook(self._hook_factory_weight(module_name="fc"))
        self.hook_handles_weights.append(h)

    def register_hooks_activation(self) -> None:
        h = self.unlearning_target_feedforward.register_forward_hook(
            self._hook_factory_activation(module_name="unlearning_target_feedforward")
        )
        self.hook_handles_activations.append(h)

        h = self.fc.register_forward_hook(
            self._hook_factory_activation(module_name="fc")
        )
        self.hook_handles_activations.append(h)

    def register_hooks_gradient(self) -> None:

        h = self.unlearning_target_feedforward.register_full_backward_hook(
            self._hook_factory_gradient(module_name="unlearning_target_feedforward")
        )
        self.hook_handles_gradients.append(h)

        h = self.fc.register_full_backward_hook(
            self._hook_factory_gradient(module_name="fc")
        )
        self.hook_handles_gradients.append(h)

    # ------------------------------------------------------------


def model_factory(
    model_class: Type[nn.Module],
    load_pretrained_from_path: Optional[Path | str] = None,
    compile: bool = True,
    eval_mode: bool = False,
    **kwargs: Any,
) -> nn.Module:
    # logger.info("Model factory received keyword arguments: ")
    # logger.info(kwargs)
    model = model_class(**kwargs) if len(kwargs) > 0 else model_class()

    if compile or load_pretrained_from_path is not None:
        # exiting pretrained models are all compiled
        # model = torch.compile(model)
        if eval_mode:
            # logger.info("eval mode")
            model = model.eval()
        if load_pretrained_from_path is not None and not compile:
            logger.info(
                "Loading pretrained model with compile off is not supported, turning compile on and loading pretrained model."
            )

    if load_pretrained_from_path is not None:
        try:
            model.load_state_dict(
                torch.load(load_pretrained_from_path, weights_only=True)
            )
        except Exception:
            state_dict = torch.load(load_pretrained_from_path, weights_only=True)
            if any(k.startswith("module.") for k in state_dict.keys()):
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    new_state_dict[k.replace("module.", "")] = v
                state_dict = new_state_dict
            model.load_state_dict(state_dict)
        logger.info(f"Loaded pretrained model from {load_pretrained_from_path}")

    return model


class CLIP:
    def __init__(self):
        self.pipeline = pipeline(
            task="zero-shot-image-classification",
            model="openai/clip-vit-base-patch32",
            torch_dtype=torch.bfloat16,
            # device=0,
            device_map="auto",
        )
        self.labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

    def classify(self, image="http://images.cocodataset.org/val2017/000000039769.jpg"):
        return self.pipeline(image=image, candidate_labels=self.labels)


class GemmaVLM:
    def __init__(self):
        self.pipeline = pipeline(
            "image-text-to-text",
            model="google/medgemma-4b-it",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def inference(
        self,
        image_url="https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png",
    ):

        image = Image.open(
            requests.get(image_url, headers={"User-Agent": "example"}, stream=True).raw
        )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert radiologist."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this X-ray"},
                    {"type": "image", "image": image},
                ],
            },
        ]

        output = self.pipeline(text=messages, max_new_tokens=200)
        return output[0]["generated_text"][-1]["content"]


class QwenVLM:
    def __init__(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    def inference(
        self,
        image_url="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    ):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_url,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text


def vision_model_loader(
    model_type: SupportedVisionModels,
    dataset: SupportedDatasets,
    load_pretrained_from_path: Optional[Path | str] = None,
    compile: bool = True,
    eval_mode: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """Loads model with default setting. To override default arguments, pass in model specific **kwargs
    Specify dataset to initalize models with suitable input channels or output classes.
    """

    if model_type == SupportedVisionModels.HookedMLPClassifier:
        if dataset == SupportedDatasets.MNIST:
            return model_factory(
                HookedMLPClassifier,
                load_pretrained_from_path,
                compile,
                eval_mode,
                **kwargs,
            )
        if dataset == SupportedDatasets.CIFAR10:
            return model_factory(
                HookedMLPClassifier,
                load_pretrained_from_path,
                compile,
                eval_mode,
                image_dim=3 * 32 * 32,
                num_classes=get_vision_dataset_classes(SupportedDatasets.CIFAR10),
            )
        if dataset == SupportedDatasets.CIFAR100:
            return model_factory(
                HookedMLPClassifier,
                load_pretrained_from_path,
                compile,
                eval_mode,
                image_dim=3 * 32 * 32,
                num_classes=get_vision_dataset_classes(SupportedDatasets.CIFAR100),
            )
        if dataset == SupportedDatasets.SVHN:
            return model_factory(
                HookedMLPClassifier,
                load_pretrained_from_path,
                compile,
                eval_mode,
                image_dim=3 * 32 * 32,
                num_classes=get_vision_dataset_classes(SupportedDatasets.SVHN),
            )
        if dataset == SupportedDatasets.IMAGENET_SMALL:
            return model_factory(
                HookedMLPClassifier,
                load_pretrained_from_path,
                compile,
                eval_mode,
                image_dim=3 * 128 * 128,
                num_classes=get_vision_dataset_classes(
                    SupportedDatasets.IMAGENET_SMALL
                ),
            )
        if dataset == SupportedDatasets.PLANT_CLASSIFICATION:
            return model_factory(
                HookedMLPClassifier,
                load_pretrained_from_path,
                compile,
                eval_mode,
                image_dim=3 * 128 * 128,
                num_classes=get_vision_dataset_classes(
                    SupportedDatasets.PLANT_CLASSIFICATION
                ),
            )
        if dataset == SupportedDatasets.POKEMON_CLASSIFICATION:
            return model_factory(
                HookedMLPClassifier,
                load_pretrained_from_path,
                compile,
                eval_mode,
                image_dim=3 * 128 * 128,
                num_classes=get_vision_dataset_classes(
                    SupportedDatasets.POKEMON_CLASSIFICATION
                ),
            )
    elif model_type == SupportedVisionModels.HookedResnet:
        if dataset == SupportedDatasets.MNIST:
            return model_factory(
                HookedResnet,
                load_pretrained_from_path,
                compile,
                eval_mode,
                num_in_channels=1,
                num_classes=get_vision_dataset_classes(SupportedDatasets.MNIST),
                **kwargs,
            )
        if dataset == SupportedDatasets.CIFAR10:
            return model_factory(
                HookedResnet,
                load_pretrained_from_path,
                compile,
                eval_mode,
                num_in_channels=3,
                num_classes=get_vision_dataset_classes(SupportedDatasets.CIFAR10),
                **kwargs,
            )
        if dataset == SupportedDatasets.CIFAR100:
            return model_factory(
                HookedResnet,
                load_pretrained_from_path,
                compile,
                eval_mode,
                num_in_channels=3,
                num_classes=get_vision_dataset_classes(SupportedDatasets.CIFAR100),
                **kwargs,
            )

        if dataset == SupportedDatasets.SVHN:
            return model_factory(
                HookedResnet,
                load_pretrained_from_path,
                compile,
                eval_mode,
                num_in_channels=3,
                num_classes=get_vision_dataset_classes(SupportedDatasets.SVHN),
                **kwargs,
            )
        if dataset == SupportedDatasets.IMAGENET_SMALL:
            return model_factory(
                HookedResnet,
                load_pretrained_from_path,
                compile,
                eval_mode,
                num_in_channels=3,
                num_classes=get_vision_dataset_classes(
                    SupportedDatasets.IMAGENET_SMALL
                ),
                **kwargs,
            )
        if dataset == SupportedDatasets.PLANT_CLASSIFICATION:
            return model_factory(
                HookedResnet,
                load_pretrained_from_path,
                compile,
                eval_mode,
                num_in_channels=3,
                num_classes=get_vision_dataset_classes(
                    SupportedDatasets.PLANT_CLASSIFICATION
                ),
                **kwargs,
            )
        if dataset == SupportedDatasets.POKEMON_CLASSIFICATION:
            return model_factory(
                HookedResnet,
                load_pretrained_from_path,
                compile,
                eval_mode,
                num_in_channels=3,
                num_classes=get_vision_dataset_classes(
                    SupportedDatasets.POKEMON_CLASSIFICATION
                ),
                **kwargs,
            )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


if __name__ == "__main__":
    # import code

    # m = HookedResnet()
    # code.interact(local=locals())

    clip_classifier = CLIP()
    out = clip_classifier.classify()
    logger.info(out)

    med_vlm = GemmaVLM()
    out = med_vlm.inference()
    logger.info(out)

    qwen = QwenVLM()
    out = qwen.inference()
    logger.info(out)
