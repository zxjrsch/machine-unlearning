import copy
import glob
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict, Union

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.distributions import Categorical
from tqdm import tqdm

from data import GCNBatch
from model import (HookedMLPClassifier, HookedModel, MaskingGCN,
                   SupportedVisionModels, vision_model_loader)
from utils_data import (SupportedDatasets, get_unlearning_dataset,
                        get_vision_dataset_classes)

global_config = OmegaConf.load("configs/config.yaml")


class VisionModelTrainerStatistics:
    def __init__(
        self,
        vision_model_architecture: SupportedVisionModels,
        vision_dataset: SupportedDatasets,
        val_interval_size: int,
        device: str = global_config["device"],
    ):
        self.vision_model_architecture = vision_model_architecture
        self.vision_dataset = vision_dataset
        self.train_loss = []
        self.test_loss = []
        self.test_score = []
        self.gradient_norm = []
        self.save_dir = self.get_save_dir()
        self.device = device
        self.val_interval_size = val_interval_size

    def get_save_dir(self):
        dir = (
            Path("observability")
            / f"{self.vision_model_architecture.value}_{self.vision_dataset.value}_{datetime.now().strftime("%d_%H_%M")}"
        )
        dir.mkdir(parents=True, exist_ok=True)
        return dir

    def record_train_loss(self, loss: float) -> None:
        self.train_loss.append(loss)

    def record_grad_norm(self, grad_norm: float) -> None:
        self.gradient_norm.append(grad_norm)

    def record_val_loss(self, loss):
        self.test_loss.append(loss)

    def record_val_score(self, score):
        self.test_score.append(score)

    def get_grad_norm(self, model: nn.Module) -> float:

        sum_of_squares = torch.tensor(0.0, device=self.device)
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                sum_of_squares += p.grad.norm(2).pow(2)

        grad_norm = sum_of_squares.sqrt().item()
        return grad_norm

    def plot_loss(self):

        plt.clf()
        x = list(range(len(self.train_loss)))
        plt.plot(x, self.train_loss, label="train loss")

        if len(self.test_loss) > 0:
            z = [self.val_interval_size]
            for _ in self.test_loss[1:]:
                z.append(z[-1] + self.val_interval_size)
            plt.plot(z, self.test_loss, label="test loss")

        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Resnet Loss")

        plt.grid(True)
        plt.legend()

        save_path = (
            self.save_dir / f"train_val_loss_{datetime.now().strftime("%d_%H_%M")}.png"
        )
        plt.savefig(
            save_path, dpi=1000
        )  # You can change the filename and dpi as needed
        plt.clf()

    def plot_grad_norm(self):

        plt.clf()
        x = list(range(len(self.gradient_norm)))
        plt.plot(x, self.gradient_norm)
        plt.xlabel("train step")
        plt.ylabel("grad norm")
        plt.title("Resnet Gradient Norm")
        plt.grid(True)

        save_path = (
            self.save_dir / f"grad_norm_{datetime.now().strftime("%d_%H_%M")}.png"
        )
        plt.savefig(
            save_path, dpi=1000
        )  # You can change the filename and dpi as needed
        plt.clf()

    def plot_test_score(self):
        plt.clf()
        x = list(range(len(self.test_score)))
        plt.plot(x, self.test_score)
        plt.xlabel("test step")
        plt.ylabel("test score")
        plt.title("Resnet Validation Score")
        plt.grid(True)

        save_path = (
            self.save_dir / f"test_score_{datetime.now().strftime("%d_%H_%M")}.png"
        )
        plt.savefig(
            save_path, dpi=1000
        )  # You can change the filename and dpi as needed
        plt.clf()

    def plot(self):
        self.plot_loss()
        self.plot_grad_norm()
        self.plot_test_score()


@dataclass
class VisionModelTrainerConfig:
    architecture: SupportedVisionModels = SupportedVisionModels.HookedResnet
    vision_dataset: SupportedDatasets = SupportedDatasets.MNIST
    checkpoint_dir: Path | str = Path("checkpoints")
    logging_steps: int = 50
    plot_statistics: bool = True
    batch_size: int = 256
    epochs: int = 1
    steps: Optional[int] = 1  # steps per epoch; set to 1 for quick run / debugging
    lr: float = 1e-3
    device: str = global_config["device"]


class VisionModelTrainer:
    def __init__(self, config: VisionModelTrainerConfig) -> None:
        self.config = config
        self.model: nn.Module = vision_model_loader(
            model_type=config.architecture, dataset=config.vision_dataset
        )
        self.validation_dataloader = get_unlearning_dataset(
            dataset=config.vision_dataset, batch_size=config.batch_size
        ).get_val_loader()
        if config.plot_statistics:
            self.statistics = VisionModelTrainerStatistics(
                vision_model_architecture=config.architecture,
                vision_dataset=config.vision_dataset,
                val_interval_size=config.logging_steps,
                device=config.device,
            )

    def train(self, log_completion: bool = False) -> Path:
        """Returns checkpoint path."""
        try:
            torch.set_float32_matmul_precision("high")
        except:
            logger.info("torch float32_matmul_precision not supported")

        adam_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        criterion = nn.CrossEntropyLoss()

        train_dataloader = get_unlearning_dataset(
            dataset=self.config.vision_dataset, batch_size=self.config.batch_size
        ).get_train_loader()

        # images/second
        throughput = []
        for epoch in range(self.config.epochs):

            # prepare model
            self.model.train()
            self.model = self.model.to(self.config.device)

            # prepare data
            a = time.perf_counter()
            step = 0
            for input, target in train_dataloader:
                if self.config.steps is not None and step == self.config.steps:
                    break
                step += 1
                t = time.perf_counter()
                input: Tensor = input.to(self.config.device)
                target: Tensor = target.to(self.config.device)

                preds = self.model(input)
                train_loss: Tensor = criterion(preds, target)

                train_loss.backward()
                adam_optimizer.step()

                if self.config.plot_statistics:
                    grad_norm: float = self.statistics.get_grad_norm(self.model)
                    self.statistics.record_grad_norm(grad_norm)
                    self.statistics.record_train_loss(train_loss.item())

                adam_optimizer.zero_grad()

                throughput.append(self.config.batch_size / (time.perf_counter() - t))

                if (1 + step) % self.config.logging_steps == 0:
                    average_throughput = int(sum(throughput) / len(throughput))
                    logger.info(
                        f"Epoch {epoch} | Step {step} | {self.config.architecture.value} loss {round(train_loss.item(), 2)}| {average_throughput} images/second"
                    )
                    throughput = []
                    a = time.perf_counter()
                    self.val()
                    b = time.perf_counter()
                    logger.info(
                        f"Epoch {epoch} | Step {step} | Validation took {b-a} sec"
                    )
                    self.statistics.plot()
                    self.model.train()

        checkpoint_path: Path = self.checkpoint()
        if log_completion:
            logger.info(
                f"{self.config.architecture.value} checkpoints saved to {checkpoint_path}"
            )
            logger.info(f"{self.config.architecture.value} training complete.")
        return checkpoint_path

    @torch.no_grad()
    def val(self) -> None:
        # statistics
        test_loss = accuracy = 0
        num_batches = len(self.validation_dataloader)
        data_size = len(self.validation_dataloader.dataset)

        # prepare model
        self.model = self.model.to(self.config.device)
        self.model.eval()

        criterion = nn.CrossEntropyLoss()

        for i, (input, target) in enumerate(self.validation_dataloader):
            input: Tensor = input.to(self.config.device)
            target: Tensor = target.to(self.config.device)

            preds: torch.Tensor = self.model(input)
            test_loss += criterion(preds, target).item()

            accuracy += (preds.argmax(1) == target).type(torch.float).sum().item()

        test_loss /= num_batches
        accuracy /= data_size

        if self.config.plot_statistics:
            self.statistics.record_val_loss(test_loss)
            self.statistics.record_val_score(accuracy)

        logger.info(
            f"Test loss {round(test_loss, 5)} | Score {round(100 * accuracy, 1)} %"
        )

    def get_save_dir(self) -> Path:

        if type(self.config.checkpoint_dir) is str:
            self.config.checkpoint_dir = Path(self.config.checkpoint_dir)

        dataset_name = self.config.vision_dataset.value
        model_name = self.config.architecture.value
        main_dir: Path = self.config.checkpoint_dir / f"{model_name}_{dataset_name}"
        main_dir.mkdir(exist_ok=True, parents=True)

        datetime_run_id = datetime.now().strftime("%d_%H_%M")

        return main_dir / f"{model_name}_{dataset_name}_{datetime_run_id}.pt"

    def checkpoint(self) -> Path:
        """Returns checkpoint path."""

        checkpoint_path = self.get_save_dir()
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path


class Percentage:
    def __init__(self, p):
        self.dec_value = self.convert_to_decimal(p)

    def convert_to_decimal(self, p) -> float:
        assert isinstance(p, float) or isinstance(p, int)
        assert 0 <= p <= 100
        return p / 100


class GraphDataLoader:
    def __init__(self, graph_data_dir: Path):
        self.graph_data_dir = graph_data_dir
        self.edge_matrix = self.load_edge_matrix()
        self.batch_paths = sorted(
            glob.glob(os.path.join(self.graph_data_dir, "batch_*.pt"))
        )
        self.next_batch_path = 0

    def load_edge_matrix(self) -> Tensor:
        file_path = self.graph_data_dir / "graph_edge_matrix.pt"
        return torch.load(file_path)

    def _load_weight_feature(self) -> Tensor:
        """legacy code, to store weights, go to GraphGenerator and set save_redundant_features to True"""
        file_path = self.graph_data_dir / "flattened_model_weights.pt"
        return torch.load(file_path)

    def next(self) -> Tuple[GCNBatch, Tensor]:
        """Returns batched feature tensor, and edge matrix."""
        i = self.next_batch_path % len(self.batch_paths)
        file_path = self.batch_paths[i]
        gcn_batch = torch.load(file_path, weights_only=False)
        # logger.info(f'Getting batch {self.next_batch_path} === {i} mod {len(self.batch_paths)}')
        self.next_batch_path += 1
        return gcn_batch, self.edge_matrix


class GCNTrainerStatistics(TypedDict):
    loss_term_1: List[float] = []
    loss_term_2: List[float] = []
    loss_term_3: List[float] = []


@dataclass
class GCNTrainerConfig:
    vision_model_architecture: SupportedVisionModels
    vision_model_path: Path
    vision_dataset: SupportedDatasets

    # specify the partial gcn_dataset_dir, default datasets/Graphs
    # since GCN trainer will assemble the remaining path as <neural net name>_<dataset name>
    gcn_dataset_dir: Path = Path("datasets/Graphs")

    device: str = global_config["device"]
    mask_layer: Union[None, int] = -2
    steps: int = global_config["gcn_train_steps"]
    lr:float = 0.01
    weight_decay: float = 5e-4
    mask_K: Union[int, Percentage] = 2_500
    logging_steps: int = 2
    gcn_checkpoint_path: Path = Path("checkpoints/gcn")


class GCNTrainer:
    def __init__(self, config: GCNTrainerConfig) -> None:
        self.config = config

        self.vision_model: HookedModel = vision_model_loader(
            model_type=config.vision_model_architecture,
            dataset=config.vision_dataset,
            load_pretrained_from_path=config.vision_model_path,
        ).eval()

        self.gcn = MaskingGCN()
        self.graph_data_loader = GraphDataLoader(
            graph_data_dir=self.get_full_graph_data_dir()
        )

        self.mask_layer = self.validate_layer(config.mask_layer)
        self.weight_vector = self.vectorize_model()
        self.vision_model_dim = (
            self.weight_vector.numel()
        )  # or the number of parameters in the target layer for masking

        self.vision_model_layer_dims = self.get_model_signature()

        if (
            config.vision_model_architecture
            == SupportedVisionModels.HookedMLPClassifier
        ):
            self.vision_model_layer_shapes = self.vision_model.dim_array + [
                self.vision_model.out_dim
            ]
        else:
            self.vision_model_layer_shapes = None

        self.device = self.config.device
        self.prior_distribution = self.get_prior_distribution()

        if isinstance(config.mask_K, int):
            assert config.mask_K > 0
            self.K: int = config.mask_K
        elif isinstance(config.mask_K, Percentage):
            self.K: int = math.ceil(config.mask_K.dec_value * self.vision_model_dim)

    def get_full_graph_data_dir(self) -> str:
        return (
            self.config.gcn_dataset_dir
            / f"{self.vision_model.model_string}_{self.config.vision_dataset.value}"
        )

    def validate_layer(self, mask_layer) -> Union[None, int]:
        if mask_layer is not None:
            try:
                [p for p in self.vision_model.parameters() if p.requires_grad][
                    mask_layer
                ]
            except Exception:
                logger.info(f"Layer {mask_layer} is invalid.")
                exit()
        return mask_layer

    def vectorize_model(self) -> Tensor:
        trainable_layers = [
            torch.flatten(p) for p in self.vision_model.parameters() if p.requires_grad
        ]
        if self.mask_layer is None:
            # full model unlearning is not currently supported due to computation burden seen in inital experiments
            return torch.cat(trainable_layers)
        return trainable_layers[self.mask_layer]

    def get_model_signature(self) -> List[int]:
        counts = [p.numel() for p in self.vision_model.parameters() if p.requires_grad]
        if self.mask_layer is None:
            return counts
        return [counts[self.mask_layer]]

    def mask_full_model(self, mask: Tensor) -> nn.Module:
        # assumes HookedMLPClassifier used so parameters are matrices
        # full model unlearning is not currently supported due to computation burden seen in inital experiments

        assert (
            self.config.vision_model_architecture
            == SupportedVisionModels.HookedMLPClassifier
        )
        mask = mask.to(self.device)
        self.weight_vector = self.weight_vector.to(self.device)

        model_vect = torch.mul(mask, self.weight_vector)

        i, layers = 0, []
        for k in range(len(self.vision_model_layer_dims)):
            n, m = self.vision_model_layer_shapes[k : k + 2]
            assert m * n == self.vision_model_layer_dims[k]
            j = i + self.vision_model_layer_dims[k]
            layer_vect = model_vect[i:j]
            # logger.info(f'{layer_vect.shape}, {m} x {n} = {m*n}')
            layers.append(layer_vect.unflatten(dim=0, sizes=(m, n)))
            i = j

        state_dict = copy.deepcopy(self.vision_model.state_dict())
        i = 0
        for key in state_dict.keys():
            state_dict[key] = layers[i]
            i += 1

        model: HookedMLPClassifier = torch.compile(HookedMLPClassifier())
        model.load_state_dict(state_dict)
        return model

    def mask_single_layer(self, mask: Tensor) -> nn.Module:
        mask = mask.to(self.device)
        self.weight_vector = self.weight_vector.to(self.device)
        layer_vect = torch.mul(mask, self.weight_vector)
        assert layer_vect.shape[0] == self.vision_model_layer_dims[0]
        m, n = [p for p in self.vision_model.parameters() if p.requires_grad][
            self.mask_layer
        ].shape
        layer_matrix = layer_vect.unflatten(dim=0, sizes=(m, n))

        state_dict = copy.deepcopy(self.vision_model.state_dict())
        key = list(state_dict.keys())[self.mask_layer]
        state_dict[key] = layer_matrix

        model: nn.Module = vision_model_loader(
            model_type=self.config.vision_model_architecture,
            dataset=self.config.vision_dataset,
        )
        model.load_state_dict(state_dict)
        return model

    def mask_model(self, mask: Tensor) -> nn.Module:
        if self.mask_layer is None:
            return self.mask_full_model(mask).eval()
        return self.mask_single_layer(mask).eval()

    def get_prior_distribution(self) -> Categorical:
        assert self.weight_vector is not None
        weight_vector = self.weight_vector.detach().clone()
        assert not weight_vector.requires_grad
        probs = torch.abs(weight_vector) / torch.linalg.vector_norm(
            weight_vector, ord=1
        )
        probs = probs.to(self.device)
        return Categorical(probs=probs)

    def train(self, plot_stats: bool = True) -> Path:
        adam_optimizer = torch.optim.AdamW(
            self.gcn.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.gcn = self.gcn.to(self.device)
        self.gcn.train()

        # logger.info(f'====== GCN training for top-{self.config.mask_K} MIMU masking | Mask layer: {self.config.mask_layer} =====')
        final_loss = 0

        if plot_stats:
            trainer_stats: GCNTrainerStatistics = {
                "loss_term_1": [],
                "loss_term_2": [],
                "loss_term_3": [],
            }

        for s in tqdm(range(self.config.steps)):
            gcn_batch, edge_matrix = self.graph_data_loader.next()

            feature_batch = gcn_batch.feature_batch.to(self.device)
            input_batch = gcn_batch.input_batch.to(self.device)
            target_batch = gcn_batch.target_batch.to(self.device)
            edge_matrix = edge_matrix.to(self.device)

            batch_size = feature_batch.shape[0]
            # logger.info(f'GCN training batch size: {batch_size}')

            # logger.info(f'Batch {feature_batch.shape} | Edge {edge_matrix.shape} | Input {input_batch.shape} | Target {target_batch.shape}')

            loss: Tensor = torch.tensor(0.0).to(self.device)

            for i in range(batch_size):
                # logger.info(f'step{i}')
                x, target = input_batch[i].unsqueeze(0), target_batch[i]

                emperical_Q_logits: Tensor = self.gcn(
                    x=feature_batch[i], edge_index=edge_matrix
                )
                mask = gumbel_top_k_sampling_v2(logits=emperical_Q_logits, k=self.K).to(
                    self.device
                )

                # # delete the next line
                # mask = F.softmax(emperical_Q_logits, dim=-1)

                masked_model = self.mask_model(mask=mask).to(self.device)

                try:
                    masked_model_probability = F.softmax(
                        masked_model(x), dim=-1
                    ).squeeze()[target]
                except Exception:
                    # logger.info(x.shape)
                    x = x.unsqueeze(1)
                    masked_model_probability = F.softmax(
                        masked_model(x), dim=-1
                    ).squeeze()[target]

                # 1/3 term of loss
                term1 = torch.log(masked_model_probability.detach().clone())
                loss -= term1

                # 2/3 term of loss
                idx = torch.arange(
                    start=0, end=self.vision_model_dim, step=1, device=self.device
                )
                log_probs = self.prior_distribution.log_prob(idx)
                term2 = torch.dot(mask, log_probs)
                loss -= term2

                # 3/3 term of loss
                Q_distribution = Categorical(
                    probs=F.softmax(emperical_Q_logits, dim=-1)
                )
                log_probs: Tensor = Q_distribution.log_prob(idx)
                term3 = torch.dot(mask, log_probs)
                loss += term3

                if plot_stats:
                    trainer_stats["loss_term_1"].append(term1.item())
                    trainer_stats["loss_term_2"].append(term2.item())
                    trainer_stats["loss_term_3"].append(term3.item())

                # logger.info(f'Step {s} sample {i} | term 1 {-term1.item()} | term 2 {-term2.item()} | term 3 {term3.item()}')

            loss /= batch_size  # prevent exploding gradients while optimizing same objective

            loss.backward()
            adam_optimizer.step()
            adam_optimizer.zero_grad()
            if (1 + s) % self.config.logging_steps == 0:
                logger.info(f"Step {s+1} | GCN loss {loss.item()}")
            final_loss = loss.item()

        if plot_stats:
            plt.clf()
            fig, ax = plt.subplots()

            L = len(trainer_stats["loss_term_1"])
            x = list(range(L))
            ax.plot(x, trainer_stats["loss_term_1"], label="term 1")
            ax.plot(x, trainer_stats["loss_term_2"], label="term 2")
            ax.plot(x, trainer_stats["loss_term_3"], label="term 3")

            plt.xlabel("Samples")
            plt.ylabel("Loss")
            plt.title(f"Compare 3 terms in GCN loss.")
            ax.legend()

            mid_path = self.get_save_path(return_dir=True)
            save_dir = Path("observability") / f"GCN_{mid_path}"
            save_dir.mkdir(exist_ok=True, parents=True)
            save_path = (
                save_dir / f"GCN_{mid_path}_loss_terms_top-{self.config.mask_K}.png"
            )
            plt.savefig(save_path)

        ckpt_path = self.checkpoint()
        logger.info(f"GCN checkpoint saved at {ckpt_path}")
        logger.info(f"GCN Training complete with final loss {final_loss}")
        return ckpt_path

    def checkpoint(self) -> Path:
        """Returns checkpoint path."""
        checkpoint_path = self.get_save_path()
        torch.save(self.gcn.state_dict(), checkpoint_path)
        return checkpoint_path

    def get_save_path(self, return_dir: bool = False) -> Path:

        if type(self.config.gcn_checkpoint_path) is str:
            self.config.gcn_checkpoint_path = Path(self.config.gcn_checkpoint_path)

        dataset_name = self.config.vision_dataset.value
        model_name = self.config.vision_model_architecture.value
        main_dir: Path = (
            self.config.gcn_checkpoint_path / f"GCN_{model_name}_{dataset_name}"
        )
        main_dir.mkdir(exist_ok=True, parents=True)

        datetime_run_id = datetime.now().strftime("%d_%H_%M")

        if return_dir:
            return f"{model_name}_{dataset_name}"

        return main_dir / f"GCN_{model_name}_{dataset_name}_{datetime_run_id}.pt"


def gumbel_top_k_sampling_v2(logits, k, temperature=1.0, eps=1e-10) -> Tensor:
    """
    Alternative implementation using continuous relaxation of top-k operation.
    This version maintains better gradients by avoiding hard masking.

    The code for this method is shared by Wuga, see
    https://claude.ai/public/artifacts/138b83ce-f40f-495f-81a7-bc8bd7416fce

    See Also
    [1] https://arxiv.org/pdf/1903.06059
    [2] https://papers.nips.cc/paper_files/paper/2014/file/937debc749f041eb5700df7211ac795c-Paper.pdf
    [3] https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html

    Args:
        logits (torch.Tensor): Input logits of shape (..., vocab_size)
        k (int): Number of top elements to sample
        temperature (float): Temperature parameter
        eps (float): Small constant for numerical stability

    Returns:
        torch.Tensor: Soft top-k samples
    """
    # Sample Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
    gumbel_logits = logits + gumbel_noise

    # Use continuous relaxation of top-k
    # Sort the gumbel logits to find the k-th largest value
    sorted_gumbel, _ = torch.sort(gumbel_logits, dim=-1, descending=True)
    threshold = sorted_gumbel[..., k - 1 : k]  # k-th largest value

    # Create soft mask using sigmoid
    soft_mask = torch.sigmoid((gumbel_logits - threshold) / temperature)

    # Apply soft mask and normalize
    masked_logits = logits * soft_mask
    return F.softmax(masked_logits / temperature, dim=-1)


class SFTModes(Enum):
    Randomize_Forget = "Randomize_Forget"
    Finetune_Retain = "Finetune_Retain"
    Randomize_Forget_And_Finetune_Retain = "Randomize_Forget_And_Finetune_Retain"


@dataclass
class SFTConfig:
    architecture: SupportedVisionModels
    vision_dataset: SupportedDatasets

    # dataset
    forget_class: int = 9
    batch_size: int = 4

    # model
    init_model: Optional[HookedMLPClassifier] = None  # pass in argument
    original_model_path: Optional[Path] = None  # or loaded from path

    # hardware
    device: str = global_config["device"]

    # training
    finetuning_mode: SFTModes = SFTModes.Randomize_Forget
    finetune_layer: int = -2
    lr: float = 1e-2
    steps: int = 30
    logging_steps: int = 10

    # saving
    save_dir: Path = Path("checkpoints/finetuned_models")


class SFTTrainer:
    """"""

    def __init__(self, config: SFTConfig) -> None:
        self.config = config
        self.model = self.load_forzen_model()
        self.dataset = get_unlearning_dataset(
            dataset=config.vision_dataset,
            batch_size=config.batch_size,
            forget_class=config.forget_class,
        )
        self.num_classes = get_vision_dataset_classes(config.vision_dataset)
        self.sampling_distribution = Categorical(
            probs=torch.ones(self.num_classes) / self.num_classes
        )

    def load_forzen_model(self) -> HookedMLPClassifier:
        """Freezes all layers except finetuning layer."""

        if self.config.init_model is not None:
            assert (
                self.config.original_model_path is None
            ), "Must specify exactly one of UnlearningSFTConfig.init_model or UnlearningSFTConfig.original_model_path"
            model = self.config.init_model
        else:
            assert (
                self.config.original_model_path is not None
            ), "Must specify either UnlearningSFTConfig.init_model or UnlearningSFTConfig.original_model_path"
            model = vision_model_loader(
                model_type=self.config.architecture,
                dataset=self.config.vision_dataset,
                load_pretrained_from_path=self.config.original_model_path,
            )

        # To handle target layer specified as a negative value such as -2
        num_layers = sum([1 for layer in model.parameters() if layer.requires_grad])
        target_layer = self.config.finetune_layer % num_layers

        i = 0
        for p in model.parameters():
            if p.requires_grad:
                if i != target_layer:
                    # freeze layer
                    p.requires_grad = False
                i += 1

        assert sum([1 for layer in model.parameters() if layer.requires_grad]) == 1
        return model

    def get_random_targets(self) -> Tensor:
        batch_size = self.config.batch_size
        return self.sampling_distribution.sample((batch_size,))

    def finetune_randomize_forget(self) -> nn.Module:
        adam_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()
        self.model = self.model.to(self.config.device)

        forget_set = self.dataset.get_forget_set()
        step = 0
        for inputs, _ in forget_set:
            if step == self.config.steps:
                break
            step += 1
            inputs: Tensor = inputs.to(self.config.device)
            target = self.get_random_targets().to(self.config.device)

            preds = self.model(inputs)
            train_loss: Tensor = loss_fn(preds, target)
            train_loss.backward()
            adam_optimizer.step()
            adam_optimizer.zero_grad()

            if step % self.config.logging_steps == 0:
                logger.info(
                    f"Finetuning step {step} | classifier loss {train_loss.item()}"
                )

        logger.info(
            f"Finetuning complete | classifier loss wrt randomized target {train_loss.item()}"
        )

        return self.model

    def finetune_retain(self) -> nn.Module:
        adam_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()
        self.model = self.model.to(self.config.device)

        retain_set = self.dataset.get_retain_set()

        step = 0
        for inputs, targets in retain_set:
            if step == self.config.steps:
                break
            step += 1
            inputs: Tensor = inputs.to(self.config.device)
            targets = targets.to(self.config.device)

            preds = self.model(inputs)
            train_loss: Tensor = loss_fn(preds, targets)
            train_loss.backward()
            adam_optimizer.step()
            adam_optimizer.zero_grad()

            if step % self.config.logging_steps == 0:
                logger.info(
                    f"Finetuning step {step} | classifier loss {train_loss.item()}"
                )

        logger.info(
            f"Finetuning complete | classifier loss on retain targets {train_loss.item()}"
        )
        return self.model

    def finetune(self, save_checkpoint: bool = True) -> Union[Path, nn.Module]:
        if self.config.finetuning_mode == SFTModes.Randomize_Forget:
            self.model = self.finetune_randomize_forget()
        elif self.config.finetuning_mode == SFTModes.Finetune_Retain:
            self.model = self.finetune_retain()
        elif (
            self.config.finetuning_mode == SFTModes.Randomize_Forget_And_Finetune_Retain
        ):
            self.model = self.finetune_randomize_forget()
            self.model = self.finetune_retain()

        if save_checkpoint:
            path = self.checkpoint()
            logger.info(f"Finetune checkpoint saved at {path}")

        return self.model

    def checkpoint(self) -> Path:
        """Returns checkpoint path."""
        d: Union[str, Path] = self.config.save_dir
        if type(d) is str:
            d = Path(d)
        d.mkdir(exist_ok=True, parents=True)

        now = datetime.now().strftime("%d_%H_%M")
        file_name = f"sft_{self.config.finetuning_mode.value}_{self.config.finetuning_mode.value}_{self.config.forget_class}_{now}.pt"

        checkpoint_path = d / file_name
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path
