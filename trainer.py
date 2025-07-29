import copy
import glob
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Type, TypedDict, Union

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from data import GCNBatch
from model import HookedMNISTClassifier, HookedResnet, MaskingGCN
from utils_data import get_cifar10_train_loader, get_cifar10_val_loader

global_config = OmegaConf.load("configs/config.yaml")


@dataclass
class TrainerConfig:
    architecture: Type[nn.Module] = HookedMNISTClassifier
    batch_size: int = 256
    checkpoint_dir: Union[Path, str] = Path("./checkpoints/mnist_classifier")
    data_path: Union[Path, str] = Path("./datasets")
    model_name: str = "mnist_classifier"
    epochs: int = 1
    lr: float = 1e-3
    logging_steps: int = 50
    wandb = None


class Trainer:
    def __init__(self, config: TrainerConfig) -> None:
        self.config = config
        self.model: nn.Module = torch.compile(config.architecture())
        self.criterion = nn.CrossEntropyLoss()
        self.train_data = self.get_train_dataloader()
        self.val_data = self.get_val_dataloader()

        # TODO validate path existence

    def get_train_dataloader(self) -> DataLoader:
        train_data = MNIST(
            root=self.config.data_path, train=True, transform=ToTensor(), download=True
        )
        train_loader = DataLoader(
            train_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

        logger.info(
            f"Batch size: {self.config.batch_size} | # Train batches: {len(train_loader)}"
        )

        return train_loader

    def get_val_dataloader(self) -> DataLoader:
        val_data = MNIST(
            root=self.config.data_path, train=False, transform=ToTensor(), download=True
        )
        val_loader = DataLoader(
            val_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

        logger.info(
            f"Batch size: {self.config.batch_size} | # Validation batches: {len(val_loader)}"
        )

        return val_loader

    def train(
        self, device=global_config.device, step_limit: Union[int, None] = None
    ) -> Path:
        """Returns checkpoint path."""
        adam_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)

        for i in range(self.config.epochs):
            self.model.train()
            self.model = self.model.to(device)
            enumerator = list(enumerate(self.train_data))
            if step_limit is not None:
                enumerator = enumerator[:step_limit]
            for j, (input, target) in enumerator:
                input = input.to(device)
                target = target.to(device)

                preds = self.model(input)
                train_loss: Tensor = self.criterion(preds, target)

                train_loss.backward()
                adam_optimizer.step()
                adam_optimizer.zero_grad()

                if j % self.config.logging_steps == 0:
                    logger.info(
                        f"Epoch {i + 1} | Step {j} | Classifier loss {round(train_loss.item(), 2)}"
                    )
                    self.val()
                    self.model.train()

        checkpoint_path: Path = self.checkpoint()
        logger.info(f"Classifier checkpoints saved to {checkpoint_path}")
        logger.info("Classifier training complete.")
        return checkpoint_path

    def val(self, device=global_config.device) -> None:
        self.model = self.model.to(device)
        self.model.eval()
        test_loss, num_batches = 0, len(self.val_data)
        score, total = 0, len(self.val_data.dataset)
        with torch.no_grad():
            for i, (input, target) in enumerate(self.val_data):
                input = input.to(device)
                target = target.to(device)

                preds: torch.Tensor = self.model(input)
                test_loss += self.criterion(preds, target).item()

                score += (preds.argmax(1) == target).type(torch.float).sum().item()

            test_loss /= num_batches
            score /= total

        logger.info(
            f"Test loss {round(test_loss, 5)} | Score {round(100 * score, 1)} %"
        )

    def checkpoint(self) -> Path:
        """Returns checkpoint path."""
        d: Union[str, Path] = self.config.checkpoint_dir
        if type(d) is str:
            d = Path(d)
        d.mkdir(exist_ok=True, parents=True)

        now = datetime.now().strftime("%Y_%m_%d_%H_%M")
        file_name = self.config.model_name + "_" + now + ".pt"

        checkpoint_path = d / file_name
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path


class Percentage:
    def __init__(self, p):
        self.dec_value = self.convert_to_decimal(p)

    def convert_to_decimal(self, p) -> float:
        assert isinstance(p, float) or isinstance(p, int)
        assert 0 <= p <= 100
        return p / 100


@dataclass
class GCNTrainerConfig:
    src_checkpoint: Path
    gcn_checkpoint_path: Path = Path("checkpoints/gcn")
    lr = 0.01
    weight_decay = 5e-4
    steps = global_config.gcn_train_steps
    mask_layer: Union[None, int] = -2
    mask_K: Union[int, Percentage] = (
        2_500  # number of parameters to keep or some Percentage of the model/layer
    )
    device = global_config.device


class GraphDataLoader:
    def __init__(self, graph_data_dir: Path = Path("datasets/Graphs")):
        self.graph_data_dir = graph_data_dir
        self.edge_matrix = self.load_edge_matrix()
        self.batch_paths = sorted(
            glob.glob(os.path.join(self.graph_data_dir, "batch_*.pt"))
        )
        self.next_batch_path = 0

    def load_edge_matrix(self) -> Tensor:
        file_path = self.graph_data_dir / "graph_edge_matrix.pt"
        return torch.load(file_path)

    def load_weight_feature(self) -> Tensor:
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


class GCNTrainer:
    def __init__(self, config: GCNTrainerConfig) -> None:
        self.config = config
        self.src_model: HookedMNISTClassifier = self.load_src_model()
        self.mask_layer = self.validate_layer(config.mask_layer)
        self.weight_vector = self.vectorize_model()
        self.src_model_layer_dims = self.get_model_signature()
        self.src_model_layer_shapes = self.src_model.dim_array + [
            self.src_model.out_dim
        ]
        self.src_model_dim = (
            self.weight_vector.numel()
        )  # or the number of parameters in the target layer for masking
        self.gcn = MaskingGCN()
        self.graph_data_loader = GraphDataLoader()
        self.device = self.config.device
        self.prior_distribution = self.get_prior_distribution()

        if isinstance(config.mask_K, int):
            assert config.mask_K > 0
            self.K: int = config.mask_K
        elif isinstance(config.mask_K, Percentage):
            self.K: int = math.ceil(config.mask_K.dec_value * self.src_model_dim)

    def load_src_model(self) -> nn.Module:
        model: HookedMNISTClassifier = HookedMNISTClassifier()
        model = torch.compile(model)
        model.load_state_dict(torch.load(self.config.src_checkpoint, weights_only=True))
        return model

    def validate_layer(self, mask_layer) -> Union[None, int]:
        if mask_layer is not None:
            try:
                [p for p in self.src_model.parameters() if p.requires_grad][mask_layer]
            except Exception:
                logger.info(f"Layer {mask_layer} is invalid.")
                exit()
        return mask_layer

    def vectorize_model(self) -> Tensor:
        trainable_layers = [
            torch.flatten(p) for p in self.src_model.parameters() if p.requires_grad
        ]
        if self.mask_layer is None:
            return torch.cat(trainable_layers)
        return trainable_layers[self.mask_layer]

    def get_model_signature(self) -> List[int]:
        counts = [p.numel() for p in self.src_model.parameters() if p.requires_grad]
        if self.mask_layer is None:
            return counts
        return [counts[self.mask_layer]]

    def mask_full_model(self, mask: Tensor) -> nn.Module:
        # assumes HookedMNISTClassifier used so parameters are matrices
        mask = mask.to(self.device)
        self.weight_vector = self.weight_vector.to(self.device)

        model_vect = torch.mul(mask, self.weight_vector)

        i, layers = 0, []
        for k in range(len(self.src_model_layer_dims)):
            n, m = self.src_model_layer_shapes[k : k + 2]
            assert m * n == self.src_model_layer_dims[k]
            j = i + self.src_model_layer_dims[k]
            layer_vect = model_vect[i:j]
            # logger.info(f'{layer_vect.shape}, {m} x {n} = {m*n}')
            layers.append(layer_vect.unflatten(dim=0, sizes=(m, n)))
            i = j

        state_dict = copy.deepcopy(self.src_model.state_dict())
        i = 0
        for key in state_dict.keys():
            state_dict[key] = layers[i]
            i += 1

        model: HookedMNISTClassifier = torch.compile(HookedMNISTClassifier())
        model.load_state_dict(state_dict)
        return model

    def mask_single_layer(self, mask: Tensor) -> HookedMNISTClassifier:
        mask = mask.to(self.device)
        self.weight_vector = self.weight_vector.to(self.device)
        layer_vect = torch.mul(mask, self.weight_vector)
        assert layer_vect.shape[0] == self.src_model_layer_dims[0]
        m, n = [p for p in self.src_model.parameters() if p.requires_grad][
            self.mask_layer
        ].shape
        layer_matrix = layer_vect.unflatten(dim=0, sizes=(m, n))

        state_dict = copy.deepcopy(self.src_model.state_dict())
        key = list(state_dict.keys())[self.mask_layer]
        state_dict[key] = layer_matrix

        model: HookedMNISTClassifier = torch.compile(HookedMNISTClassifier())
        model.load_state_dict(state_dict)
        return model

    def mask_model(self, mask: Tensor) -> HookedMNISTClassifier:
        if self.mask_layer is None:
            return self.mask_full_model(mask)
        return self.mask_single_layer(mask)

    def get_prior_distribution(self) -> Categorical:
        assert self.weight_vector is not None
        weight_vector = self.weight_vector.detach().clone()
        assert not weight_vector.requires_grad
        probs = torch.abs(weight_vector) / torch.linalg.vector_norm(
            weight_vector, ord=1
        )
        probs = probs.to(self.device)
        return Categorical(probs=probs)

    def train(self, plot_stats: bool = True) -> None:
        adam_optimizer = torch.optim.Adam(
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

        for s in range(self.config.steps):
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

                masked_model_probability = F.softmax(masked_model(x), dim=-1).squeeze()[
                    target
                ]

                # 1/3 term of loss
                term1 = torch.log(masked_model_probability)
                loss -= term1

                # 2/3 term of loss
                idx = torch.arange(
                    start=0, end=self.src_model_dim, step=1, device=self.device
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
            if s % 10 == 0:
                logger.info(f"Step {s} | GCN loss {loss.item()}")
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

            save_dir = Path("observability")
            save_dir.mkdir(exist_ok=True, parents=True)
            save_path = save_dir / "gcn_loss_terms_{self.config.mask_K}.png"
            plt.savefig(save_path)

        ckpt_path = self.checkpoint()
        logger.info(f"GCN checkpoint saved at {ckpt_path}")
        logger.info(f"GCN Training complete with final loss {final_loss}")
        return ckpt_path

    def checkpoint(self) -> Path:
        """Returns checkpoint path."""
        d: Union[str, Path] = self.config.gcn_checkpoint_path
        if type(d) is str:
            d = Path(d)
        d.mkdir(exist_ok=True, parents=True)

        now = datetime.now().strftime("%Y_%m_%d_%H_%M")
        file_name = f"gcn_{now}.pt"

        checkpoint_path = d / file_name
        torch.save(self.gcn.state_dict(), checkpoint_path)
        return checkpoint_path


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


@dataclass
class UnlearningSFTConfig:
    init_model: Union[HookedMNISTClassifier, None] = (
        None  # if None, model will load from path
    )
    original_model_path: Union[Path, None] = (
        None  # Path(sorted(glob.glob(os.path.join('checkpoints/mnist_classifier', '*.pt')))[0])
    )
    save_dir: Path = Path("checkpoints/finetuned_mnist_classifier")
    data_dir: Path = Path("datasets")
    finetune_layer: int = -2
    lr: float = 1e-2
    device: str = global_config.device
    steps: int = 30
    batch_size: int = 4
    logging_steps: int = 16
    forget_digit: int = 9


class UnlearningSFT:
    """"""

    def __init__(self, config: UnlearningSFTConfig) -> None:
        self.config = config
        self.model = self.load_forzen_model()
        self.sampling_distribution = Categorical(
            probs=torch.ones(10) / 10
        )  # 10 mnist classes

    def load_forzen_model(self) -> HookedMNISTClassifier:
        """Freezes all layers except finetuning layer."""

        if self.config.init_model is None:
            model: HookedMNISTClassifier = HookedMNISTClassifier()
            model = torch.compile(model)
            model.load_state_dict(
                torch.load(self.config.original_model_path, weights_only=True)
            )
        else:
            model = self.config.init_model

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

    def finetune(
        self, save_checkpoint: bool = False
    ) -> Union[Path, HookedMNISTClassifier]:
        adam_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()
        self.model = self.model.to(self.config.device)

        train_data = list(enumerate(self.mnist_forget_set()))[: self.config.steps]

        for step, (input, _) in train_data:
            input: Tensor = input.to(self.config.device)
            target = self.get_random_targets().to(self.config.device)

            preds = self.model(input)
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

        if save_checkpoint:
            path = self.checkpoint()
            logger.info(f"Finetune checkpoint saved at {path}")

        return self.model

    def mnist_forget_set(self) -> DataLoader:
        dataset = MNIST(
            root=self.config.data_dir, train=False, transform=ToTensor(), download=True
        )
        forget_indices = (dataset.targets == self.config.forget_digit).nonzero(
            as_tuple=True
        )[0]
        forget_set = Subset(dataset, forget_indices)
        return DataLoader(dataset=forget_set, batch_size=self.config.batch_size)

    def checkpoint(self) -> Path:
        """Returns checkpoint path."""
        d: Union[str, Path] = self.config.save_dir
        if type(d) is str:
            d = Path(d)
        d.mkdir(exist_ok=True, parents=True)

        now = datetime.now().strftime("%Y_%m_%d_%H_%M")
        file_name = f"sft_randomize_class_{self.config.forget_digit}_{now}.pt"

        checkpoint_path = d / file_name
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path


class ResnetTrainStatistics:
    def __init__(
        self,
        val_interval_size: int,
        save_dir: Path = Path("observability/Resnet"),
        device: str = global_config.device,
    ):
        self.train_loss = []
        self.test_loss = []
        self.test_score = []
        self.gradient_norm = []
        self.save_dir = save_dir
        self.device = device
        self.val_interval_size = val_interval_size

        save_dir.mkdir(parents=True, exist_ok=True)

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

        save_path = self.save_dir / "train_val_loss.png"
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

        save_path = self.save_dir / "grad_norm.png"
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

        save_path = self.save_dir / "test_score.png"
        plt.savefig(
            save_path, dpi=1000
        )  # You can change the filename and dpi as needed
        plt.clf()

    def plot(self):
        self.plot_loss()
        self.plot_grad_norm()
        self.plot_test_score()


@dataclass
class ResnetTrainerConfig:
    batch_size = 32
    lr = 1e-3
    epochs = 1
    logging_steps = 100
    device = global_config.device
    checkpoint_dir = Path("checkpoints/resnet")
    model_name: str = "resnet"


class ResnetTrainer:
    def __init__(self, config: ResnetTrainerConfig):
        self.config = config
        self.train_statistics = ResnetTrainStatistics(
            val_interval_size=self.config.logging_steps
        )
        self.model = self.get_model()

    def get_model(self) -> nn.Module:
        model = HookedResnet(num_classes=10, unlearning_target_layer_dim=1024)
        return torch.compile(model)

    def train(self, step_limit=1):

        torch.set_float32_matmul_precision("high")
        adam_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        criterion = nn.CrossEntropyLoss()
        train_loader = get_cifar10_train_loader(batch_size=self.config.batch_size)
        self.model.train()
        self.model = self.model.to(self.config.device)

        if step_limit is not None:
            enumerator = list(enumerate(train_loader))[:step_limit]
        else:
            enumerator = enumerate(train_loader)

        for e in range(self.config.epochs):
            for step, (inputs, targets) in enumerator:
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)

                preds = self.model(inputs)
                train_loss: Tensor = criterion(preds, targets)
                train_loss.backward()
                adam_optimizer.step()

                grad_norm: float = self.train_statistics.get_grad_norm(self.model)
                self.train_statistics.record_grad_norm(grad_norm)
                self.train_statistics.record_train_loss(train_loss.item())

                adam_optimizer.zero_grad()

                if step % self.config.logging_steps == 0:
                    logger.info(
                        f"Epoch {e} | Step {step} | ResNet train loss {train_loss.item()} | Grad norm {grad_norm}"
                    )
                    self.train_statistics.plot()
                    self.val()
                    self.model.train()

        checkpoint_path: Path = self.checkpoint()
        logger.info(f"Resnet checkpoints saved to {checkpoint_path}")
        logger.info("Resnet training complete.")
        return checkpoint_path

    def val(self) -> None:
        criterion = nn.CrossEntropyLoss()

        self.model = self.model.to(global_config.device)
        self.model.eval()
        val_loader = get_cifar10_val_loader(batch_size=self.config.batch_size)

        test_loss, num_batches = 0, len(val_loader)
        score, total = 0, len(val_loader.dataset)
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                input = input.to(global_config.device)
                target = target.to(global_config.device)

                preds: torch.Tensor = self.model(input)
                test_loss += criterion(preds, target).item()

                score += (preds.argmax(1) == target).type(torch.float).sum().item()

            test_loss /= num_batches
            score /= total

        self.train_statistics.record_val_loss(test_loss)
        self.train_statistics.record_val_score(score)

        logger.info(
            f"Resnet test loss {round(test_loss, 5)} | Score {round(100 * score, 1)} %"
        )

    def checkpoint(self) -> Path:
        """Returns checkpoint path."""
        d: Union[str, Path] = self.config.checkpoint_dir
        if type(d) is str:
            d = Path(d)
        d.mkdir(exist_ok=True, parents=True)

        now = datetime.now().strftime("%Y_%m_%d_%H_%M")
        file_name = self.config.model_name + "_" + now + ".pt"

        checkpoint_path = d / file_name
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path


if __name__ == "__main__":
    config = ResnetTrainerConfig()
    trainer = ResnetTrainer(config)
    trainer.train()
