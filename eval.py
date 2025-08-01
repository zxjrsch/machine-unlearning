import copy
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.utils.data import DataLoader

from data import GraphGenerator
from model import MaskingGCN, SupportedVisionModels, vision_model_loader
from sampler import DifferentiableTopK, GumbelSampler
from trainer import SFTConfig, SFTModes, SFTTrainer
from utils_data import (SupportedDatasets, get_unlearning_dataset,
                        get_vision_dataset_classes)

global_config = OmegaConf.load("/home/claire/mimu/configs/config.yaml")


@dataclass
class EvalConfig:

    # vision model
    vision_model: SupportedVisionModels
    vision_model_path: Path

    # dataset
    vision_dataset: SupportedVisionModels

    # gcn model
    gcn_path: Optional[Path] = None

    # unlearning hyperparameters
    forget_class: int = 0
    mask_layer: int = -2
    topK: int = 7000
    kappa: int = 5000
    use_set_difference_masking_strategy: bool = False

    use_sinkhorn_sampler: bool = True

    # select finetuning options
    sft_steps: int = 50
    sft_mode: SFTModes = SFTModes.Randomize_Forget

    # subfolder will be created to suit model / dataset
    gcn_base_path: Path = Path("checkpoints/gcn/")
    graph_data_base_path: Path = Path("eval/Graphs")
    metrics_base_path: Path = Path("eval/Metrics and Plots")

    batch_size: int = 256
    device: str = global_config["device"]

    # plotting
    draw_eval_plots: bool = True
    plot_category_probabilities: bool = True


class Eval:
    def __init__(self, config: EvalConfig):

        self.config = config
        self.draw_eval_plots = config.draw_eval_plots
        self.use_set_difference_masking_strategy = (
            config.use_set_difference_masking_strategy
        )

        self.classifier = vision_model_loader(
            model_type=config.vision_model,
            dataset=config.vision_dataset,
            load_pretrained_from_path=config.vision_model_path,
        )

        self.gcn = self.load_gcn()
        self.dataset = get_unlearning_dataset(
            dataset=config.vision_dataset,
            batch_size=config.batch_size,
            forget_class=config.forget_class,
        )

        self.graph_generator = self.load_graph_generator()

        if config.use_sinkhorn_sampler:
            self.sampler = DifferentiableTopK(k=self.config.topK).to(config.device)
        else:
            self.sampler = GumbelSampler(k=self.config.topK).to(config.device)

        # sft unlearning baseline
        self.finetuned_unlearning_model = self.train_sft_model()

    def get_model_graph_date_str(self, include_date: bool = False) -> str:
        if include_date:
            return f'{self.config.vision_model.value}_{self.config.vision_dataset.value}_{datetime.now().strftime("%d_%H_%M")}'
        else:
            return (
                f"{self.config.vision_model.value}_{self.config.vision_dataset.value}"
            )

    def get_metric_save_dir(self) -> str:
        p = (
            self.config.metrics_base_path
            / f"{self.config.vision_model.value}_{self.config.vision_dataset.value}_top-{self.config.topK}_kappa_{self.config.kappa}"
        )
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_raw_json_metric_save_dir(self) -> str:
        p = (
            self.config.metrics_base_path
            / f"json/{self.config.vision_model.value}_{self.config.vision_dataset.value}_top-{self.config.topK}_kappa_{self.config.kappa}"
        )
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_metrics_filename_prefix_str(self, include_date: bool = False) -> str:
        if include_date:
            return f'{self.config.vision_model.value}_{self.config.vision_dataset.value}_top-{self.config.topK}_kappa_{self.config.kappa}_{datetime.now().strftime("%d_%H_%M")}'
        else:
            return f"{self.config.vision_model.value}_{self.config.vision_dataset.value}_top-{self.config.topK}_kappa_{self.config.kappa}"

    def get_gcn_path(self) -> Path:
        if self.config.gcn_path is not None:
            return self.config.gcn_path
        else:
            logger.info("No GCN path given, guessing ...")
            pattern = (
                self.config.gcn_base_path
                / f"GCN_{self.get_model_graph_date_str()}/*.pt"
            )
            return sorted(glob(str(pattern)))[-1]

    def get_graph_data_dir(self) -> Path:
        # Path("eval/Graphs")
        path = self.config.graph_data_base_path / self.get_model_graph_date_str(
            include_date=True
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load_graph_generator(self) -> GraphGenerator:
        return GraphGenerator(
            vision_model_type=self.config.vision_model,
            unlearning_dataset=self.config.vision_dataset,
            checkpoint_path=self.config.vision_model_path,
            process_save_batch_size=1,  # we want per-sample grad
            forget_class=self.config.forget_class,
            save_redundant_features=False,
        )

    def load_gcn(self) -> MaskingGCN:
        model = MaskingGCN()
        model.load_state_dict(torch.load(self.get_gcn_path()))
        return model

    def get_vision_class_representatives(self) -> List[Tuple[Tensor, Tensor]]:
        old_batch_size = self.dataset.batch_size
        self.dataset.reset_batch_size(new_batch_size=1)

        representatives = []
        for d in range(get_vision_dataset_classes(self.config.vision_dataset)):
            ds = self.dataset.get_single_class(class_id=d, is_train=True)
            data = next(iter(ds))
            data[0] = data[0]
            representatives.append(data)
        self.dataset.reset_batch_size(new_batch_size=old_batch_size)
        return representatives

    def get_forget_set(self) -> DataLoader:
        return self.dataset.get_forget_set(is_train=False)

    def get_retain_set(self) -> DataLoader:
        return self.dataset.get_retain_set(is_train=False)

    def get_single_class(self, class_id: int) -> DataLoader:
        return self.dataset.get_single_class(class_id=class_id, is_train=False)

    def compute_representative_masks(
        self, perform_sampling: bool = False
    ) -> List[Tuple[Tensor, int]]:
        """Gets masks for all the classes."""
        reps = self.get_vision_class_representatives()

        # a list for all classes
        node_features = self.graph_generator.get_representative_features(reps)
        edge_matrix = self.graph_generator.edge_matrix.to(self.config.device)

        self.gcn.eval()
        self.gcn = self.gcn.to(self.config.device)

        mask_label_list = []
        with torch.no_grad():
            for features, cls in node_features:
                Q_logits: Tensor = self.gcn(x=features, edge_index=edge_matrix)

                if perform_sampling:
                    mask = self.sampler(Q_logits)
                else:
                    topK_indices = Q_logits.topk(k=self.config.topK).indices
                    mask = torch.zeros_like(Q_logits, dtype=torch.float)
                    mask[topK_indices] = 1

                mask_label_list.append((mask, cls.item()))

        return mask_label_list

    def get_model_mask(self) -> Tensor:
        """
        Set different masking strategy computes the set A of important weights for forget class_id prediction,
        and the set B of important weights for the retain class_ids, then takes the set difference C = A - B which
        is considered to be the weights solely responsible for forget class_id, and when masking out set C, the model
        degradation on retain is minimal. Thus the cardinality of #C <= top-K value.

        If is_set_difference_strategy == False then we generate the mask based on feasibility scores of weights.
        """
        mask_label_list = self.compute_representative_masks()

        forget_mask = None
        retain_masks_sum = None

        for mask, class_id in mask_label_list:
            if class_id == self.config.forget_class:
                forget_mask = mask
            else:
                retain_masks_sum = (
                    mask if retain_masks_sum is None else retain_masks_sum + mask
                )

        if self.use_set_difference_masking_strategy:
            # weights important for forget set prediction but not important for retain classes correspond to 1s
            mask = torch.clamp(forget_mask - retain_masks_sum, min=0)

            # negation operation return True for weight values which are are important to predict RETAIN set but not forget data
            mask = mask == 0
            num_keep_weights = mask.sum().item()
            percent = round(
                num_keep_weights / self.graph_generator.num_vertices * 100, 2
            )

            logger.info(
                f"{self.get_model_graph_date_str(include_date=False)} | top-{self.config.topK} | target layer {self.config.mask_layer} | Retaining {num_keep_weights} ({percent} %)"
            )

            return (mask == 0).float()

        else:
            # To handle target layer specified as a negative value such as -2
            num_layers = sum(
                [1 for layer in self.classifier.parameters() if layer.requires_grad]
            )
            target_layer = torch.tensor(
                self.config.mask_layer % num_layers + 1
            )  # index from 1 to avoid taking log(0)

            num_retain_classes = len(mask_label_list) - 1
            assert num_retain_classes == 9
            scores = retain_masks_sum / num_retain_classes - forget_mask

            feasibility_ranking = torch.log(target_layer) / torch.log(1 + scores)
            masked_weight_indices = feasibility_ranking.topk(
                self.config.kappa
            ).indices  # zero weights with top-K feasibility
            mask = torch.ones_like(feasibility_ranking, dtype=torch.float)
            mask[masked_weight_indices] = 0

            # tune kappa to desired loss
            percent = round(
                1 - self.config.kappa / self.graph_generator.num_vertices * 100, 2
            )
            logger.info(
                f"{self.get_model_graph_date_str(include_date=False)} | top-{self.config.topK} | target layer {self.config.mask_layer} | retaining (weights - kappa) = {self.graph_generator.num_vertices - self.config.kappa} ({percent} %) parameters."
            )

            return mask

    def model_mask_forget_class(self) -> nn.Module:
        forget_class_mask = None
        for mask, class_id in self.compute_representative_masks():
            if class_id == self.config.forget_class:
                forget_class_mask = mask
                break

        topK_indices = forget_class_mask.topk(self.config.topK).indices
        mask = torch.zeros_like(forget_class_mask, dtype=torch.float)
        mask[topK_indices] = 1
        return self.get_masked_model(custom_mask=mask)

    def get_masked_model(self, custom_mask: Union[Tensor, None] = None) -> nn.Module:
        """Single layer masking."""
        if custom_mask is None:
            mask: Tensor = self.get_model_mask()
        else:
            mask = custom_mask
        mask = mask.to(self.config.device)
        weight_vector = self.graph_generator.weight_feature.to(self.config.device)
        layer_vect = torch.mul(mask, weight_vector)

        1 - self.config.kappa / self.graph_generator.num_vertices

        # # no significant improvement
        # layer_vect /= retain_rate

        m, n = [p for p in self.classifier.parameters() if p.requires_grad][
            self.config.mask_layer
        ].shape
        layer_matrix = layer_vect.unflatten(dim=0, sizes=(m, n))

        state_dict = copy.deepcopy(self.classifier.state_dict())
        key = list(state_dict.keys())[self.config.mask_layer]
        state_dict[key] = layer_matrix

        model: nn.Module = vision_model_loader(
            model_type=self.config.vision_model,
            dataset=self.config.vision_dataset,
            eval_mode=True,
        )
        model.load_state_dict(state_dict)
        return model

    def get_randomly_masked_model(self) -> nn.Module:
        mask = torch.zeros(self.graph_generator.num_vertices, dtype=torch.float32)
        num_1s = self.graph_generator.num_vertices - self.config.kappa
        assert num_1s > 0
        indices = torch.randperm(self.graph_generator.num_vertices)[:num_1s]
        mask[indices] = 1
        return self.get_masked_model(custom_mask=mask)

    def train_sft_model(self) -> nn.Module:
        sft_config = SFTConfig(
            architecture=self.config.vision_model,
            vision_dataset=self.config.vision_dataset,
            forget_class=self.config.forget_class,
            batch_size=32,
            original_model_path=self.config.vision_model_path,
            finetuning_mode=self.config.sft_mode,
            save_dir=Path("checkpoints/finetuned_mnist_classifier"),
            finetune_layer=self.config.mask_layer,
            lr=1e-2,
            steps=self.config.sft_steps,
            logging_steps=10,
            device=self.config.device,
        )
        trainer = SFTTrainer(sft_config)
        model = trainer.finetune(save_checkpoint=False)
        return model

    def inference(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        is_forget_set: bool = True,
        description: str = None,
    ) -> Dict:
        model = model.to(self.config.device)
        model.eval()

        test_loss, num_batches = 0, len(data_loader)
        score, total = 0, len(data_loader.dataset)
        with torch.no_grad():
            mean_prob = None
            plotted = False  # generate one plot per inference
            for i, (input, target) in enumerate(data_loader):
                input = input.to(self.config.device)
                target = target.to(self.config.device)

                preds: Tensor = model(input)
                classifier_probs: Tensor = F.softmax(preds, dim=-1)

                if (
                    self.config.plot_category_probabilities
                    and not plotted
                    and is_forget_set
                ):
                    # the mean probability is only meaningful over a single class, which is why we check is_forget_set
                    probs: List[float] = classifier_probs.mean(dim=0).tolist()
                    mean_prob = probs[self.config.forget_class]

                    plt.clf()
                    fig, ax = plt.subplots()
                    plt.figure(figsize=(10, 6))
                    plt.bar(range(10), probs, tick_label=[str(i) for i in range(10)])

                    save_path: Path = (
                        self.get_metric_save_dir() / f"probabilities/{description}/"
                    )
                    save_path.mkdir(parents=True, exist_ok=True)
                    save_path = (
                        save_path
                        / f"{self.get_metrics_filename_prefix_str(include_date=True)}.png"
                    )

                    plt.xlabel("Class")
                    plt.ylabel("Probability")
                    plt.title(f"Classifier Probabilities ({description})")
                    plt.xticks(range(10))
                    plt.ylim(0, 1)
                    plt.savefig(save_path)
                    plotted = True

                test_loss += self.graph_generator.loss_fn(preds, target).item()
                score += (
                    (classifier_probs.argmax(1) == target)
                    .type(torch.float)
                    .sum()
                    .item()
                )

                # logger.info(f'@@@@@ Predictions {classifier_probs.tolist()[0]} | Argmax: {preds.argmax(1)[0]} | Forget class_id {self.config.forget_class} @@@@@')

            test_loss /= num_batches
            test_loss = round(test_loss, 5)
            score /= total
            score = round(100 * score, 1)  # percentage

        eval_set = "forget set" if is_forget_set else "retain set"
        logger.info(
            f"Exp: {description} | {self.get_metrics_filename_prefix_str()} | test_loss on {eval_set} {test_loss} | Score {score} %"
        )

        metrics = {
            "experiment": description,
            "eval_data": eval_set,
            "test_loss": test_loss,
            "score": score,
        }

        if is_forget_set:
            metrics["mean_classifier_probability_on_forget_class"] = mean_prob

        return metrics

    def draw_visualization(
        self, loss: List[float], score: List[float], experiment: str = ""
    ) -> None:
        plt.clf()

        # apend new baseline label below
        categories = ["Original", "MIMU", "Random", "SFT"]

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

        # Plot loss
        axs[0].bar(categories, loss, color="blue")
        axs[0].set_title("test_loss")
        axs[0].set_ylabel("Loss")

        # Plot score
        axs[1].bar(categories, score, color="blue")
        axs[1].set_title("Score")
        axs[1].set_ylabel("Score")

        plt.suptitle(f"{experiment} (top-{self.config.topK} kappa-{self.config.kappa})")
        plt.tight_layout()
        save_path: Path = self.get_metric_save_dir() / f"metrics"
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = (
            save_path / f"{self.get_metrics_filename_prefix_str(include_date=True)}.png"
        )

        plt.savefig(save_path)
        plt.close()

    def get_class_probability(self, class_id: int, model: nn.Module) -> List[float]:
        data_loader = self.get_single_class(class_id=class_id)
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
        fig, axes = plt.subplots(
            2, 5, figsize=(20, 8), sharex=True, sharey=True, constrained_layout=True
        )

        fig.suptitle(
            f"Unlearning class_id {self.config.forget_class}", fontsize=16, y=1.05
        )

        x = list(range(10))
        distributions = {}
        for class_id, ax in enumerate(axes.flatten()):
            distribution = self.get_class_probability(class_id=class_id, model=model)
            distributions[class_id] = distribution
            colors = plt.cm.viridis(torch.linspace(0, 1, len(x)).numpy())
            ax.bar(x, distribution, align="center", alpha=0.8, color=colors)
            ax.set_title(f"class_id {class_id}")
            ax.set_xticks(x)
            ax.set_xticklabels(x)
            ax.grid(axis="y", linestyle="--", alpha=0.7)

        fig.supxlabel("Classes", fontsize=14)
        fig.supylabel("Classifier Probability", fontsize=14)

        save_path: Path = (
            self.get_metric_save_dir() / f"probabilities_all_classes/{description}/"
        )
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = (
            save_path / f"{self.get_metrics_filename_prefix_str(include_date=True)}.png"
        )

        plt.savefig(save_path)
        plt.close()
        return distributions

    def draw_weight_distribution(
        self, model: nn.Module, model_name: str, is_abs_val: bool = True
    ) -> None:
        layer_vector: Tensor = [p.flatten() for p in model.parameters()][
            self.config.mask_layer
        ]
        if is_abs_val:
            layer_vector = layer_vector.abs()
        layer_vector = layer_vector.tolist()
        assert len(layer_vector) == self.graph_generator.num_vertices
        plt.clf()
        plt.hist(layer_vector, bins=15, edgecolor="black")
        plt.xlabel("Weight")
        plt.ylabel("Frequency")
        plt.title(f"{model_name} parameter frequency (layer: {self.config.mask_layer})")

        save_path: Path = (
            self.get_metric_save_dir()
            / f"probabilities_all_classes/weight_histograms/{model_name}/"
        )
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = (
            save_path / f"{self.get_metrics_filename_prefix_str(include_date=True)}.png"
        )

        plt.savefig(save_path)
        plt.close()

    def eval_class_probability(self) -> Dict:
        # record dictionaries to json later when needed
        self.draw_class_probability(model=self.classifier, description="classifier")
        self.draw_class_probability(model=self.get_masked_model(), description="mimu")
        self.draw_class_probability(
            model=self.get_randomly_masked_model(), description="random"
        )
        self.draw_class_probability(
            model=self.finetuned_unlearning_model, description="sft"
        )

    def eval_weight_distributions(self) -> None:
        self.draw_weight_distribution(model=self.classifier, model_name="classifier")
        self.draw_weight_distribution(model=self.get_masked_model(), model_name="mimu")
        self.draw_weight_distribution(
            model=self.get_randomly_masked_model(), model_name="random"
        )
        self.draw_weight_distribution(
            model=self.finetuned_unlearning_model, model_name="sft"
        )

    def eval_unlearning(self) -> Dict:
        """Evaluate model before and after MIMU masking on forget set."""

        logger.info("..... eval unlearning .....")
        before_masking_eval_metrics = self.inference(
            description="no masking on forget set",
            model=self.classifier,
            data_loader=self.get_forget_set(),
            is_forget_set=True,
        )

        after_masking_eval_metrics = self.inference(
            description="mimu on forget set",
            model=self.get_masked_model(),
            data_loader=self.get_forget_set(),
            is_forget_set=True,
        )

        random_baseline_eval_metrics = self.inference(
            description="random on forget set (unlearning)",
            model=self.get_randomly_masked_model(),
            data_loader=self.get_forget_set(),
            is_forget_set=True,
        )

        sft_baseline_eval_metrics = self.inference(
            description="SFT baseline on forget set",
            model=self.finetuned_unlearning_model,
            data_loader=self.get_forget_set(),
            is_forget_set=True,
        )

        if self.draw_eval_plots:
            self.draw_visualization(
                loss=[
                    before_masking_eval_metrics["test_loss"],
                    after_masking_eval_metrics["test_loss"],
                    random_baseline_eval_metrics["test_loss"],
                    sft_baseline_eval_metrics["test_loss"],
                ],
                score=[
                    before_masking_eval_metrics["score"],
                    after_masking_eval_metrics["score"],
                    random_baseline_eval_metrics["score"],
                    sft_baseline_eval_metrics["score"],
                ],
                experiment="eval_unlearning_on_forget_set",
            )

        return {
            "experiment": "eval_unlearning",
            # 'top-K value': self.config.topK,
            "eval_data": "forget_set",
            "before_masking_loss": before_masking_eval_metrics["test_loss"],
            "before_masking_score": before_masking_eval_metrics["score"],
            "before_mask_probability": before_masking_eval_metrics[
                "mean_classifier_probability_on_forget_class"
            ],
            "after_masking_loss": after_masking_eval_metrics["test_loss"],
            "after_masking_score": after_masking_eval_metrics["score"],
            "after_mask_probability": after_masking_eval_metrics[
                "mean_classifier_probability_on_forget_class"
            ],
            # random baseline
            "random_masking_loss": random_baseline_eval_metrics["test_loss"],
            "random_masking_score": random_baseline_eval_metrics["score"],
            "random_masking_probability": random_baseline_eval_metrics[
                "mean_classifier_probability_on_forget_class"
            ],
            # sft baseline
            "sft_baseline_loss": sft_baseline_eval_metrics["test_loss"],
            "sft_baseline_score": sft_baseline_eval_metrics["score"],
            "sft_baseline_probability": sft_baseline_eval_metrics[
                "mean_classifier_probability_on_forget_class"
            ],
        }

    def eval_performance_degradation(self) -> Dict:
        """Eval model before and after MIMU masking on retain set."""

        logger.info("..... eval performance degradation .....")

        before_masking_eval_metrics = self.inference(
            description="no masking on retain set",
            model=self.classifier,
            data_loader=self.get_retain_set(),
            is_forget_set=False,
        )

        after_masking_eval_metrics = self.inference(
            description="mimu on retain set",
            model=self.get_masked_model(),
            data_loader=self.get_retain_set(),
            is_forget_set=False,
        )

        random_baseline_eval_metrics = self.inference(
            description="random retain set (degradation)",
            model=self.get_randomly_masked_model(),
            data_loader=self.get_retain_set(),
            is_forget_set=False,
        )

        sft_baseline_eval_metrics = self.inference(
            description="SFT baseline on retain set",
            model=self.finetuned_unlearning_model,
            data_loader=self.get_retain_set(),
            is_forget_set=False,
        )

        if self.draw_eval_plots:
            self.draw_visualization(
                loss=[
                    before_masking_eval_metrics["test_loss"],
                    after_masking_eval_metrics["test_loss"],
                    random_baseline_eval_metrics["test_loss"],
                    sft_baseline_eval_metrics["test_loss"],
                ],
                score=[
                    before_masking_eval_metrics["score"],
                    after_masking_eval_metrics["score"],
                    random_baseline_eval_metrics["score"],
                    sft_baseline_eval_metrics["score"],
                ],
                experiment="eval_performance_degradation_on_retain_set",
            )

        return {
            "experiment": "eval_performance_degradation",
            # 'top-K value': self.config.topK,
            "eval_data": "retain_set",
            "before_masking_loss": before_masking_eval_metrics["test_loss"],
            "before_masking_score": before_masking_eval_metrics["score"],
            "after_masking_loss": after_masking_eval_metrics["test_loss"],
            "after_masking_score": after_masking_eval_metrics["score"],
            # random baseline
            "random_masking_loss": random_baseline_eval_metrics["test_loss"],
            "random_masking_score": random_baseline_eval_metrics["score"],
            # sft baseline
            "sft_baseline_loss": sft_baseline_eval_metrics["test_loss"],
            "sft_baseline_score": sft_baseline_eval_metrics["score"],
        }

    def eval_mask_efficacy(self) -> Dict:
        """Eval whether mask identified weights important for predicting desired forget class."""

        logger.info("..... eval mask efficacy .....")

        before_masking_eval_metrics = self.inference(
            description="no masking on forget set",
            model=self.classifier,
            data_loader=self.get_forget_set(),
            is_forget_set=True,
        )

        after_masking_eval_metrics = self.inference(
            description="pure class mask on forget set",
            model=self.model_mask_forget_class(),
            data_loader=self.get_forget_set(),
            is_forget_set=True,
        )

        random_baseline_eval_metrics = self.inference(
            description="random forget set (efficacy)",
            model=self.get_randomly_masked_model(),
            data_loader=self.get_forget_set(),
            is_forget_set=True,
        )

        sft_baseline_eval_metrics = self.inference(
            description="SFT baseline on forget set",
            model=self.finetuned_unlearning_model,
            data_loader=self.get_forget_set(),
            is_forget_set=False,
        )

        if self.draw_eval_plots:
            self.draw_visualization(
                loss=[
                    before_masking_eval_metrics["test_loss"],
                    after_masking_eval_metrics["test_loss"],
                    random_baseline_eval_metrics["test_loss"],
                    sft_baseline_eval_metrics["test_loss"],
                ],
                score=[
                    before_masking_eval_metrics["score"],
                    after_masking_eval_metrics["score"],
                    random_baseline_eval_metrics["score"],
                    sft_baseline_eval_metrics["score"],
                ],
                experiment="eval_mask_efficacy_on_foget_set",
            )

        return {
            "experiment": "eval_performance_degradation",
            # 'top-K value': self.config.topK,
            "eval_data": "forget_set",
            "before_masking_loss": before_masking_eval_metrics["test_loss"],
            "before_masking_score": before_masking_eval_metrics["score"],
            "after_masking_loss": after_masking_eval_metrics["test_loss"],
            "after_masking_score": after_masking_eval_metrics["score"],
            "random_masking_loss": random_baseline_eval_metrics["test_loss"],
            "random_masking_score": random_baseline_eval_metrics["score"],
            "sft_baseline_loss": sft_baseline_eval_metrics["test_loss"],
            "sft_baseline_score": sft_baseline_eval_metrics["score"],
        }

    def eval(self, save_metrics: bool = True) -> Dict:
        unlearning_metrics = self.eval_unlearning()
        performance_degradation_metrics = self.eval_performance_degradation()
        mask_efficacy_metrics = self.eval_mask_efficacy()

        self.eval_weight_distributions()
        self.eval_class_probability()

        metrics = {
            "id": str(uuid.uuid1()),
            "forget_class": self.config.forget_class,
            "num_graph_vertices": self.graph_generator.num_vertices,
            "num_graph_edges": self.graph_generator.edge_matrix.shape[1],
            "maked_layer": self.config.mask_layer,
            "top_k_value": self.config.topK,
            "kappa": self.config.kappa,
            "unlearning_metrics": unlearning_metrics,
            "performance_degradation_metrics": performance_degradation_metrics,
            "mask_efficacy_metrics": mask_efficacy_metrics,
        }
        if save_metrics:

            file_path: Path = self.get_raw_json_metric_save_dir()

            if self.use_set_difference_masking_strategy:
                file_path = file_path / f"top-{self.config.topK}.json"
            else:
                # use feasibility ranking (topK, kappa)
                file_path = (
                    file_path / f"top-{self.config.topK}-kappa-{self.config.kappa}.json"
                )

            with open(file_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")

        return metrics


if __name__ == "__main__":
    from itertools import product

    model_architectures = [
        SupportedVisionModels.HookedMLPClassifier,
        SupportedVisionModels.HookedResnet,
    ]
    supported_datasets = [
        # SupportedDatasets.MNIST,
        SupportedDatasets.CIFAR10,
        # SupportedDatasets.CIFAR100,
        # SupportedDatasets.SVHN,
        # SupportedDatasets.IMAGENET_SMALL,
        # SupportedDatasets.PLANT_CLASSIFICATION,
        # SupportedDatasets.POKEMON_CLASSIFICATION
    ]
    for ma, ds in product(model_architectures, supported_datasets):

        config = EvalConfig(
            vision_model=ma,
            vision_model_path=sorted(glob(f"checkpoints/{ma.value}_{ds.value}/*.pt"))[
                -1
            ],
            vision_dataset=ds,
        )
        eval = Eval(config)
        # logger.info(eval.get_gcn_path())
        # reps = eval.get_vision_class_representatives()
        # logger.info(len(reps))
        # logger.info(type(reps))
        # logger.info(type(reps[0]))
        # logger.info(reps[0][0].shape)
        # logger.info('=======')
        # logger.info(reps[0][1])
        eval.eval()
