from glob import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
from utils_data import SupportedDatasets
from model import SupportedVisionModels
from loguru import logger
from itertools import product

@dataclass
class xData:
    topK = "Top K"
    kappa = "kappa"


@dataclass
class ReporterConfig:
    metrics_dir: Path = Path("json")
    report_dir: Path = Path("reports")
    x_axis_data: str = xData.kappa


class Reporter:
    def __init__(self, config: ReporterConfig):
        logger.info(f'>>> Current plotting works for fixed topK, variable kappa <<<')
        self.config = config
        self.reports_base_path = Path(config.report_dir)
        self.sort_fn = lambda x: (
            int(re.search(r"kappa-(\d+)\.json", x).group(1))
            if xData.kappa
            else lambda x: int(re.search(r"top-(\d+)\.json", x).group(1))
        )
        self.metrics_paths = sorted(
            glob(os.path.join(self.config.metrics_dir, "*.json")), key=self.sort_fn
        )

        # --------------- init by get_topK_array
        self.mask_layer = None
        self.forget_class = None
        self.classifier_no_intervention_stats = {
            "forget_set_loss": None,
            "forget_set_score": None,
            "retain_set_loss": None,
            "retain_set_score": None,
            "forget_class_probability": None,
        }
        # ---------------------------------------

        # self.topK = self.get_initial_data()
        # self.kappa = self._legacy_get_kappa_array()

        self.x_axis_data: Optional[float] = None # this is set later in plot single experiment set #self.topK if config.x_axis_data == xData.topK else self.kappa
        self.x_axis_label = 'Kappa' # x-axis is hard coded to be kappa for now
        #(xData.topK if config.x_axis_data == xData.topK else xData.kappa)

    def get_experiment_paths(self, vision_model: SupportedVisionModels, vision_dataset: SupportedDatasets, is_folder: bool = False) -> List[Path]:
        """
        Get folder or .json path depending on whether is_folder = True
        The list is sorted by topK value then kappa values
        """
        if is_folder:
            search_pattern = '*'
        else:
            search_pattern = '*/*.json'

        paths = glob(os.path.join(self.config.metrics_dir, search_pattern))

        def characteristic_fn(p: str) -> bool:
            # example string: json/HookedResnet_MNIST_top-8000_kappa_2000/
            experiment_string = p.split('/')[1]
            model, ds = experiment_string.split('_')[:2]
            # logger.info(f'{model}, {ds}')
            return model == vision_model.value and ds == vision_dataset.value

        paths = list(filter(characteristic_fn, paths))
        paths = sorted(paths, key=lambda p: tuple(int(x) for x in re.search(r'top-(\d+)_kappa_(\d+)', p).groups()))
        paths = list(map(lambda s: Path(s), paths))

        # logger.info(paths)
        return paths

    def get_kappa(self, is_kappa: bool = True) -> List[float]:
        if is_kappa:
            key = 'kappa'
        else:
            raise AssertionError("topK is not supported currently")
        data = []
        for p in self.metrics_paths:
            with open(p, "r") as f:
                metrics = json.loads(f.readline().strip())
                if self.mask_layer is None:
                    # init
                    self.mask_layer = metrics["maked_layer"]
                    self.forget_class = metrics["forget_class"]
                    self.classifier_no_intervention_stats[
                        "forget_class_probability"
                    ] = metrics["unlearning_metrics"]["before_mask_probability"]
                    self.classifier_no_intervention_stats["forget_set_loss"] = metrics[
                        "unlearning_metrics"
                    ]["before_masking_loss"]
                    self.classifier_no_intervention_stats["forget_set_score"] = metrics[
                        "unlearning_metrics"
                    ]["before_masking_score"]
                    self.classifier_no_intervention_stats["retain_set_loss"] = metrics[
                        "performance_degradation_metrics"
                    ]["before_masking_loss"]
                    self.classifier_no_intervention_stats["retain_set_score"] = metrics[
                        "unlearning_metrics"
                    ]["before_masking_score"]
                else:
                    # sanity check
                    assert self.mask_layer == metrics["maked_layer"]
                    assert self.forget_class == metrics["forget_class"]

                data.append(metrics[key])
        return data

    def _legacy_get_kappa_array(self) -> List[float]:
        kappa = []
        for p in self.metrics_paths:
            with open(p, "r") as f:
                metrics = json.loads(f.readline().strip())
                kappa.append(metrics["kappa"])
        return kappa

    ############ MIMU ############
    def get_mimu_retain_set_loss(self) -> List[float]:
        """Measures degradation as loss"""
        mimu_retain_loss = []
        for p in self.metrics_paths:
            with open(p, "r") as f:
                metrics = json.loads(f.readline().strip())
                l = metrics["performance_degradation_metrics"]["after_masking_loss"]
                mimu_retain_loss.append(l)
        return mimu_retain_loss

    def get_mimu_retain_set_score(self) -> List[float]:
        """Measures degradation as percent score"""
        mimu_retain_score = []
        for p in self.metrics_paths:
            with open(p, "r") as f:
                metrics = json.loads(f.readline().strip())
                l = metrics["performance_degradation_metrics"]["after_masking_score"]
                mimu_retain_score.append(l)
        return mimu_retain_score

    def get_mimu_forget_set_loss(self) -> List[float]:
        """Measures unlearning as loss"""
        mimu_forget_loss = []
        for p in self.metrics_paths:
            with open(p, "r") as f:
                metrics = json.loads(f.readline().strip())
                l = metrics["unlearning_metrics"]["after_masking_loss"]
                mimu_forget_loss.append(l)
        return mimu_forget_loss

    def get_mimu_forget_set_score(self) -> List[float]:
        """Measures unlearning as percent score"""
        mimu_forget_score = []
        for p in self.metrics_paths:
            with open(p, "r") as f:
                metrics = json.loads(f.readline().strip())
                l = metrics["unlearning_metrics"]["after_masking_score"]
                mimu_forget_score.append(l)
        return mimu_forget_score

    def get_mimu_forget_set_probability(self) -> List[float]:
        """Measures unlearning as average probability of classifier evaluated on forget class."""
        mimu_forget_set_probability = []
        for p in self.metrics_paths:
            with open(p, "r") as f:
                metrics = json.loads(f.readline().strip())
                l = metrics["unlearning_metrics"]["after_mask_probability"]
                mimu_forget_set_probability.append(l)
        return mimu_forget_set_probability

    ############ Random Baseline ############
    def get_random_masking_retain_set_loss(self) -> List[float]:
        """Measures degradation as loss"""
        random_retain_loss = []
        for p in self.metrics_paths:
            with open(p, "r") as f:
                metrics = json.loads(f.readline().strip())
                l = metrics["performance_degradation_metrics"]["random_masking_loss"]
                random_retain_loss.append(l)
        return random_retain_loss

    def get_random_masking_retain_set_score(self) -> List[float]:
        """Measures degradation as score"""
        random_retain_score = []
        for p in self.metrics_paths:
            with open(p, "r") as f:
                metrics = json.loads(f.readline().strip())
                l = metrics["performance_degradation_metrics"]["random_masking_score"]
                random_retain_score.append(l)
        return random_retain_score

    def get_random_masking_forget_set_loss(self) -> List[float]:
        """Measures degradation as loss"""
        random_forget_loss = []
        for p in self.metrics_paths:
            with open(p, "r") as f:
                metrics = json.loads(f.readline().strip())
                l = metrics["unlearning_metrics"]["random_masking_loss"]
                random_forget_loss.append(l)
        return random_forget_loss

    def get_random_masking_forget_set_score(self) -> List[float]:
        """Measures unlearning as score"""
        random_forget_score = []
        for p in self.metrics_paths:
            with open(p, "r") as f:
                metrics = json.loads(f.readline().strip())
                l = metrics["unlearning_metrics"]["random_masking_score"]
                random_forget_score.append(l)
        return random_forget_score

    def get_random_baseline_forget_set_probability(self) -> List[float]:
        """Measures unlearning as average probability of classifier evaluated on forget class."""
        random_baseline_forget_set_probability = []
        for p in self.metrics_paths:
            with open(p, "r") as f:
                metrics = json.loads(f.readline().strip())
                l = metrics["unlearning_metrics"]["random_masking_probability"]
                random_baseline_forget_set_probability.append(l)
        return random_baseline_forget_set_probability

    ############ SFT Unlearning Baseline ############
    def get_sft_unleraning_retain_set_loss(self) -> List[float]:
        """Measures degradation as loss"""
        sft_retain_loss = []
        for p in self.metrics_paths:
            with open(p, "r") as f:
                metrics = json.loads(f.readline().strip())
                l = metrics["performance_degradation_metrics"]["sft_baseline_loss"]
                sft_retain_loss.append(l)
        return sft_retain_loss

    def get_sft_unlearning_retain_set_score(self) -> List[float]:
        """Measures degradation as score"""
        sft_retain_score = []
        for p in self.metrics_paths:
            with open(p, "r") as f:
                metrics = json.loads(f.readline().strip())
                l = metrics["performance_degradation_metrics"]["sft_baseline_score"]
                sft_retain_score.append(l)
        return sft_retain_score

    def get_sft_unlearning_forget_set_loss(self) -> List[float]:
        """Measures degradation as loss"""
        sft_forget_loss = []
        for p in self.metrics_paths:
            with open(p, "r") as f:
                metrics = json.loads(f.readline().strip())
                l = metrics["unlearning_metrics"]["sft_baseline_loss"]
                sft_forget_loss.append(l)
        return sft_forget_loss

    def get_sft_unlearning_forget_set_score(self) -> List[float]:
        """Measures unlearning as score"""
        sft_retain_score = []
        for p in self.metrics_paths:
            with open(p, "r") as f:
                metrics = json.loads(f.readline().strip())
                l = metrics["unlearning_metrics"]["sft_baseline_score"]
                sft_retain_score.append(l)
        return sft_retain_score

    def get_sft_unlearning_forget_set_probability(self) -> List[float]:
        """Measures unlearning as average probability of classifier evaluated on forget class."""
        sft_forget_set_probability = []
        for p in self.metrics_paths:
            with open(p, "r") as f:
                metrics = json.loads(f.readline().strip())
                l = metrics["unlearning_metrics"]["sft_baseline_probability"]
                sft_forget_set_probability.append(l)
        return sft_forget_set_probability

    def draw_loss_curves_on_retain_set(self) -> None:
        plt.clf()
        plt.title("Loss on Retain Set")
        plt.xlabel(self.x_axis_label)
        plt.ylabel("Cross Entropy")

        x = self.x_axis_data

        y_no_treatment_baseline = [
            self.classifier_no_intervention_stats["retain_set_loss"] for _ in x
        ]
        plt.plot(x, y_no_treatment_baseline, label="original classifier")

        y_random_baseline = self.get_random_masking_retain_set_loss()
        plt.plot(x, y_random_baseline, label="random masking")

        y_mimu = self.get_mimu_retain_set_loss()
        plt.plot(x, y_mimu, label="mimu topK masking")

        y_sft = self.get_sft_unleraning_retain_set_loss()
        plt.plot(x, y_sft, label="sft unlearning")

        # add more baselines

        plt.legend()
        plt.grid(True)

        self.config.report_dir.mkdir(exist_ok=True, parents=True)
        save_path = self.config.report_dir / f"loss_retain_set.png"

        plt.savefig(save_path)
        plt.close()

    def draw_score_curves_on_retain_set(self) -> None:
        plt.clf()
        plt.title("Score on Retain Set")
        plt.xlabel(self.x_axis_label)
        plt.ylabel("% Correct")

        x = self.x_axis_data

        y_no_treatment_baseline = [
            self.classifier_no_intervention_stats["retain_set_score"] for _ in x
        ]
        plt.plot(x, y_no_treatment_baseline, label="original classifier")

        y_random_baseline = self.get_random_masking_retain_set_score()
        plt.plot(x, y_random_baseline, label="random masking")

        y_mimu = self.get_mimu_retain_set_score()
        plt.plot(x, y_mimu, label="mimu topK masking")

        y_sft = self.get_sft_unlearning_retain_set_score()
        plt.plot(x, y_sft, label="sft unlearning")

        # add more baselines

        plt.legend()
        plt.grid(True)

        self.config.report_dir.mkdir(exist_ok=True, parents=True)
        save_path = self.config.report_dir / f"score_retain_set.png"

        plt.savefig(save_path)
        plt.close()

    def draw_classifier_probability_on_forget_class(self) -> None:
        plt.clf()
        plt.title("Probability of Forget Class")
        plt.xlabel(self.x_axis_label)
        plt.ylabel("probability")

        x = self.x_axis_data

        y_no_treatment_baseline = [
            self.classifier_no_intervention_stats["forget_class_probability"] for _ in x
        ]
        plt.plot(x, y_no_treatment_baseline, label="original classifier")

        y_random_baseline = self.get_random_baseline_forget_set_probability()
        plt.plot(x, y_random_baseline, label="random masking")

        y_mimu = self.get_mimu_forget_set_probability()
        plt.plot(x, y_mimu, label="mimu topK masking")

        y_sft = self.get_sft_unlearning_forget_set_probability()
        plt.plot(x, y_sft, label="sft unlearning")

        # add more baselines

        plt.legend()
        plt.grid(True)

        self.config.report_dir.mkdir(exist_ok=True, parents=True)
        save_path = self.config.report_dir / f"probability_on_forget_class.png"

        plt.savefig(save_path)
        plt.close()

    def draw_loss_curves_on_forget_set(self) -> None:
        plt.clf()
        plt.title("Loss on Forget Set")
        plt.xlabel(self.x_axis_label)
        plt.ylabel("Cross Entropy")

        x = self.x_axis_data

        y_no_treatment_baseline = [
            self.classifier_no_intervention_stats["forget_set_loss"] for _ in x
        ]
        plt.plot(x, y_no_treatment_baseline, label="original classifier")

        y_random_baseline = self.get_random_masking_forget_set_loss()
        plt.plot(x, y_random_baseline, label="random masking")

        y_mimu = self.get_mimu_forget_set_loss()
        plt.plot(x, y_mimu, label="mimu topK masking")

        y_sft = self.get_sft_unlearning_forget_set_loss()
        plt.plot(x, y_sft, label="sft unlearning")

        # add more baselines

        plt.legend()
        plt.grid(True)

        self.config.report_dir.mkdir(exist_ok=True, parents=True)
        save_path = self.config.report_dir / f"loss_forget_set.png"

        plt.savefig(save_path)
        plt.close()

    def draw_score_curves_on_forget_set(self) -> None:
        plt.clf()
        plt.title("Score on Forget Set")
        plt.xlabel(self.x_axis_label)
        plt.ylabel("% Correct")

        x = self.x_axis_data
        y_no_treatment_baseline = [
            self.classifier_no_intervention_stats["forget_set_score"] for _ in x
        ]
        plt.plot(x, y_no_treatment_baseline, label="original classifier")

        y_random_baseline = self.get_random_masking_forget_set_score()
        plt.plot(x, y_random_baseline, label="random masking")

        y_mimu = self.get_mimu_forget_set_score()
        plt.plot(x, y_mimu, label="mimu topK masking")

        y_sft = self.get_sft_unlearning_forget_set_score()
        plt.plot(x, y_sft, label="sft unlearning")

        # add more baselines

        plt.legend()
        plt.grid(True)

        self.config.report_dir.mkdir(exist_ok=True, parents=True)
        save_path = self.config.report_dir / f"score_forget_set.png"

        plt.savefig(save_path)
        plt.close()

    def plot_single_experiment_set(self, vision_model: SupportedVisionModels, vision_dataset: SupportedDatasets) -> None:
        """By an experiment set we mean a fixed (vision model and vision dataset) pair and across all possible (topK, kappa) values """
        self.metrics_paths = self.get_experiment_paths(vision_model=vision_model, vision_dataset=vision_dataset, is_folder=False)
        if len(self.metrics_paths) == 0:
            logger.info(f'No data found for ({vision_model.value}, {vision_dataset.value}), skipping.')
            return
        # ===== hard coded to handle kappa only =====
        self.x_axis_data = self.get_kappa() # assume self.metrics_paths have been updated
        inner_folder = f'{vision_model.value}_{vision_dataset.value}'
        self.config.report_dir = self.config.report_dir / inner_folder
        logger.info(f'Saving at {self.config.report_dir}')


        # measures model degradation
        self.draw_loss_curves_on_retain_set()
        self.draw_score_curves_on_retain_set()

        # measures model unlearning
        self.draw_loss_curves_on_forget_set()
        self.draw_score_curves_on_forget_set()

        self.draw_classifier_probability_on_forget_class()

        self.config.report_dir = self.reports_base_path

    def plot(self):
        model_architectures = [
            SupportedVisionModels.HookedMLPClassifier,
            SupportedVisionModels.HookedResnet,
        ]
        supported_datasets = [
            SupportedDatasets.SVHN,
            SupportedDatasets.CIFAR100,
            SupportedDatasets.POKEMON_CLASSIFICATION,
            SupportedDatasets.MNIST,
            SupportedDatasets.CIFAR10,
            SupportedDatasets.IMAGENET_SMALL,
            SupportedDatasets.PLANT_CLASSIFICATION,
        ]
        c = 1
        for ds, ma in product(supported_datasets, model_architectures):
            logger.info(
                f"Plotting experiment set {c} for {ma.value} on {ds.value}"
            )
            self.plot_single_experiment_set(vision_model=ma, vision_dataset=ds)
            c += 1


if __name__ == "__main__":
    config = ReporterConfig()
    reporter = Reporter(config)
    reporter.plot()
    # paths = reporter.get_experiment_paths(vision_model=SupportedVisionModels.HookedMLPClassifier, vision_dataset=SupportedDatasets.MNIST)
    # print(reporter.metrics_paths)
