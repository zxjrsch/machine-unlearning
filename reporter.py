import glob
import json
import os
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


@dataclass
class xData:
    topK = 'Top K'
    kappa = 'kappa'

@dataclass
class ReporterConfig:
    metrics_dir: Path = Path('eval/Metrics and Plots/metrics')
    report_dir: Path = Path('reports')
    x_axis_data: str = xData.kappa

class Reporter:
    def __init__(self, config: ReporterConfig):

        self.config = config
        self.metrics_paths = sorted(glob.glob(os.path.join(self.config.metrics_dir, '*.json')))

        # --------------- init by get_topK_array
        self.mask_layer = None  
        self.forget_digit = None 
        self.classifier_no_intervention_stats = {
            'forget_set_loss': None, 
            'forget_set_score': None, 
            'retain_set_loss': None,
            'retain_set_score': None, 
            'forget_digit_probability': None,
        }
        # ---------------------------------------

        self.topK = self.get_topK_array()
        self.kappa = self.get_kappa_array()

        self.x_axis_data = self.topK if config.x_axis_data == xData.topK else self.kappa
        self.x_axis_label = xData.topK if config.x_axis_data == xData.topK else xData.kappa

    def get_topK_array(self) -> List[float]:

        topK = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                if self.mask_layer is None:
                    # init
                    self.mask_layer = metrics['maked_layer']
                    self.forget_digit = metrics['forget_digit']
                    self.classifier_no_intervention_stats['forget_digit_probability'] = metrics['unlearning_metrics']['before_mask_probability']
                    self.classifier_no_intervention_stats['forget_set_loss'] = metrics['unlearning_metrics']['before_masking_loss']
                    self.classifier_no_intervention_stats['forget_set_score'] = metrics['unlearning_metrics']['before_masking_score']
                    self.classifier_no_intervention_stats['retain_set_loss'] = metrics['performance_degradation_metrics']['before_masking_loss']
                    self.classifier_no_intervention_stats['retain_set_score'] = metrics['unlearning_metrics']['before_masking_score']
                else:
                    # sanity check 
                    assert self.mask_layer == metrics['maked_layer']
                    assert self.forget_digit == metrics['forget_digit']
                    
                topK.append(metrics['top_k_value'])
        return topK
    
    def get_kappa_array(self) -> List[float]:

        kappa = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                kappa.append(metrics['kappa'])
        return kappa
    
    ############ MIMU ############
    def get_mimu_retain_set_loss(self) -> List[float]:
        """Measures degradation as loss"""
        mimu_retain_loss = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                l = metrics['performance_degradation_metrics']['after_masking_loss']
                mimu_retain_loss.append(l)
        return mimu_retain_loss
    
    def get_mimu_retain_set_score(self) -> List[float]:
        """Measures degradation as percent score"""
        mimu_retain_score = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                l = metrics['performance_degradation_metrics']['after_masking_score']
                mimu_retain_score.append(l)
        return mimu_retain_score
    
    def get_mimu_forget_set_loss(self) -> List[float]:
        """Measures unlearning as loss"""
        mimu_forget_loss = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                l = metrics['unlearning_metrics']['after_masking_loss']
                mimu_forget_loss.append(l)
        return mimu_forget_loss
    
    def get_mimu_forget_set_score(self) -> List[float]:
        """Measures unlearning as percent score"""
        mimu_forget_score = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                l = metrics['unlearning_metrics']['after_masking_score']
                mimu_forget_score.append(l)
        return mimu_forget_score
    
    def get_mimu_forget_set_probability(self) -> List[float]:
        """Measures unlearning as average probability of classifier evaluated on forget digit."""
        mimu_forget_set_probability = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                l = metrics['unlearning_metrics']['after_mask_probability']
                mimu_forget_set_probability.append(l)
        return mimu_forget_set_probability
    

    ############ Random Baseline ############
    def get_random_masking_retain_set_loss(self) -> List[float]:
        """Measures degradation as loss"""
        random_retain_loss = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                l = metrics['performance_degradation_metrics']['random_masking_loss']
                random_retain_loss.append(l)
        return random_retain_loss
    
    def get_random_masking_retain_set_score(self) -> List[float]:
        """Measures degradation as score"""
        random_retain_score = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                l = metrics['performance_degradation_metrics']['random_masking_score']
                random_retain_score.append(l)
        return random_retain_score

    def get_random_masking_forget_set_loss(self) -> List[float]:
        """Measures degradation as loss"""
        random_forget_loss = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                l = metrics['unlearning_metrics']['random_masking_loss']
                random_forget_loss.append(l)
        return random_forget_loss
    
    def get_random_masking_forget_set_score(self) -> List[float]:
        """Measures unlearning as score"""
        random_forget_score = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                l = metrics['unlearning_metrics']['random_masking_score']
                random_forget_score.append(l)
        return random_forget_score
    
    def get_random_baseline_forget_set_probability(self) -> List[float]:
        """Measures unlearning as average probability of classifier evaluated on forget digit."""
        random_baseline_forget_set_probability = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                l = metrics['unlearning_metrics']['random_masking_probability']
                random_baseline_forget_set_probability.append(l)
        return random_baseline_forget_set_probability
    

    ############ SFT Unlearning Baseline ############
    def get_sft_unleraning_retain_set_loss(self) -> List[float]:
        """Measures degradation as loss"""
        sft_retain_loss = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                l = metrics['performance_degradation_metrics']['sft_baseline_loss']
                sft_retain_loss.append(l)
        return sft_retain_loss
    
    def get_sft_unlearning_retain_set_score(self) -> List[float]:
        """Measures degradation as score"""
        sft_retain_score = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                l = metrics['performance_degradation_metrics']['sft_baseline_score']
                sft_retain_score.append(l)
        return sft_retain_score

    def get_sft_unlearning_forget_set_loss(self) -> List[float]:
        """Measures degradation as loss"""
        sft_forget_loss = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                l = metrics['unlearning_metrics']['sft_baseline_loss']
                sft_forget_loss.append(l)
        return sft_forget_loss
    
    def get_sft_unlearning_forget_set_score(self) -> List[float]:
        """Measures unlearning as score"""
        sft_retain_score = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                l = metrics['unlearning_metrics']['sft_baseline_score']
                sft_retain_score.append(l)
        return sft_retain_score
    
    def get_sft_unlearning_forget_set_probability(self) -> List[float]:
        """Measures unlearning as average probability of classifier evaluated on forget digit."""
        sft_forget_set_probability = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                l = metrics['unlearning_metrics']['sft_baseline_probability']
                sft_forget_set_probability.append(l)
        return sft_forget_set_probability

    
    def draw_loss_curves_on_retain_set(self) -> None:
        plt.clf()
        plt.title("Loss on Retain Set")
        plt.xlabel(self.x_axis_label)
        plt.ylabel("Cross Entropy")

        x = self.x_axis_data

        y_no_treatment_baseline = [self.classifier_no_intervention_stats['retain_set_loss'] for _ in x]
        plt.plot(x, y_no_treatment_baseline, label='original classifier')

        y_random_baseline = self.get_random_masking_retain_set_loss()
        plt.plot(x, y_random_baseline, label='random masking')

        y_mimu = self.get_mimu_retain_set_loss()
        plt.plot(x, y_mimu, label='mimu topK masking')

        y_sft = self.get_sft_unleraning_retain_set_loss()
        plt.plot(x, y_sft, label='sft unlearning')

        # add more baselines 

        plt.legend()
        plt.grid(True)

        self.config.report_dir.mkdir(exist_ok=True, parents=True)
        save_path = self.config.report_dir / f'loss_retain_set.png'

        plt.savefig(save_path)
        plt.close()

    def draw_score_curves_on_retain_set(self) -> None:
        plt.clf()
        plt.title("Score on Retain Set")
        plt.xlabel(self.x_axis_label)
        plt.ylabel("% Correct")

        x = self.x_axis_data

        y_no_treatment_baseline = [self.classifier_no_intervention_stats['retain_set_score'] for _ in x]
        plt.plot(x, y_no_treatment_baseline, label='original classifier')

        y_random_baseline = self.get_random_masking_retain_set_score()
        plt.plot(x, y_random_baseline, label='random masking')

        y_mimu = self.get_mimu_retain_set_score()
        plt.plot(x, y_mimu, label='mimu topK masking')

        y_sft = self.get_sft_unlearning_retain_set_score()
        plt.plot(x, y_sft, label='sft unlearning')

        # add more baselines 

        plt.legend()
        plt.grid(True)

        self.config.report_dir.mkdir(exist_ok=True, parents=True)
        save_path = self.config.report_dir / f'score_retain_set.png'

        plt.savefig(save_path)
        plt.close()

    def draw_classifier_probability_on_forget_class(self) -> None:
        plt.clf()
        plt.title("Probability of Forget Class")
        plt.xlabel(self.x_axis_label)
        plt.ylabel("probability")

        x = self.x_axis_data

        y_no_treatment_baseline = [self.classifier_no_intervention_stats['forget_digit_probability'] for _ in x]
        plt.plot(x, y_no_treatment_baseline, label='original classifier')

        y_random_baseline = self.get_random_baseline_forget_set_probability()
        plt.plot(x, y_random_baseline, label='random masking')

        y_mimu = self.get_mimu_forget_set_probability()
        plt.plot(x, y_mimu, label='mimu topK masking')

        y_sft = self.get_sft_unlearning_forget_set_probability()
        plt.plot(x, y_sft, label='sft unlearning')

        # add more baselines 

        plt.legend()
        plt.grid(True)

        self.config.report_dir.mkdir(exist_ok=True, parents=True)
        save_path = self.config.report_dir / f'probability_on_forget_class.png'

        plt.savefig(save_path)
        plt.close()


    def draw_loss_curves_on_forget_set(self) -> None:
        plt.clf()
        plt.title("Loss on Forget Set")
        plt.xlabel(self.x_axis_label)
        plt.ylabel("Cross Entropy")

        x = self.x_axis_data

        y_no_treatment_baseline = [self.classifier_no_intervention_stats['forget_set_loss'] for _ in x]
        plt.plot(x, y_no_treatment_baseline, label='original classifier')

        y_random_baseline = self.get_random_masking_forget_set_loss()
        plt.plot(x, y_random_baseline, label='random masking')

        y_mimu = self.get_mimu_forget_set_loss()
        plt.plot(x, y_mimu, label='mimu topK masking')

        y_sft = self.get_sft_unlearning_forget_set_loss()
        plt.plot(x, y_sft, label='sft unlearning')

        # add more baselines 

        plt.legend()
        plt.grid(True)

        self.config.report_dir.mkdir(exist_ok=True, parents=True)
        save_path = self.config.report_dir / f'loss_forget_set.png'

        plt.savefig(save_path)
        plt.close()

    def draw_score_curves_on_forget_set(self) -> None:
        plt.clf()
        plt.title("Score on Forget Set")
        plt.xlabel(self.x_axis_label)
        plt.ylabel("% Correct")

        x = self.x_axis_data
        y_no_treatment_baseline = [self.classifier_no_intervention_stats['forget_set_score'] for _ in x]
        plt.plot(x, y_no_treatment_baseline, label='original classifier')

        y_random_baseline = self.get_random_masking_forget_set_score()
        plt.plot(x, y_random_baseline, label='random masking')

        y_mimu = self.get_mimu_forget_set_score()
        plt.plot(x, y_mimu, label='mimu topK masking')

        y_sft = self.get_sft_unlearning_forget_set_score()
        plt.plot(x, y_sft, label='sft unlearning')

        # add more baselines 

        plt.legend()
        plt.grid(True)

        self.config.report_dir.mkdir(exist_ok=True, parents=True)
        save_path = self.config.report_dir / f'score_forget_set.png'

        plt.savefig(save_path)
        plt.close()

    def plot(self) -> None:
        # measures model degradation 
        self.draw_loss_curves_on_retain_set()
        self.draw_score_curves_on_retain_set()

        # measures model unlearning
        self.draw_loss_curves_on_forget_set()
        self.draw_score_curves_on_forget_set()


        self.draw_classifier_probability_on_forget_class()

if __name__ == '__main__':
    config = ReporterConfig()
    reporter = Reporter(config)
    reporter.plot()