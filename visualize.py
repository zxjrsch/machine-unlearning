import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


@dataclass
class ReporterConfig:
    metrics_dir: Path = Path('eval/Metrics and Plots/metrics')
    report_dir: Path = Path('reports')

class Reporter:
    def __init__(self, config: ReporterConfig):

        self.config = config
        self.metrics_paths = sorted(glob.glob(os.path.join(self.config.metrics_dir, '*.json')))

        # --------------- init by get_topK_array
        self.mask_layer = None  
        self.forget_digit = None 
        self.classifier_no_intervention_stats = {
            'foget_set_loss': None, 
            'foget_set_score': None, 
            'retain_set_loss': None,
            'retain_set_score': None, 
            'forget_digit_probability': None,
        }
        # ---------------------------------------

        self.topK = self.get_topK_array()

    def get_topK_array(self) -> List[float]:

        topK = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                if self.mask_layer is None:
                    # init
                    self.mask_layer = metrics['maked_layer']
                    self.forget_digit = metrics['forget_digit']
                    self.classifier_no_intervention_stats['forget_digit_probability'] = metrics['unlearning_metrics']['before_masking_probability']
                    self.classifier_no_intervention_stats['foget_set_loss'] = metrics['unlearning_metrics']['before_masking_loss']
                    self.classifier_no_intervention_stats['foget_set_score'] = metrics['unlearning_metrics']['before_masking_score']
                    self.classifier_no_intervention_stats['retain_set_loss'] = metrics['performance_degradation_metrics']['before_masking_loss']
                    self.classifier_no_intervention_stats['retain_set_score'] = metrics['unlearning_metrics']['before_masking_score']
                else:
                    # sanity check 
                    assert self.mask_layer == metrics['maked_layer']
                    assert self.forget_digit == metrics['forget_digit']
                    
                topK.append(metrics['top_k_value'])
        return topK
    
    def get_mimu_retain_set_loss(self) -> List[float]:
        """Measures degradation as loss"""
        mimu_retain_loss = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                l = metrics['performance_degradation_metrics']['after_masking_loss']
                mimu_retain_loss.append(l)
        return mimu_retain_loss
    
    def get_mimu_forget_set_loss(self) -> List[float]:
        """Measures unlearning as loss"""
        mimu_forget_loss = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                l = metrics['unlearning_metrics']['after_masking_loss']
                mimu_forget_loss.append(l)
        return mimu_forget_loss
    
    def get_mimu_forget_set_probability(self) -> List[float]:
        """Measures unlearning as average probability of classifier evaluated on forget digit."""
        mimu_forget_set_probability = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                l = metrics['unlearning_metrics']['after_mask_probability']
                mimu_forget_set_probability.append(l)
        return mimu_forget_set_probability
    
    def get_mimu_retain_set_score(self) -> List[float]:
        """Measures degradation as percent score"""
        mimu_retain_score = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                l = metrics['performance_degradation_metrics']['after_masking_score']
                mimu_retain_score.append(l)
        return mimu_retain_score
    
    def get_mimu_forget_set_score(self) -> List[float]:
        """Measures unlearning as percent score"""
        mimu_forget_score = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                l = metrics['unlearning_metrics']['after_masking_score']
                mimu_forget_score.append(l)
        return mimu_forget_score

    # random baseline
    def get_random_masking_retain_set_loss(self) -> List[float]:
        """Measures degradation as loss"""
        random_retain_loss = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                l = metrics['performance_degradation_metrics']['random_masking_loss']
                random_retain_loss.append(l)
        return random_retain_loss
    
    # random baseline
    def get_random_masking_forget_set_score(self) -> List[float]:
        """Measures unlearning as score"""
        random_retain_score = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                l = metrics['performance_degradation_metrics']['after_masking_score']
                random_retain_score.append(l)
        return random_retain_score
    
    # random baseline
    def get_random_baseline_forget_set_probability(self) -> List[float]:
        """Measures unlearning as average probability of classifier evaluated on forget digit."""
        mimu_forget_set_probability = []
        for p in self.metrics_paths:
            with open(p, 'r') as f:
                metrics = json.loads(f.read().strip())
                l = metrics['unlearning_metrics']['random_masking_probability']
                mimu_forget_set_probability.append(l)
        return mimu_forget_set_probability
    
    def draw_unlearning_comparison(self) -> None:
        pass