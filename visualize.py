import glob
import json
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

class Reporter:
    def __init__(self, metrics_dir: Path = Path('eval/Metrics and Plots/metrics'), report_dir: Path = Path('reports')):

        self.metrics_paths = sorted(glob.glob(os.path.join(metrics_dir, '*.json')))

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
    
