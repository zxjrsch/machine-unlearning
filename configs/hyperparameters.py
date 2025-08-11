import os
from enum import Enum
from pathlib import Path

from trainer import GCNPriorDistribution, SFTModes


class DEFAULT_HP(Enum):
    vision_model_epochs = 8
    vision_model_max_steps_per_epoch = 1024 * 16
    vision_model_logging_steps = 1024
    vision_model_batch_size = 512  # 256
    vision_model_learning_rate = 1e-3
    vision_model_checkpoint_dir = Path(os.path.expanduser("~/mimu/vision_checkpoints"))
    plot_vision_model_train_statistics = False
    num_workers = 2  # num gpus
    device = "cuda"
    forget_class = 0
    graph_dataset_size = 2048
    graph_dataset_dir = Path(os.path.expanduser("~/mimu/graphs"))
    gcn_checkpoint_dir = Path(os.path.expanduser("~/mimu/gcn_checkpoints"))
    graph_batch_size = 64
    use_sinkhorn_sampler = True
    use_set_difference_masking_strategy = False
    gcn_prior_distribution = GCNPriorDistribution.WEIGHT
    gcn_train_steps = 1  # adjust to something larger like 130
    gcn_learning_rate = 1e-2
    gcn_weight_decay = 5e-4
    gcn_logging_steps = 1
    sft_mode = SFTModes.Randomize_Forget
    sft_steps = 2  # adjust to something larger like 50
    eval_batch_size = 256
    eval_draw_plots = True
    eval_draw_category_probabilities = True
    eval_metrics_base_path = Path(os.path.expanduser("~/mimu/metrics_and_plots"))
    topK_list = [8000]
    kappa_list = [6000, 7000]
    working_dir = Path(os.path.expanduser("~/mimu/"))
