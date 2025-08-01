from dataclasses import dataclass
from loguru import logger
from model import SupportedVisionModels, SupportedDatasets
from trainer import *
from time import perf_counter
from torch import nn
from data import GraphGenerator
from eval import Eval, EvalConfig
from typing import Dict
from itertools import product

@dataclass
class PipelineConfig:
    model_architecture: SupportedVisionModels
    vision_dataset: SupportedDatasets
    vision_model_epochs: int
    vision_model_max_steps_per_epoch: int
    vision_model_logging_steps: int
    vision_model_batch_size: int
    vision_model_learning_rate: float # 1e-3
    device: str
    forget_class: int # 0
    graph_dataset_size: int # 1024
    graph_batch_size: int # 64
    gcn_train_steps: int # 130
    gcn_learning_rate: float # 1e-2
    gcn_logging_steps: int # 10
    sft_steps: int # 50
    eval_batch_size: int # 256
    eval_draw_plots: bool # True
    eval_draw_category_probabilities: bool # True
    topK_list: List[int] # [8000]
    kappa_list: List[int] # [7000]

    # these following optionals can be genereated by the pipeline 
    # when it is run in full but can also be passed in 
    trained_vision_model_path: Optional[Path] = None
    graph_dir: Optional[Path] = None
    gcn_path: Optional[Path] = None

class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.trained_vision_model: Optional[nn.Module] = None
        self.trained_vision_model_path: Optional[Path] = config.trained_vision_model_path
        self.graph_dir: Optional[Path] = config.graph_dir
        self.gcn: Optional[nn.Module] = None
        self.gcn_path: Optional[Path] = config.gcn_path


    def run_vision_model_training(self) -> Tuple[Path, nn.Module]:
        """Returns tuple of model weight path and the model"""
        logger.info(f'========== Training {self.cfg.model_architecture.value} on {self.cfg.vision_dataset.value} ==========')

        config = VisionModelTrainerConfig(
            architecture=self.cfg.model_architecture,
            vision_dataset=self.cfg.vision_dataset,
            logging_steps = self.cfg.vision_model_logging_steps,
            plot_statistics=True,
            batch_size=self.cfg.vision_model_batch_size,
            epochs=self.cfg.vision_model_epochs,
            steps=self.cfg.vision_model_max_steps_per_epoch,
            lr=self.cfg.vision_model_learning_rate,
            device=self.cfg.device
        )

        trainer = VisionModelTrainer(config)
        a = perf_counter()
        path = trainer.train()
        b = perf_counter()

        logger.info(f'Compled training of {self.cfg.model_architecture.value} on {self.cfg.vision_dataset.value} in {round((b-a) / 60)} min, checkpoint saved to {path}.')
        
        self.trained_vision_model = trainer.model
        self.trained_vision_model_path = path
        return path, trainer.model
    
    def run_gcn_graph_generation(self) -> Path:

        assert self.trained_vision_model_path is not None, "Model not found. Train vision model before generating data, or pass in path to model weights in config."

        logger.info(f'========== Generating GCN graphs from {self.cfg.model_architecture.value} using {self.cfg.vision_dataset.value} ==========')

        generator = GraphGenerator(
            vision_model_type=self.cfg.model_architecture,
            unlearning_dataset=self.cfg.vision_dataset,
            checkpoint_path=self.trained_vision_model_path,
            graph_dataset_dir= Path("./datasets/Graphs"),
            graph_data_cardinaility=self.cfg.graph_dataset_size,
            process_save_batch_size=self.cfg.graph_batch_size,
            forget_class=self.cfg.forget_class,
            device=self.cfg.device,
            mask_layer = -2,  # only 2 is relevant
            save_redundant_features = True, # artifacts for graph generation
        )

        a = perf_counter()
        self.graph_dir = generator.genereate_graphs()
        b = perf_counter()
        logger.info(f'Compled graph generation for {self.cfg.model_architecture.value} on {self.cfg.vision_dataset.value} in {round((b-a) / 60)} min, saved to {self.graph_dir}.')
        return self.graph_dir
    
    def run_gcn_training(self, topK: int) -> Tuple[Path, nn.Module]:
        """Returns tuple of model weight path and the model"""
        assert self.trained_vision_model_path is not None, "Model not found. Train vision model before generating data, or pass in path to model weights in config."
        assert self.graph_dir is not None, "Graph dir not found. Run GCN graph generation before training GCN, or pass in path to graph dir in the config."

        logger.info(f'========== Training GCN for {self.cfg.model_architecture.value} on {self.cfg.vision_dataset.value} ==========')

        config = GCNTrainerConfig(    
            vision_model_architecture=self.cfg.model_architecture,
            vision_model_path=self.trained_vision_model_path,
            vision_dataset=self.cfg.vision_dataset,
            gcn_dataset_dir= Path("datasets/Graphs"),
            device = self.cfg.device,
            mask_layer = -2,
            steps = self.cfg.gcn_train_steps,
            lr = self.cfg.gcn_learning_rate,
            weight_decay = 5e-4,
            mask_K = topK,
            logging_steps = self.cfg.gcn_logging_steps,
            gcn_checkpoint_path = Path("checkpoints/gcn")
        )
        a = perf_counter()
        trainer = GCNTrainer(config=config)
        self.gcn_path = trainer.train()
        self.gcn = trainer.gcn
        b = perf_counter()
        logger.info(f'Compled GCN training for {self.cfg.model_architecture.value} on {self.cfg.vision_dataset.value} in {round((b-a) / 60)} min, saved to {self.gcn_path}.')
        return self.gcn_path, self.gcn

    def run_single_evaluation_round(self, topK: int, kappa: int) -> Dict:
        """Metrics in raw json and plots are saved, metrics dictionary returned by this method."""
        assert self.trained_vision_model_path is not None

        self.run_gcn_training(topK=topK)

        logger.info(f'========== Evaluating top-{topK} kappa-{kappa} for {self.cfg.model_architecture.value} on {self.cfg.vision_dataset.value} ==========')

        config = EvalConfig(
            vision_model=self.cfg.model_architecture,
            vision_model_path=self.trained_vision_model_path,
            vision_dataset=self.cfg.vision_dataset,
            gcn_path=self.gcn_path,
            forget_class=self.cfg.forget_class,
            mask_layer = -2,
            topK=topK,
            kappa=kappa,
            use_set_difference_masking_strategy=False,
            sft_steps=self.cfg.sft_steps,
            sft_mode=SFTModes.Randomize_Forget,
            gcn_base_path = Path('checkpoints/gcn/'),
            graph_data_base_path= Path("eval/Graphs"),
            metrics_base_path=Path("eval/Metrics and Plots"),
            batch_size=self.cfg.eval_batch_size,
            device=self.cfg.device,
            draw_eval_plots=self.cfg.eval_draw_plots,
            plot_category_probabilities=self.cfg.eval_draw_category_probabilities
        )
        a = perf_counter()
        eval = Eval(config)
        b = perf_counter()
        logger.info(f'Compled evaluation top-{topK} kappa-{kappa} for {self.cfg.model_architecture.value} on {self.cfg.vision_dataset.value} in {round((b-a) / 60)} min.')

        return eval.eval()
    
    def eval(self) -> List[Dict]:
        topK_kappa_pairs = list(product(self.cfg.topK_list, self.cfg.kappa_list))
        metric_dict_array = []
        for topK, kappa in topK_kappa_pairs:
            d = self.run_single_evaluation_round(topK=topK, kappa=kappa)
            metric_dict_array.append(d)
        return metric_dict_array
            
    def run(self) -> List[Dict]:
        self.run_vision_model_training()
        self.run_gcn_graph_generation()
        # NOTE gcn training is run in the method run_single_evaluation_round
        metric_dict_array = self.eval()
        return metric_dict_array