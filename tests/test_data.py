from data import GraphGenerator
from model import SupportedDatasets, SupportedVisionModels


def test_graph_generator():
    path = "checkpoints/HookedResnet_MNIST_30_13_41.pt"
    generator = GraphGenerator(
        vision_model_type=SupportedVisionModels.HookedResnet,
        unlearning_dataset=SupportedDatasets.MNIST,
        checkpoint_path=path,
    )
    generator.save_forward_backward_features()
