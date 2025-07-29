import glob
from pathlib import Path

from model import (HookedMNISTClassifier, HookedResnet, SupportedVisionModels,
                   vision_model_loader)


def test_supported_models():
    assert isinstance(SupportedVisionModels.HookedResnet, SupportedVisionModels)
    assert isinstance(SupportedVisionModels.HookedResnet.value, type(HookedResnet))
    assert isinstance(SupportedVisionModels.HookedResnet.value(), HookedResnet)


def test_vision_model_loader():
    model = vision_model_loader(
        model_type=SupportedVisionModels.HookedMNISTClassifier, compile=False
    )
    assert isinstance(model, HookedMNISTClassifier)

    checkpoint_dir = Path("checkpoints/resnet")
    if checkpoint_dir.exists() and len(list(checkpoint_dir.iterdir())):
        resnet_files = sorted(glob.glob(str(checkpoint_dir / "resnet_*.pt")))
        model = vision_model_loader(
            model_type=SupportedVisionModels.HookedResnet,
            load_pretrained_from_path=resnet_files[0],
            compile=False,
        )
        assert hasattr(model, "_orig_mod")
        assert isinstance(model._orig_mod, HookedResnet)

    model: HookedResnet = vision_model_loader(
        model_type=SupportedVisionModels.HookedResnet,
        num_classes=1000,
        unlearning_target_layer_dim=256,
    )

    assert model.num_classes == 1000
    assert model.unlearning_target_layer_dim == 256
