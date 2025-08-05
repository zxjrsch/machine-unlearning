from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

try:
    workding_dir = Path.cwd()
    global_config = OmegaConf.load(workding_dir / "configs/config.yaml")
except Exception:
    global_config = {"device": "cuda"}


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    pass


if __name__ == "__main__":
    print(global_config.device)
