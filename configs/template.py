import hydra
from omegaconf import DictConfig, OmegaConf

global_config = OmegaConf.load("/home/claire/mimu/configs/config.yaml")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    pass


if __name__ == "__main__":
    print(global_config.device)
