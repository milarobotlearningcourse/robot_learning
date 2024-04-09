from hw2.hw2 import my_app
import hydra, json
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="config_hw2")
def my_main(cfg: DictConfig):
    results = my_app(cfg)
    # print ("Results:", results)
    return results


if __name__ == "__main__":
    import os
    results = my_main()