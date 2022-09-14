from omegaconf import DictConfig
import numpy as np
import torch

import random


__all__ = ["fix_randomness"]


def fix_randomness(cfg: DictConfig) -> None:
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed(cfg.random_seed)
