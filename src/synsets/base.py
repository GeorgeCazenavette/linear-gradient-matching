from typing import Tuple

import torch
from einops import rearrange
from torch import Tensor


class BaseDistilledDataset:
    optimizer: torch.optim.Optimizer
    syn_lr: Tensor
    res: int
    num_samples: int

    def __init__(self):
        self.color_correlation_svd_sqrt = torch.tensor(
            [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]
        ).cuda()

        self.max_norm_svd_sqrt = torch.max(
            torch.linalg.norm(self.color_correlation_svd_sqrt, axis=0)
        )

        self.color_mean = torch.tensor([0.48, 0.46, 0.41]).cuda()

    def get_data(self) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def log_images(self, step: int = None):
        raise NotImplementedError

    def upkeep(self, step: int = None):
        return

    def get_save_dict(self):
        return

    def load_from_dict(self, load_dict: dict):
        return

    def linear_decorrelate_color(self, im: Tensor):
        b, c, h, w = im.shape
        im = rearrange(im, "b c h w -> (b h w) c")
        color_correlation_normalized = (
            self.color_correlation_svd_sqrt / self.max_norm_svd_sqrt
        )
        im = im @ color_correlation_normalized.T
        im = rearrange(im, "(b h w) c -> b c h w", h=h, w=w)
        return im
