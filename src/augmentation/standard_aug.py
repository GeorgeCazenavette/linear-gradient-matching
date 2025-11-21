import torch.nn as nn
from torch import Tensor

from .ops import RandomGaussianNoise, RandomHorizontalFlip, RandomResizedCrop


class AugStandard(nn.Module):

    def __init__(self, crop_res: int):

        super().__init__()

        self.aug = nn.Sequential(
            RandomHorizontalFlip(p=0.5),
            RandomResizedCrop(size=crop_res),
            RandomGaussianNoise(mean=0.5, std=0.2),
        )

    def __call__(self, image: Tensor) -> Tensor:
        return self.aug(image)
