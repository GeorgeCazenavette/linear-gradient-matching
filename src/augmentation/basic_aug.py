from torch import Tensor, nn

from .ops import RandomHorizontalFlip, RandomResizedCrop


class AugBasic(nn.Module):

    def __init__(self, crop_res: int):

        super().__init__()

        self.aug = nn.Sequential(
            RandomHorizontalFlip(p=0.5),
            RandomResizedCrop(size=crop_res),
        )

    def __call__(self, image: Tensor) -> Tensor:
        return self.aug(image)
