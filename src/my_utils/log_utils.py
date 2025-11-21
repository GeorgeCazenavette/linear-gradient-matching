import torchvision
from torch import Tensor

import wandb


def log_images(syn_images: Tensor, step: int):

    grid = torchvision.utils.make_grid(
        syn_images, normalize=False, scale_each=False, nrow=5
    )
    wandb.log({"grids/raw": wandb.Image(grid)}, step=step)
