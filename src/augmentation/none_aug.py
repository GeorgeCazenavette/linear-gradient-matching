from torch import Tensor, nn


class AugNone(nn.Module):

    def __init__(self, crop_res: int):
        super().__init__(crop_res=crop_res)

    def __call__(self, image: Tensor) -> Tensor:
        return image
