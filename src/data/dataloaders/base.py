import abc
from typing import List

import torch
import torch.utils.data
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class BaseRealDataset(Dataset):

    class_names: List[str]
    mean: torch.Tensor
    std: torch.Tensor
    num_classes: int = 0
    res: int

    def __init__(self):
        pass

    def __len__(self) -> int:
        raise NotImplementedError("__len__ is not implemented for this dataset.")

    def normalize(self, x) -> torch.Tensor:

        return (x - self.mean) / self.std

    def denormalize(self, x) -> torch.Tensor:

        return (x * self.std) + self.mean

    def get_random_reals(self, ipc=1):

        images = [[] for c in range(self.num_classes)]
        loader_iter = iter(DataLoader(self, num_workers=8, batch_size=1, shuffle=True))

        while any([len(images[c]) < ipc for c in range(self.num_classes)]):
            x, y = next(loader_iter)
            if len(images[y]) < ipc:
                images[y].append(x)

        images = [torch.cat(l) for l in images]
        images = torch.cat(images).cuda()

        return images

    @abc.abstractmethod
    def get_single_class(self, cls: int) -> Tensor:
        raise NotImplementedError(
            "get_single_class is not implemented for this dataset."
        )
