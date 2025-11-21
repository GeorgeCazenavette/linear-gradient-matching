import copy
import ssl
from typing import Literal

import torch
import torchvision.transforms as transforms
import tqdm
import wilds
from torch import Tensor
from torch.utils.data import DataLoader

from .base import BaseRealDataset

ssl._create_default_https_context = ssl._create_unverified_context


class Waterbirds(BaseRealDataset):

    def __init__(
        self,
        split: str = "train",
        res=256,
        crop_res: int = 256,
        crop_mode: Literal["center", "random"] = "center",
        data_root: str = "data/datasets",
    ):

        super().__init__()

        self.class_names = [
            "Land Bird",
            "Water Bird",
        ]

        self.num_classes = 2

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose(
            [
                transforms.Resize(res),
                (
                    transforms.CenterCrop(crop_res)
                    if crop_mode == "center"
                    else transforms.RandomCrop(crop_res)
                ),
                transforms.ToTensor(),
            ]
        )

        self.mean = torch.tensor(mean, device="cuda").reshape(1, 3, 1, 1)
        self.std = torch.tensor(std, device="cuda").reshape(1, 3, 1, 1)

        self.full_ds = wilds.get_dataset(
            dataset="waterbirds", root_dir=f"{data_root}/waterbirds", download=True
        ).get_subset(split=split, transform=self.transform)

        self.ds = copy.deepcopy(self.full_ds)
        self.targets = self.get_targets()
        self.full_ds.targets = self.targets

    def __getitem__(self, index):

        image, label, background = self.ds.__getitem__(index)
        return image, label

    def __len__(self):
        return len(self.full_ds)

    def get_targets(self) -> Tensor:
        targets = []
        loader = DataLoader(self.ds, num_workers=16, batch_size=16)
        for image, label, background in tqdm.tqdm(loader, desc="Getting targets..."):
            targets.append(label)

        targets = torch.cat(targets)
        return targets

    def get_single_class(self, cls: int) -> Tensor:
        mask = torch.isin(torch.tensor(self.full_ds.targets), torch.tensor([cls]))
        subset = torch.utils.data.Subset(self.full_ds, torch.argwhere(mask))
        num_samples = len(subset)
        loader = DataLoader(subset, batch_size=num_samples, num_workers=8)
        images = []
        labels = []
        print(f"Loading all {num_samples} images for class {cls}...")
        for x, y, _ in loader:
            images.append(x)
            labels.append(y)
        images = torch.cat(images)
        labels = torch.cat(labels)
        print("Done.")

        return images
