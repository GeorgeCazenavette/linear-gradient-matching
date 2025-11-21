import copy
import os
import shutil
import tarfile
import urllib.request
from copy import deepcopy
from typing import Literal

import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch import Tensor
from torch.utils.data import DataLoader

from .base import BaseRealDataset


class ArtBench(BaseRealDataset):

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
            "Art Nouveau",
            "Baroque",
            "Expressionism",
            "Impressionism",
            "Post-Impressionism",
            "Realism",
            "Renaissance",
            "Romanticism",
            "Surrealism",
            "Ukiyo-e",
        ]

        self.verify_files()

        self.num_classes = 10

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose(
            [
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

        self.full_ds = torchvision.datasets.ImageFolder(
            root="{}/artbench/{}".format(data_root, split), transform=self.transform
        )
        self.ds = copy.deepcopy(self.full_ds)
        self.targets = self.ds.targets

    def __getitem__(self, index):

        image, label = self.ds.__getitem__(index)
        return image, label

    def __len__(self):
        return len(self.ds)

    def verify_files(self):

        if not os.path.exists("artbench/train"):
            os.makedirs("artbench", exist_ok=True)

            print("Downloading ArtBench (this may take some time)...")
            urllib.request.urlretrieve(
                "https://artbench.eecs.berkeley.edu/files/artbench-10-imagefolder-split.tar",
                "artbench/artbench-10-imagefolder-split.tar",
            )

            print("Extracting ArtBench (this may take some time)...")
            with tarfile.open("artbench/artbench-10-imagefolder-split.tar") as tar:
                tar.extractall(path="artbench/")

            os.rename("artbench/artbench-10-imagefolder-split/train", "artbench/train")
            os.rename("artbench/artbench-10-imagefolder-split/test", "artbench/test")
            shutil.rmtree("artbench/artbench-10-imagefolder-split")
            os.remove("artbench/artbench-10-imagefolder-split.tar")

            print("ArtBench download complete!")

    def get_single_class(self, cls: int) -> Tensor:

        copy_ds = deepcopy(self.full_ds)
        copy_ds.samples = [s for s in copy_ds.samples if s[1] == cls]
        copy_ds.targets = [s[1] for s in copy_ds.samples]

        num_samples = len(copy_ds.samples)
        loader = DataLoader(copy_ds, batch_size=64, num_workers=8)
        images = []
        labels = []
        print(f"Loading all {num_samples} images for class {cls}...")
        for x, y in loader:
            images.append(x)
            labels.append(y)
        images = torch.cat(images)
        labels = torch.cat(labels)
        print("Done.")

        return images
