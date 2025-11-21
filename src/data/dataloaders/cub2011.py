import copy
import os
import re
from typing import Literal

import pandas as pd
import torch
import torchvision.transforms as transforms
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url

from .base import BaseRealDataset


# adapted from https://github.com/TDeVries/cub2011_dataset/blob/master/cub2011.py
class Cub2011(BaseRealDataset):
    base_folder = "CUB_200_2011/images"
    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    filename = "CUB_200_2011.tgz"
    tgz_md5 = "97eceeb196236b17998738112f37df78"

    def __init__(
        self,
        split: str = "train",
        res=256,
        crop_res: int = 256,
        crop_mode: Literal["center", "random"] = "center",
        data_root: str = "data/datasets",
    ):
        super().__init__()
        self.root = os.path.join(os.path.expanduser(data_root), "cub")

        self.loader = default_loader
        self.split = split

        self.num_classes = 200

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

        self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        self._get_class_names()

        self.full_data = copy.deepcopy(self.data)

    def _load_metadata(self):
        images = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "images.txt"),
            sep=" ",
            names=["img_id", "filepath"],
        )
        image_class_labels = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "image_class_labels.txt"),
            sep=" ",
            names=["img_id", "target"],
        )
        train_test_split = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "train_test_split.txt"),
            sep=" ",
            names=["img_id", "is_training_img"],
        )

        data = images.merge(image_class_labels, on="img_id")
        self.data = data.merge(train_test_split, on="img_id")

        if self.split == "train":
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def _get_class_names(self):
        directory = os.path.join(self.root, self.base_folder)
        folders = list(
            sorted(
                [
                    name
                    for name in os.listdir(directory)
                    if os.path.isdir(os.path.join(directory, name))
                ]
            )
        )
        self.class_names = [f.split(".")[1] for f in folders]
        self.class_names = [
            re.sub(r"_(?=[a-z])", "-", re.sub(r"_(?=[^a-z])", " ", s))
            for s in self.class_names
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def get_single_class(self, cls: int) -> Tensor:
        self.data = self.full_data[self.full_data["target"] == cls + 1]

        num_samples = len(self.data)

        loader = DataLoader(self, batch_size=64, num_workers=8)
        images = []
        labels = []
        print(f"Loading all {num_samples} images for class {cls}...")
        for x, y in loader:
            images.append(x)
            labels.append(y)
        images = torch.cat(images)
        labels = torch.cat(labels)
        print("Done.")

        self.data = self.full_data

        return images
