import copy
from typing import Literal

import spawrious.torch as sp
import torch
import torchvision.transforms as transforms
import tqdm
from torch import Tensor
from torch.utils.data import DataLoader

from .base import BaseRealDataset


class Spawrious(BaseRealDataset):

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
            "Bulldog",
            "Corgi",
            "Dachshund",
            "Labrador",
        ]

        def _prepare_data_lists(
            self, train_combinations, test_combinations, root_dir, augment
        ):
            test_transforms = transforms.Compose(
                [
                    transforms.Resize(res),
                    transforms.CenterCrop(crop_res),
                    transforms.transforms.ToTensor(),
                ]
            )

            train_transforms = test_transforms

            train_data_list = self._create_data_list(
                train_combinations, root_dir, train_transforms
            )

            test_data_list = self._create_data_list(
                test_combinations, root_dir, test_transforms
            )

            return train_data_list, test_data_list

        sp.SpawriousBenchmark._prepare_data_lists = _prepare_data_lists
        sp.SpawriousBenchmark.input_shape = (3, res, res)

        spawrious_benchmark = sp.SpawriousBenchmark(
            "o2o_hard", f"{data_root}/spawrious", augment=False
        )

        self.num_classes = 4

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.mean = torch.tensor(mean, device="cuda").reshape(1, 3, 1, 1)
        self.std = torch.tensor(std, device="cuda").reshape(1, 3, 1, 1)

        if split == "train":
            self.full_ds = spawrious_benchmark.get_train_dataset()
        else:
            self.full_ds = spawrious_benchmark.get_test_dataset()

        self.ds = copy.deepcopy(self.full_ds)

        self.targets = None

        # self.targets = self.get_targets()
        # self.full_ds.targets = self.targets

    def __getitem__(self, index):

        image, label, background = self.ds.__getitem__(index)
        return image, label

    def __len__(self):
        return len(self.ds)

    def get_targets(self) -> Tensor:
        targets = []
        loader = DataLoader(self.ds, num_workers=16, batch_size=16)
        for image, label, background in tqdm.tqdm(loader, desc="Getting targets..."):
            targets.append(label)

        targets = torch.cat(targets)
        return targets

    def get_single_class(self, cls: int) -> Tensor:
        if self.targets is None:
            self.targets = self.get_targets()
            self.full_ds.targets = self.targets

        mask = torch.isin(torch.tensor(self.full_ds.targets), torch.tensor([cls]))
        subset = torch.utils.data.Subset(self.full_ds, torch.argwhere(mask))

        num_samples = len(subset)
        loader = DataLoader(subset, batch_size=num_samples, num_workers=8)
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
