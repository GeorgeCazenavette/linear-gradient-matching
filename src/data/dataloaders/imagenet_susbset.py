from copy import deepcopy
from typing import Literal

import torch
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional
from torch import Tensor
from torch.utils.data import DataLoader

from .base import BaseRealDataset


class ImageNetSubset(BaseRealDataset):

    subset_classes = {
        "imagenet-birds": [84, 130, 88, 144, 145, 22, 96, 9, 100, 89],
        "imagenet-fruits": [953, 954, 949, 950, 951, 957, 952, 945, 943, 948],
        "imagenet-cats": [281, 282, 283, 284, 285, 291, 292, 290, 289, 287],
        "imagenet-imagenette": [0, 217, 482, 491, 497, 566, 569, 571, 574, 701],
        "imagenet-imagewoof": [193, 182, 258, 162, 155, 167, 159, 273, 207, 229],
        "imagenet-a": [255, 376, 984, 364, 500, 986, 333, 576, 148, 135],
        "imagenet-b": [129, 916, 90, 275, 995, 874, 102, 259, 685, 139],
        "imagenet-c": [565, 94, 554, 535, 92, 392, 291, 136, 324, 11],
        "imagenet-d": [9, 258, 13, 262, 19, 339, 321, 24, 93, 322],
        "imagenet-e": [816, 96, 100, 145, 739, 713, 783, 76, 688, 326],
        "imagenet-100": [
            452,
            64,
            374,
            236,
            993,
            176,
            882,
            904,
            503,
            74,
            57,
            959,
            953,
            508,
            872,
            228,
            122,
            421,
            599,
            858,
            157,
            449,
            994,
            608,
            151,
            209,
            15,
            876,
            246,
            766,
            455,
            857,
            131,
            119,
            234,
            90,
            45,
            936,
            479,
            272,
            665,
            653,
            659,
            158,
            960,
            765,
            908,
            703,
            407,
            560,
            317,
            938,
            724,
            748,
            331,
            619,
            120,
            267,
            155,
            708,
            368,
            772,
            167,
            494,
            180,
            431,
            342,
            854,
            305,
            54,
            268,
            667,
            166,
            277,
            662,
            798,
            313,
            498,
            299,
            222,
            682,
            593,
            775,
            674,
            592,
            137,
            758,
            717,
            606,
            281,
            796,
            211,
            620,
            830,
            544,
            275,
            242,
            570,
            99,
            169,
        ],
        "imagenet-1k": list(range(1000)),
    }

    def __init__(
        self,
        split: str = "train",
        res=256,
        subset_name: str = None,
        crop_res: int = 256,
        crop_mode: Literal["center", "random"] = "center",
        data_root: str = "data/datasets",
    ):

        super().__init__()
        self.res = res

        if subset_name not in self.subset_classes.keys():
            raise NotImplementedError(
                f'ImageNet subset "{subset_name}" is not implemented. You can register a new subset in data/dataloaders/imagenet_subset.py by adding it to the subset_classes dictionary.'
            )
        self.classes = self.subset_classes[subset_name]
        self.label_dict = {self.classes[i]: i for i in range(len(self.classes))}
        self.inv_label_dict = {v: k for k, v in self.label_dict.items()}

        self.num_classes = len(self.classes)

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

        self.full_ds = torchvision.datasets.ImageNet(
            root=f"{data_root}/imagenet", split=split, transform=self.transform
        )

        mask = torch.isin(
            torch.tensor(self.full_ds.targets), torch.tensor(self.classes)
        )

        ds = torch.utils.data.Subset(self.full_ds, torch.argwhere(mask))
        self.ds = ds

        self.class_names = [self.full_ds.classes[c][0] for c in self.classes]

    def __getitem__(self, index):
        image, label = self.ds.__getitem__(index)
        label = self.convert_label(label)
        return image, label

    def __len__(self):
        return len(self.ds)

    def convert_label(self, label):
        return self.label_dict[label]

    def get_single_class(self, cls: int) -> Tensor:

        copy_ds = deepcopy(self.full_ds)
        copy_ds.samples = [
            s for s in copy_ds.samples if s[1] == self.inv_label_dict[cls]
        ]
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
