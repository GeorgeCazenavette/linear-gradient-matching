from typing import Literal, Tuple

from .artbench import ArtBench
from .base import BaseRealDataset
from .cub2011 import Cub2011
from .flowers102 import Flowers102
from .food101 import Food101
from .imagenet_susbset import ImageNetSubset
from .spawrious import Spawrious
from .stanford_dogs import StanfordDogs
from .waterbirds import Waterbirds


def get_dataset(
    name: str,
    res: int,
    crop_res: int,
    train_crop_mode: Literal["center", "random"],
    data_root: str,
) -> Tuple[BaseRealDataset, BaseRealDataset]:

    match name.lower():

        case "food101":
            train_dataset = Food101(
                split="train",
                res=res,
                crop_res=res,
                crop_mode=train_crop_mode,
                data_root=data_root,
            )
            test_dataset = Food101(
                split="test",
                res=res,
                crop_res=crop_res,
                crop_mode="center",
                data_root=data_root,
            )

        case "artbench":
            train_dataset = ArtBench(
                split="train",
                res=res,
                crop_res=res,
                crop_mode=train_crop_mode,
                data_root=data_root,
            )
            test_dataset = ArtBench(
                split="test",
                res=res,
                crop_res=crop_res,
                crop_mode="center",
                data_root=data_root,
            )

        case "cub2011":
            train_dataset = Cub2011(
                split="train",
                res=res,
                crop_res=res,
                crop_mode=train_crop_mode,
                data_root=data_root,
            )
            test_dataset = Cub2011(
                split="test",
                res=res,
                crop_res=crop_res,
                crop_mode="center",
                data_root=data_root,
            )

        case "stanforddogs":
            train_dataset = StanfordDogs(
                split="train",
                res=res,
                crop_res=res,
                crop_mode=train_crop_mode,
                data_root=data_root,
            )
            test_dataset = StanfordDogs(
                split="test",
                res=res,
                crop_res=crop_res,
                crop_mode="center",
                data_root=data_root,
            )

        case "spawrious":
            train_dataset = Spawrious(
                split="train",
                res=res,
                crop_res=res,
                crop_mode=train_crop_mode,
                data_root=data_root,
            )
            test_dataset = Spawrious(
                split="test",
                res=res,
                crop_res=crop_res,
                crop_mode="center",
                data_root=data_root,
            )

        case "waterbirds":
            train_dataset = Waterbirds(
                split="train",
                res=res,
                crop_res=res,
                crop_mode=train_crop_mode,
                data_root=data_root,
            )
            test_dataset = Waterbirds(
                split="test",
                res=res,
                crop_res=crop_res,
                crop_mode="center",
                data_root=data_root,
            )

        case "flowers102":
            train_dataset = Flowers102(
                split="train",
                res=res,
                crop_res=res,
                crop_mode=train_crop_mode,
                data_root=data_root,
            )
            test_dataset = Flowers102(
                split="test",
                res=res,
                crop_res=crop_res,
                crop_mode="center",
                data_root=data_root,
            )

        case "imagenet-1k":
            train_dataset = ImageNetSubset(
                split="train",
                res=res,
                subset_name=name,
                crop_res=res,
                crop_mode=train_crop_mode,
                data_root=data_root,
            )
            test_dataset = ImageNetSubset(
                split="val",
                res=res,
                subset_name=name,
                crop_res=crop_res,
                crop_mode="center",
                data_root=data_root,
            )

        case name if name.startswith("imagenet") and name not in [
            "imagenet-1k",
        ]:
            train_dataset = ImageNetSubset(
                split="train",
                res=res,
                subset_name=name,
                crop_res=res,
                crop_mode=train_crop_mode,
                data_root=data_root,
            )
            test_dataset = ImageNetSubset(
                split="val",
                res=res,
                subset_name=name,
                crop_res=crop_res,
                crop_mode="center",
                data_root=data_root,
            )

        case _:
            raise NotImplementedError("Dataset {} not implemented".format(name))

    return train_dataset, test_dataset
