from typing import List, Tuple

import torch
from torch import Tensor

from config import DistillCfg
from data.dataloaders import BaseRealDataset
from my_utils.log_utils import log_images

from .base import BaseDistilledDataset


class PyramidDataset(BaseDistilledDataset):

    def __init__(self, train_dataset: BaseRealDataset, cfg: DistillCfg):

        super().__init__()
        self.train_dataset = train_dataset
        self.cfg = cfg
        self.pyramid, self.syn_labels = self.init_synset()
        self.optimizer = self.init_optimizer()

    def init_optimizer(self):
        optimizer = torch.optim.Adam(
            [{"params": p, "lr": self.cfg.lr} for p in self.pyramid]
        )
        return optimizer

    def init_synset(self) -> Tuple[List[Tensor], Tensor]:

        syn_labels = torch.cat(
            [
                torch.tensor([c] * self.cfg.ipc, dtype=torch.long)
                for c in range(self.train_dataset.num_classes)
            ],
            dim=0,
        ).cuda()

        num_images = self.cfg.ipc * self.train_dataset.num_classes

        pyramid = []
        res = 1
        while res <= self.cfg.pyramid_start_res:
            level = torch.randn((num_images, 3, res, res), device="cuda")
            if self.cfg.init_mode == "zero":
                level = level * 0
            pyramid.insert(0, level)
            res *= 2

            # to make it work when res is not power of 2
            if res > self.cfg.syn_res:
                res = self.cfg.syn_res

        pyramid = [p / len(pyramid) for p in pyramid]

        for p in pyramid:
            p.requires_grad_(True)

        return pyramid, syn_labels

    def extend_pyramid(self) -> bool:

        print("extending pyramid...")

        old_len = len(self.pyramid)
        new_len = len(self.pyramid) + 1

        old_res = self.pyramid[0].shape[-1]

        if old_res == self.cfg.syn_res:
            print("already max res")
            return False
        else:
            new_res = old_res * 2
            # to make it work when res is not power of 2
            if new_res > self.cfg.syn_res:
                new_res = self.cfg.syn_res
            print("new res: {}".format(new_res))

        num_images = self.pyramid[-1].shape[0]

        self.pyramid = [p.detach().clone() * old_len / new_len for p in self.pyramid]
        if self.cfg.init_mode == "zero":
            new_layer = torch.sum(
                torch.stack(
                    [
                        torch.nn.functional.interpolate(
                            p, (new_res, new_res), antialias=False, mode="bilinear"
                        )
                        for p in self.pyramid
                    ]
                ),
                dim=0,
            )
            new_layer = new_layer / old_len
        else:
            new_layer = (
                torch.randn((num_images, 3, new_res, new_res), device="cuda") / new_len
            )

        self.pyramid.insert(0, new_layer)

        for p in self.pyramid:
            p.requires_grad_(True)

        self.optimizer = self.init_optimizer()

        return True

    def decode_pyramid(self) -> Tensor:

        result = torch.sum(
            torch.stack(
                [
                    torch.nn.functional.interpolate(
                        p,
                        (self.cfg.syn_res, self.cfg.syn_res),
                        antialias=False,
                        mode="bilinear",
                    )
                    for p in self.pyramid
                ]
            ),
            dim=0,
        )

        if self.cfg.decorrelate_color:
            result = self.linear_decorrelate_color(result)

        result = torch.sigmoid(2 * result)

        return result

    def get_data(self) -> Tuple[Tensor, Tensor]:
        syn_images = self.decode_pyramid()
        return syn_images, self.syn_labels

    @torch.no_grad()
    def log_images(self, step: int = None):
        if len(self.pyramid[0]) > 100:
            print("Warning: too many images to log")
            return
        with torch.no_grad():

            syn_images, _ = self.get_data()
            syn_images = syn_images.detach().clone()

            log_images(syn_images=syn_images, step=step)

    def upkeep(self, step: int = None):
        if (step - 1) % self.cfg.pyramid_extent_it == 0 and step > 1:
            if self.extend_pyramid():
                self.log_images(step=step)

    def get_save_dict(self):
        save_dict = {
            "pyramid": self.pyramid,
            "opt_state": self.optimizer.state_dict(),
        }
        return save_dict

    def load_from_dict(self, load_dict: dict):

        loaded_pyramid = load_dict["pyramid"]
        while len(self.pyramid) < len(loaded_pyramid):
            self.extend_pyramid()

        with torch.no_grad():
            for p, loaded_p in zip(self.pyramid, loaded_pyramid):
                p.copy_(loaded_p)

        self.optimizer.load_state_dict(load_dict["opt_state"])
