from typing import Tuple

import torch
from torch import Tensor

from config import DistillCfg
from data.dataloaders import BaseRealDataset
from my_utils.log_utils import log_images

from .base import BaseDistilledDataset


class PixelDataset(BaseDistilledDataset):

    def __init__(self, train_dataset: BaseRealDataset, cfg: DistillCfg):

        super().__init__()
        self.train_dataset = train_dataset
        self.cfg = cfg

        # self.syn_lr = torch.tensor(self.args.lr_init).cuda().requires_grad_(True)
        self.syn_images, self.syn_labels = self.init_synset()

        self.optimizer = self.init_optimizer()

    def init_optimizer(self):
        if self.cfg.distill_opt == "sgd":
            optimizer = torch.optim.SGD(
                [
                    # {'params': self.syn_lr, 'lr': self.args.lr_lr},
                    {"params": self.syn_images, "lr": self.cfg.lr},
                ],
                momentum=0.5,
            )
        elif self.cfg.distill_opt == "adam":
            optimizer = torch.optim.Adam(
                [
                    # {'params': self.syn_lr, 'lr': self.args.lr_lr},
                    {"params": self.syn_images, "lr": self.cfg.lr},
                ]
            )
        else:
            raise NotImplementedError
        return optimizer

    def init_synset(self) -> Tuple[Tensor, Tensor]:

        syn_labels = torch.cat(
            [
                torch.tensor([c] * self.cfg.ipc, dtype=torch.long)
                for c in range(self.train_dataset.num_classes)
            ],
            dim=0,
        ).cuda()

        if self.cfg.init_mode == "real":
            syn_images = self.train_dataset.get_random_reals(self.cfg.ipc).cuda()

        else:
            syn_images = torch.randn(
                (
                    self.cfg.ipc * self.train_dataset.num_classes,
                    3,
                    self.cfg.syn_res,
                    self.cfg.syn_res,
                )
            ).cuda()

        syn_images.requires_grad_(True)

        return syn_images, syn_labels

    def get_data(self) -> Tuple[Tensor, Tensor]:
        result = self.syn_images
        if self.cfg.decorrelate_color:
            result = self.linear_decorrelate_color(result)
        result = torch.sigmoid(2 * result)
        return result, self.syn_labels

    def log_images(self, step: int = None):

        with torch.no_grad():

            syn_images, _ = self.get_data()
            syn_images = syn_images.detach().clone()

            log_images(syn_images=syn_images, step=step)

    def get_save_dict(self):
        save_dict = {
            "images": self.syn_images,
            "opt_state": self.optimizer.state_dict(),
        }
        return save_dict

    def load_from_dict(self, load_dict: dict):

        with torch.no_grad():
            self.syn_images.copy_(load_dict["images"])

        self.optimizer.load_state_dict(load_dict["opt_state"])
