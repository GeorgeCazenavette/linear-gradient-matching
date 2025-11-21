import os
import random
import signal
import sys
import types
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from augmentation import get_augmentor
from config import DistillCfg
from data.dataloaders import get_dataset
from models import get_fc, get_model
from synsets import get_distilled_dataset


class LinearGM:

    def __init__(self, cfg: DistillCfg):

        self.cfg = cfg
        self.global_step = 0
        self.log_dir = os.path.join(
            "logged_files",
            self.cfg.job_tag,
            self.cfg.dataset,
            self.cfg.model,
            self.cfg.run_name,
        )

        self.train_dataset, self.test_dataset = get_dataset(
            name=self.cfg.dataset,
            res=self.cfg.real_res,
            crop_res=cfg.crop_res,
            train_crop_mode=self.cfg.train_crop_mode,
            data_root=self.cfg.data_root,
        )

        self.real_batch_size = (
            cfg.ipc * cfg.augs_per_batch * self.train_dataset.num_classes
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=cfg.num_workers,
            batch_size=self.real_batch_size,
            drop_last=False,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=cfg.num_workers,
            batch_size=self.real_batch_size,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
        self.train_iter = iter(self.train_loader)

        self.distilled_dataset = get_distilled_dataset(
            train_dataset=self.train_dataset, cfg=self.cfg
        )

        self.syn_augmentor = get_augmentor(
            aug_mode=self.cfg.aug_mode, crop_res=self.cfg.crop_res
        )
        self.real_augmentor = get_augmentor(
            aug_mode=self.cfg.aug_mode, crop_res=self.cfg.crop_res
        )

        self.distributed = self.cfg.device_count > 1

        self.backbone_model, self.num_feats = get_model(
            name=cfg.model, distributed=self.distributed
        )

        self.load_checkpoint()

        signal.signal(signal.SIGUSR1, self.handle_interrupt)

        os.setpgrp()

    def distill(self):

        for i in tqdm(
            range(self.global_step, self.cfg.iterations + 1),
            initial=self.global_step,
            total=self.cfg.iterations + 1,
            desc="Distilling Images",
        ):

            self.global_step = i

            # handling any synthetic data specific tasks such as adding another pyramid layer
            # can be used for other things if you add other representations
            self.distilled_dataset.upkeep(step=self.global_step)

            if self.global_step % self.cfg.image_log_it == 0:
                self.distilled_dataset.log_images(step=self.global_step)

            # perform linear gradient matching
            loss = self.match_gradients()

            if self.global_step % 10 == 0:
                wandb.log(
                    {
                        "loss": loss,
                    },
                    step=self.global_step,
                )

            if self.global_step % self.cfg.checkpoint_it == 0:
                self.save_checkpoint()

            if self.global_step == self.cfg.iterations:
                self.save_data()

    def match_gradients(self):
        AMP_SCALE = 1024.0

        # initialize a random linear classifier
        fc = get_fc(
            num_feats=self.num_feats,
            num_classes=self.train_dataset.num_classes,
            distributed=self.distributed,
        )

        # get batch of real images
        x_real, y_real = self.get_real_batch()

        # get d_l_real / d_W
        grad_real = self.get_real_grad(
            x_real=x_real, y_real=y_real, model=self.backbone_model, fc=fc
        )

        # get synthetic images from pyramids
        x_syn, y_syn = self.distilled_dataset.get_data()

        # get d_l_syn / d_W
        grad_syn = self.get_syn_grad(
            x_syn=x_syn, y_syn=y_syn, model=self.backbone_model, fc=fc
        )

        # calculate meta loss as cosine distance between real and syn grads wrt W
        match_loss = 1 - torch.nn.functional.cosine_similarity(
            grad_real, grad_syn, dim=0
        )

        # we have to do manual grad scaling because of the second-order gradients
        match_loss *= AMP_SCALE

        # clear grads and backprop the meta loss
        self.distilled_dataset.optimizer.zero_grad()
        match_loss.backward()

        # we have to do manual grad scaling because of the second-order gradients
        for group in self.distilled_dataset.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad /= AMP_SCALE

        # update synthetic images (pyramids)
        self.distilled_dataset.optimizer.step()

        return match_loss.item() / AMP_SCALE

    def get_real_batch(self) -> Tuple[Tensor, Tensor]:

        batch_real = next(self.train_iter, None)

        # reset the dataloader if we're at the end
        if batch_real is None:
            self.train_iter = iter(self.train_loader)
            batch_real = next(self.train_iter, None)

        x_real, y_real = batch_real
        x_real = x_real.cuda(non_blocking=True)
        y_real = y_real.cuda(non_blocking=True)

        return x_real, y_real

    def get_real_grad(
        self, x_real: Tensor, y_real: Tensor, model: nn.Module, fc: nn.Module
    ) -> Tensor:

        # detaching because of paranoia
        x_real = x_real.detach()
        y_real = y_real.detach()

        # auto-casting was a *huge* speed up
        with autocast(enabled=True):

            # we augment each real image just one time because we loaded augs_per_batch * len(x_syn) real images
            x_real = self.real_augmentor(x_real)

            # (x - mean) / std
            x_real = self.train_dataset.normalize(x_real)

            # get \phi(x)
            z_real = model(x_real)

            # get output of randomly initialized linear classifier W
            out_real = fc(z_real)

            # get l_real
            loss_real = nn.functional.cross_entropy(out_real, y_real)

        # manually calculate d_l_real / d_W
        # do not need to create or retain this graph because we only back-propagate through the synthetic side
        grad_real_w, grad_real_b = torch.autograd.grad(
            loss_real,
            [fc.linear.weight, fc.linear.bias],
            retain_graph=False,
            create_graph=False,
        )

        # flattening weight and bias grads into a single vector
        grad_real = torch.cat(
            [grad_real_w.detach().flatten(), grad_real_b.detach().flatten()], dim=0
        )

        return grad_real

    def get_syn_grad(
        self, x_syn: Tensor, y_syn: Tensor, model: nn.Module, fc: nn.Module
    ) -> Tensor:

        # auto-casting was a *huge* speed up
        with autocast(enabled=True):

            # augment create augs_per_batch copies of x_syn and apply different aug to each one
            x_syn = self.syn_augmentor(
                torch.cat([x_syn for a in range(self.cfg.augs_per_batch)])
            )

            # create matching copies of labels
            # maybe use soft labels in future work?
            y_syn = torch.cat([y_syn for a in range(self.cfg.augs_per_batch)])

            # (x - mean) / std
            x_syn = self.train_dataset.normalize(x_syn)

            # get \phi(x)
            z_syn = model(x_syn)

            # get output of randomly initialized linear classifier W
            out_syn = fc(z_syn)

            # get l_syn
            loss_syn = nn.functional.cross_entropy(out_syn, y_syn)

        # manually calculate d_l_syn / d_W
        # we need to create and retain this graph because we need to back-propagate through it
        grad_syn_w, grad_syn_b = torch.autograd.grad(
            loss_syn,
            [fc.linear.weight, fc.linear.bias],
            retain_graph=True,
            create_graph=True,
        )

        # flattening weight and bias grads into a single vector
        grad_syn = torch.cat([grad_syn_w.flatten(), grad_syn_b.flatten()], dim=0)

        return grad_syn

    # saving synthetic images and labels to disk
    def save_data(self):
        with torch.no_grad():
            syn_images, syn_labels = self.distilled_dataset.get_data()
            syn_images = syn_images.clone().detach()
            syn_labels = syn_labels.clone().detach()

        save_dict = {
            "images": syn_images.cpu(),
            "labels": syn_labels.cpu(),
        }

        torch.save(save_dict, "{}/data.pth".format(self.log_dir))

    def load_checkpoint(self):
        # if self.cfg.job_id is None:
        #     return
        # load_dir = os.path.join("checkpoints", self.cfg.job_id)
        if os.path.exists(os.path.join(self.log_dir, "ckpt.pth")):
            print("Checkpoint found! Resuming...")
            load_dict = torch.load(
                os.path.join(self.log_dir, "ckpt.pth"), weights_only=False
            )
            self.distilled_dataset.load_from_dict(load_dict["synset"])
            self.global_step = load_dict["global_step"]

            torch.set_rng_state(load_dict["random_state"]["torch"])
            torch.cuda.set_rng_state_all(load_dict["random_state"]["cuda"])
            random.setstate(load_dict["random_state"]["python"])
            np.random.set_state(load_dict["random_state"]["numpy"])
        else:
            print("No checkpoint found, starting from scratch...")

    def save_checkpoint(self):
        print("Saving checkpoint...")
        random_state = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        }

        save_dict = {
            "synset": self.distilled_dataset.get_save_dict(),
            "global_step": self.global_step,
            "random_state": random_state,
        }

        torch.save(save_dict, os.path.join(self.log_dir, "tmp.pth"))
        os.rename(
            os.path.join(self.log_dir, "tmp.pth"),
            os.path.join(self.log_dir, "ckpt.pth"),
        )
        print("Saved!")

    def handle_interrupt(self, signum: int, frame: types.FrameType) -> None:
        print(f"Caught the signal {signum}! In the error handler")
        self.graceful_interrupt()

    def graceful_interrupt(self):
        print("Job interrupted by SLURM")
        print("Saving checkpoint...")
        self.save_checkpoint()
        print("Checkpoint saved!")
        self.train_loader._shutdown_workers()
        print("Closing W&B...")
        try:
            wandb.finish()
        except Exception as e:
            print("Error finishing wandb: ", e)
        print("Exiting (and hopefully re-queueing)")
        sys.exit(signal.SIGUSR1 + 128)
