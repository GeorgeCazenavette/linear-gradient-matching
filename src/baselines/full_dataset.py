import os.path

import numpy as np
import torch
from torch.utils.data import DataLoader

from augmentation import AugBasic
from config import FullDatasetCfg
from data.dataloaders import get_dataset
from distillation.eval import Evaluator
from models import get_fc, get_model


def eval_full_dataset(cfg: FullDatasetCfg):
    torch.multiprocessing.set_sharing_strategy("file_system")

    save_dir = os.path.join(
        "baselines",
        "results",
        "full_dataset",
        cfg.dataset,
    )
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "{}.pth".format(cfg.model))
    checkpoint_path = os.path.join(save_dir, "checkpoint_{}.pth".format(cfg.model))

    if os.path.exists(save_path) and cfg.skip_if_exists:
        print(
            "This baseline has already been run. Set `--skip_if_exists=False` to force a redo."
        )
        exit(0)

    train_dataset, test_dataset = get_dataset(
        name=cfg.dataset,
        res=cfg.real_res,
        crop_res=cfg.crop_res,
        train_crop_mode=cfg.train_crop_mode,
        data_root=cfg.data_root
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        num_workers=cfg.workers_per_gpu * torch.cuda.device_count(),
        batch_size=cfg.batch_size_per_gpu * torch.cuda.device_count(),
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=cfg.workers_per_gpu * torch.cuda.device_count(),
        batch_size=cfg.batch_size_per_gpu * torch.cuda.device_count(),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    augmentor = AugBasic(crop_res=cfg.crop_res).cuda()

    eval_model, num_feats = get_model(
        cfg.model, distributed=torch.cuda.device_count() > 1
    )

    evaluator = Evaluator(
        train_loader=train_loader,
        test_loader=test_loader,
        model=eval_model,
        checkpoint_path=checkpoint_path,
        augmentor=augmentor,
        epochs=cfg.eval_epochs,
        eval_it=cfg.eval_it,
        patience=cfg.patience,
        checkpoint_it=cfg.checkpoint_it,
        normalize=train_dataset.normalize,
        num_feats=num_feats,
        num_classes=train_dataset.num_classes,
        num_eval=cfg.num_eval,
    )

    evaluator.train_and_eval()

    top1_mean = np.mean(evaluator.top1_list)
    top1_std = np.std(evaluator.top1_list)

    top5_mean = np.mean(evaluator.top5_list)
    top5_std = np.std(evaluator.top5_list)

    save_dict = {
        "top1_mean": top1_mean,
        "top1_std": top1_std,
        "top5_mean": top5_mean,
        "top5_std": top5_std,
    }

    torch.save(save_dict, save_path)
    print(f"Results saved to {save_path}")
    os.remove(checkpoint_path)

    print("Top 1 Mean ± Std: {:.2f} ± {:.2f}".format(top1_mean*100, top1_std*100))

if __name__ == "__main__":
    args = FullDatasetCfg(explicit_bool=True).parse_args()
    print(args)
    eval_full_dataset(args)
