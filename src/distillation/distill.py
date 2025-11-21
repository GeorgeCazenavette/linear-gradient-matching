import glob
import json
import os
import random

import numpy as np
import torch

import wandb
from config import DistillCfg

from .linear_gm import LinearGM


def main(cfg: DistillCfg):
    model_dir = os.path.join("logged_files", cfg.job_tag, cfg.dataset, cfg.model)
    print("Searching for checkpoints in {}".format(model_dir))
    syn_set_files = sorted(
        list(glob.glob(os.path.join(model_dir, "**", "05000.pth"), recursive=True))
    )
    if len(syn_set_files) > 0 and cfg.skip_if_exists:
        print("This distillation already done.")
        print("Exiting...")
        exit()

    wandb.init(
        id=cfg.run_name,
        job_type="distillation",
        project="Linear-Gradient-Matching",
        config=cfg.as_dict(),
        settings=wandb.Settings(code_dir="."),
        resume="allow",
    )
    config = wandb.config
    cfg: DistillCfg = cfg.from_dict(config.as_dict(), skip_unsettable=True)
    cfg.run_name = wandb.run.name

    log_dir = os.path.join(
        "logged_files", cfg.job_tag, cfg.dataset, cfg.model, cfg.run_name
    )

    os.makedirs(log_dir, exist_ok=True)

    cfg_json_file = "{}/cfg.json".format(log_dir)
    with open(cfg_json_file, "w") as f:
        f.write(json.dumps(cfg.as_dict(), indent=4))

    distillation = LinearGM(cfg=cfg)
    distillation.distill()

    cfg_pkl_file = "{}/cfg.pth".format(log_dir)
    torch.save(cfg, cfg_pkl_file)

    print("CFG Saved!")
    wandb.finish()
    print("WandB Finshed!")
    del distillation.train_loader, distillation.test_loader
    print("Data Loaders Killed!")
    print("Distillation Over")


if __name__ == "__main__":
    torch.manual_seed(3407)
    random.seed(3407)
    np.random.seed(3407)
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = DistillCfg(explicit_bool=True).parse_args()
    print(args)
    main(args)
    print("Should be ending now")
    os._exit(0)
