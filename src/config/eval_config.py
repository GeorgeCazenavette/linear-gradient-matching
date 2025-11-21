from typing import Literal

import torch
from tap import Tap


class EvalCfg(Tap):
    dataset: str
    model: str
    eval_model: str
    job_tag: str = "distillation"
    data_root: str = "data/datasets"
    num_workers: int = 16
    real_batch_size: int = 100
    real_res: int = 256
    crop_res: int = 224
    num_eval: int = 5
    eval_epochs: int = 1000
    train_crop_mode: Literal["center", "random"] = "random"

    device_count: int = torch.cuda.device_count()

    skip_if_exists: bool = True

    patience: int = 5
    eval_it: int = -1

    checkpoint_it: int = 100

    job_id: str | None = None
