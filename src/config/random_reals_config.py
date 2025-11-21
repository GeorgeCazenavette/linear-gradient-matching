
import torch
from tap import Tap


class RandomRealsConfig(Tap):
    dataset: str
    model: str
    random_seed: int
    data_root: str = "data/datasets"
    workers_per_gpu: int = 16
    batch_size_per_gpu: int = 128
    real_res: int = 256
    crop_res: int = 224
    eval_epochs: int = 1000

    device_count: int = torch.cuda.device_count()

    job_tag: str | None = None

    patience: int = 5
    eval_it: int = -1

    skip_if_exists: bool = True

    job_id: str | None = None

    checkpoint_it: int = 100

    ipc: int = 1
