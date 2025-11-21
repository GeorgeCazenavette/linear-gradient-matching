import torch
from tap import Tap


class NeighborsCfg(Tap):
    dataset: str
    model: str
    job_tag: str = "distillation"
    data_root: str = "data/datasets"
    num_workers: int = 16
    real_batch_size: int = 100
    real_res: int = 256
    crop_res: int = 224

    device_count: int = torch.cuda.device_count()

    run_name: str | None = None

    job_id: str | None = None

    skip_if_exists: bool = True
