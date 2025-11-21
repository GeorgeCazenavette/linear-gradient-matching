
import torch
from tap import Tap


class CentroidRealsCfg(Tap):
    dataset: str
    model: str
    data_root: str = "data/datasets"
    num_workers: int = 16
    real_batch_size: int = 100
    real_res: int = 256
    crop_res: int = 224

    device_count: int = torch.cuda.device_count()

    skip_if_exists: bool = True

    job_tag: str | None = None
    job_id: str | None = None
