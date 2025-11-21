import glob
import os

import kornia
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm

from config import NeighborsCfg
from data.dataloaders import (
    BaseRealDataset,
    get_dataset,
)
from models import get_model


@torch.no_grad()
def get_closest_images(
    syn_images: Tensor,
    syn_labels: Tensor,
    model: nn.Module,
    train_dataset: BaseRealDataset,
) -> Tensor:
    crop = kornia.augmentation.CenterCrop(224)

    syn_embeddings = model(crop(train_dataset.normalize(syn_images)))

    real_neighbors = []
    for x, y in tqdm(zip(syn_embeddings, syn_labels), total=len(syn_embeddings)):
        real_images = train_dataset.get_single_class(y.item()).cuda()
        normalized_real_images = train_dataset.normalize(real_images)
        cropped_real_images = crop(normalized_real_images)

        real_embeddings = torch.cat(
            [model(chunk) for chunk in torch.split(cropped_real_images, 100)]
        )

        x_norm = F.normalize(x, dim=0)
        real_norm = F.normalize(real_embeddings, dim=1)

        scores = real_norm @ x_norm

        nearest_idx = torch.argmax(scores)
        nearest_image = real_images[nearest_idx].clone()

        real_neighbors.append(nearest_image)

    real_neighbors = torch.stack(real_neighbors)

    return real_neighbors


def closest_reals(cfg: NeighborsCfg):
    print(cfg)
    syn_set_files = sorted(
        list(
            glob.glob(
                os.path.join(
                    "logged_files",
                    cfg.job_tag,
                    cfg.dataset,
                    cfg.model,
                    "**",
                    "data.pth",
                ),
                recursive=True,
            )
        )
    )
    if len(list(syn_set_files)) > 1:
        print("Warning: multiple syn sets found. Using the first one.")

    path_parts = syn_set_files[0].split("/")
    path_parts[1] = path_parts[1] + "_neighbors"
    save_directory = "/".join(path_parts[:-1])
    os.makedirs(save_directory, exist_ok=True)

    if os.path.exists(os.path.join(save_directory, "data.pth")) and cfg.skip_if_exists:
        print("These neighbors already found. Exiting.")
        exit(0)

    train_dataset, test_dataset = get_dataset(
        name=cfg.dataset,
        res=cfg.real_res,
        crop_res=cfg.crop_res,
        train_crop_mode="center",
        data_root=cfg.data_root,
    )

    syn_set = torch.load(syn_set_files[0])
    syn_images = syn_set["images"].cuda()
    syn_labels = syn_set["labels"].cuda()

    eval_model, num_feats = get_model(
        cfg.model, distributed=torch.cuda.device_count() > 1
    )

    real_neighbors = get_closest_images(
        syn_images=syn_images,
        syn_labels=syn_labels,
        model=eval_model,
        train_dataset=train_dataset,
    )

    path_parts = syn_set_files[0].split("/")
    path_parts[1] = path_parts[1] + "_neighbors"
    save_directory = "/".join(path_parts[:-1])
    os.makedirs(save_directory, exist_ok=True)

    save_dict = {
        "images": real_neighbors.cpu(),
        "labels": syn_labels.cpu(),
    }

    torch.save(save_dict, os.path.join(save_directory, "data.pth"))


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    cfg = NeighborsCfg(explicit_bool=True).parse_args()
    closest_reals(cfg)
