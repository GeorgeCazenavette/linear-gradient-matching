# Dataset Distillation for Pre-Trained Self-Supervised Vision Models

**[Project Page](https://linear-gradient-matching.github.io) | [arXiv](http://arxiv.org/abs/2511.16674)**

[George Cazenavette](https://georgecazenavette.github.io/) · [Antonio Torralba](http://web.mit.edu/torralba/www/) · [Vincent Sitzmann](https://www.vincentsitzmann.com/)

Massachusetts Institute of Technology

![birds_high](assets/birds_high.gif)
## Method Overview

<!-- Method diagram or architecture figure -->
<p align="center">
  <img src="assets/linear_dd.png" alt="Method Overview" width="800"/>
</p>

We optimize our synthetic images such that they induce similar gradients as real images when training a linear classifier (W) on top of a pre-trained model (ϕ). To do this, we perform a bi-level optimization by finding the cosine distance between the real and synthetic gradients and back-propagating through the initial gradient calculation all the way to the synthetic images themselves.
We then evaluate by training a new linear classifier from scratch on the distilled data. Please see our [Project Page](https://linear-gradient-matching.github.io) and [Paper](http://arxiv.org/abs/2511.16674) for more details.

## Installation

### Prerequisites
- Tested with Python 13 and CUDA 12.9 and 13.0

### Setup

1. Clone the repository:
```bash
git clone https://github.com/GeorgeCazenavette/linear-gradient-matching
cd linear-gradient-matching/src
```

2. Create a virtual environment:
```bash
conda create -n linear_gm python=13
conda activate linear_gm
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Introduction
To see all available models and datasets, please see ``models/__init__.py`` and ``data/dataloaders/__init__.py`` respectively. 

The main 4 models used in the paper are the ViT-B variants of CLIP (``clip_vitb``), DINO-v2 (``dinov2_vitb``), EVA-02 (``eva02_vitb``), and MoCo-v3 (``mocov3_vitb``). 



## Data Preparation

The default ``data_root`` is ``data/datasets``. For ImageNet, you'll need to either store or symlink the dataset at ``data/datasets/imagenet``. All other datasets should download automatically.

## Distillation

The following command will distill ``imagenet-birds`` using ``dinov2_vitb`` and tag the job as ``distillation``. This should take around 30 minutes on a single rtx4090 GPU. 

```bash
python -m distillation.distill --model=dinov2_vitb --dataset=imagenet-birds --job_tag=distillation
```

The distilled data will be stored at ``logged_files/{job_tag}/{dataset}/{model}/{run_name}/data.pth``.

The distillation will automatically use all available GPUs. 

If you run out of VRAM, try lowering ``--augs_per_batch`` (default is 10).

For ``imagenet-1k``, we did not try ``--augs_per_batch > 3`` due to memory constraints. 

You can resume an interrupted run with ``--run_name={wandb_run_name}``.

## Evaluation

The following command will train a linear head on top of ``clip_vitb`` using the images distilled from ``imagenet-birds`` using ``dinov2_vitb`` (from the job with tag ``distillation``).

```bash
python -m distillation.eval --model=dinov2_vitb --eval_model=clip_vitb --dataset=imagenet-birds --job_tag=distillation
```

The mean and std of the test accuracy will be stored at ``logged_files/{job_tag}/{dataset}/{model}/{run_name}/eval/{eval_model}.pth``

Please see our [Paper](https://arxiv.org/abs/XXXX.XXXXX) for full evaluation results.

## Baselines
<details>
<summary>(Click to expand)</summary>

### Neighbors

The following command will find the nearest real neighbors for the images distilled from ``imagenet-birds`` using ``dinov2_vitb`` (from the job with tag ``distillation``).

```bash
python -m baselines.neighbors --model=dinov2_vitb --dataset=imagenet-birds --job_tag=distillation
```

This will create another stored dataset with ``job_tag=distillation_neighbors``. 

You can then evaluate using the same method as [above](#evaluation). 

### Centroids
The following command will find the real centroid image of each class of ``imagenet-birds`` using ``dinov2_vitb``.

```bash
python -m baselines.neighbors --model=dinov2_vitb --dataset=imagenet-birds
```

This will create another stored dataset with ``job_tag=real_centroids``.

You can then evaluate using the same method as [above](#evaluation).

### Random
The following command will train a linear head on top of ``dinov2_vitb`` using one random image from each class of ``imagenet-birds`` (with seed ``0``).
```bash
python -m baselines.random_reals --dataset=imagenet-birds --model=dinov2_vitb --random_seed=0
```
We ran this with seeds ``0,1,2,3,4`` to obtain the numbers in the paper.


### Full Dataset
The following command will train a linear head on top of ``dinov2_vitb`` all the real images of ``imagenet-birds``.
```bash
python -m baselines.full_dataset --dataset=imagenet-birds --model=dinov2_vitb
```
This takes much longer than training on the distilled data, but running the same command should resume an interrupted run. By default, performance is averaged over 5 runs.

</details>

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{cazenavette2025lgm,
  title={Dataset Distillation for Pre-Trained Self-Supervised Vision Models},
  author={George Cazenavette and Antonio Torralba and Vincent Sitzmann},
  journal={{NeurIPS}},
  year={2025},
}
```

## Contact

For questions or issues, please open an issue on GitHub or email George at [gcaz@mit.edu](mailto:gcaz@mit.edu).
