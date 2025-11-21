from typing import Tuple

import clip
import timm
import torch
import torch.nn as nn
from transformers import AutoModel, pipeline

from .dino import vision_transformer
from .dino.utils import load_pretrained_weights
from .lambda_layer import LambdaLayer
from .linear_classifier import LinearClassifier
from .moco_vision_tansformer import VisionTransformerMoCoV3


def get_model(name: str, distributed: bool) -> Tuple[nn.Module, int]:

    match name:

        case "dino_vits8":
            model = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
            num_feat = 384

        case "dino_vits16":
            model = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
            num_feat = 384

        case "dino_vitb8":
            model = torch.hub.load("facebookresearch/dino:main", "dino_vitb8")
            num_feat = 768

        case "dino_vitb16":
            model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
            num_feat = 768

        case "dinov2_vits":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            num_feat = 384

        case "dinov2_vitb":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            num_feat = 768

        case "dinov2_vitl":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
            num_feat = 1024

        case "dinov3_vitb16":
            model = pipeline(
                model="facebook/dinov3-vitb16-pretrain-lvd1689m",
                task="image-feature-extraction",
            ).model
            model = WrappedModel(model)
            num_feat = 768

        case "clip_resnet50":
            model = clip.load("RN50")[0].visual.float().cuda()
            num_feat = 1024

        case "clip_resnet101":
            model = clip.load("RN101")[0].visual.float().cuda()
            num_feat = 512

        case "clip_vitb":
            model = clip.load("ViT-B/32")[0].visual.float().cuda()
            num_feat = 512

        case "clip_vitb":
            model = clip.load("ViT-B/16")[0].visual.float().cuda()
            num_feat = 512

        case "clip_vitl":
            model = clip.load("ViT-L/14")[0].visual.float().cuda()
            num_feat = 768

        case "eva02_vitl":
            model = timm.create_model(
                "eva02_large_patch14_224.mim_in22k", pretrained=True
            )
            num_feat = 1024

        case "eva02_vitb":
            model = timm.create_model(
                "eva02_base_patch14_224.mim_in22k", pretrained=True
            )
            num_feat = 768

        case "eva02_vits":
            model = timm.create_model(
                "eva02_small_patch14_224.mim_in22k", pretrained=True
            )
            num_feat = 384

        case "eva02_vitt":
            model = timm.create_model(
                "eva02_tiny_patch14_224.mim_in22k", pretrained=True
            )
            num_feat = 192


        case "mocov3_vitb":
            model = VisionTransformerMoCoV3.from_pretrained(
                "nyu-visionx/moco-v3-vit-b", num_classes=0
            )
            num_feat = 768

        case "mocov3_vitl":
            model = VisionTransformerMoCoV3.from_pretrained(
                "nyu-visionx/moco-v3-vit-l", num_classes=0
            )
            num_feat = 1024

        case _:
            raise NotImplementedError("Model {} not implemented".format(name))

    if distributed:
        model = nn.DataParallel(model)
    model = model.cuda()
    # this is to disable running stats in batchnorm, dropout, etc
    model.eval()

    return model, num_feat


def get_fc(num_feats: int, num_classes: int, distributed: bool):

    fc = LinearClassifier(dim=num_feats, num_labels=num_classes).cuda()

    if distributed:
        fc = nn.DataParallel(fc)
        fc.linear = fc.module.linear

    fc.eval()

    return fc


class WrappedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model.vision_model(*args, **kwargs)[1]
