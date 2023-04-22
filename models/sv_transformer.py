# -*- encoding: utf-8 -*-
'''
@File    :   sv_transformer.py
@Time    :   2023/04/21 10:40:50
@Author  :   YouPingJie
@Function:   
'''

import torch
import torchvision
from torch import nn
from .transformer import Transformer
from .swin_transformer import build_swin_transformer
from .video_swin_transfromer import build_video_swin_transformer


class SVTransformer(nn.Module):
    def __init__(self, clip_size, *args, **kwargs) -> None:
        super(SVTransformer, self).__init__(*args, **kwargs)
        self.clip_size = clip_size
        self.swin_transformer = build_swin_transformer()
        self.video_swin_transformer = build_video_swin_transformer()

        self.tool_proj = nn.Sequential(
            nn.Linear(22000,512),
            nn.ReLU(),
            nn.Linear(512,512)
        )

    def forward(self, x):
        frames = x
        clips = x.view(-1, 3, self.clip_size, 224, 224)

        spatial_features = self.swin_transformer(frames)
        temporal_features = self.video_swin_transformer(clips)

        spatial_features = self.tool_proj(spatial_features)
        tool_features = self.tool_transformer_encoder(spatial_features)
        print(tool_features.shape)

        return x
