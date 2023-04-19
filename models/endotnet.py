# -*- encoding: utf-8 -*-
'''
@File    :   endotnet.py
@Time    :   2023/04/10 14:38:37
@Author  :   YouPingJie 
'''

import torch
import torchvision
from torch import nn


class EndotNet(nn.Module):
    def __init__(self):
        super(EndotNet, self).__init__()
        alexnet = torchvision.models.alexnet(pretrained=True)
        self.feature_extraction = alexnet.features
        self.avgpool = alexnet.avgpool
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(in_features=256*6*6, out_features=4096)
        self.dense2 = nn.Linear(in_features=4096, out_features=4096)
        self.tool_proj = nn.Linear(in_features=4096, out_features=7)
        self.phase_proj = nn.Linear(in_features=4103, out_features=7)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        tool_logits = self.tool_proj(x)
        phase_logits = self.phase_proj(torch.cat([x, tool_logits], dim=1))

        return tool_logits, phase_logits
