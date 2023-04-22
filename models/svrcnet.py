#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : svrcnet.py
# @Time     : 2023/4/22 下午2:18
# @Author   : YouPingJie
# @Function : Surgical Video Recurrent Convolutional Network，用于阶段识别

import torch
import torchvision
from torch import nn


class SVRCNet(nn.Module):
    def __init__(self, clip_size):
        super(SVRCNet, self).__init__()
        self.clip_size = clip_size
        resnet = torchvision.models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        self.fc = nn.Linear(512, 7)
        self.dropout = nn.Dropout(p=0.2)

        nn.init.xavier_normal_(self.lstm.all_weights[0][0])
        nn.init.xavier_normal_(self.lstm.all_weights[0][1])
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, self.clip_size, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        y = self.dropout(y)
        y = self.fc(y)
        return y
