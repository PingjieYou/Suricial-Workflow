# -*- encoding: utf-8 -*-
'''
@File    :   tmrnet.py
@Time    :   2023/04/19 18:25:43
@Author  :   YouPingJie
@Function:   Temporal Memory Recurrent Network
'''

import torch
import torchvision
from torch import nn

class TMRNet(nn.Module):
    def __init__(self, clip_size,*args, **kwargs) -> None:
        super(TMRNet,self).__init__(*args, **kwargs)
        self.clip_size = clip_size
        resnet = torchvision.models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential()
        self.feature_extractor.add_module("conv1", resnet.conv1)
        self.feature_extractor.add_module("bn1", resnet.bn1)
        self.feature_extractor.add_module("relu", resnet.relu)
        self.feature_extractor.add_module("maxpool", resnet.maxpool)
        self.feature_extractor.add_module("layer1", resnet.layer1)
        self.feature_extractor.add_module("layer2", resnet.layer2)
        self.feature_extractor.add_module("layer3", resnet.layer3)
        self.feature_extractor.add_module("layer4", resnet.layer4)
        self.feature_extractor.add_module("avgpool", resnet.avgpool)
        self.lstm = nn.LSTM(2048,512,batch_first=True)
        self.fc = nn.Linear(512, 7)
        self.dropoout = nn.Dropout(0.2)

        nn.init.xavier_normal_(self.lstm.all_weights[0][0])
        nn.init.xavier_normal_(self.lstm.all_weights[0][1])
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        x = x.view(-1,3,224,224)
        x = self.feature_extractor(x)
        x = x.view(-1, self.clip_size, 2048)

        self.lstm.flatten_parameters()

        x, _ = self.lstm(x)
        x = x.contiguous().view(-1, 512)
        x = self.dropoout(x)
        x = self.fc(x)
        return x