# -*- coding: utf-8 -*-
# @Author  : You.P.J
# @Time    : 2023/4/8 9:23
# @Function: Multi-Task Recurrent Convolutional Network

import torch
import torchvision
from torch import nn

class MultiTaskRCNN(nn.Module):
    def __init__(self,sequence_length):
        super(MultiTaskRCNN, self).__init__()
        self.sequence_length = sequence_length
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.share = nn.Sequential()
        self.share.add_module("conv1",resnet50.conv1)
        self.share.add_module("bn1", resnet50.bn1)
        self.share.add_module("relu", resnet50.relu)
        self.share.add_module("maxpool", resnet50.maxpool)
        self.share.add_module("layer1", resnet50.layer1)
        self.share.add_module("layer2", resnet50.layer2)
        self.share.add_module("layer3", resnet50.layer3)
        self.share.add_module("layer4", resnet50.layer4)
        self.share.add_module("avgpool", resnet50.avgpool)
        self.lstm = nn.LSTM(2048, 512, batch_first=True, dropout=1)
        self.fc = nn.Linear(512, 7)
        self.fc2 = nn.Linear(2048, 7)
        nn.init.xavier_normal(self.lstm.all_weights[0][0])
        nn.init.xavier_normal(self.lstm.all_weights[0][1])
        nn.init.xavier_uniform(self.fc.weight)
        nn.init.xavier_uniform(self.fc2.weight)

    def forward(self,x):
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        z = self.fc2(x)
        x = x.view(-1, self.sequence_length, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        y = self.fc(y)
        return z, y
    
class MTRCNN_CL(nn.Module):
    def __init__(self,clip_size):
        super(MTRCNN_CL, self).__init__()
        self.clip_size = clip_size
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.share = nn.Sequential()
        self.share.add_module("conv1",resnet50.conv1)
        self.share.add_module("bn1", resnet50.bn1)
        self.share.add_module("relu", resnet50.relu)
        self.share.add_module("maxpool", resnet50.maxpool)
        self.share.add_module("layer1", resnet50.layer1)
        self.share.add_module("layer2", resnet50.layer2)
        self.share.add_module("layer3", resnet50.layer3)
        self.share.add_module("layer4", resnet50.layer4)
        self.share.add_module("avgpool", resnet50.avgpool)
        # phase features extraction
        self.lstm = nn.LSTM(2048,512,batch_first=True,dropout=1)
        self.fc = nn.Linear(512, 7)
        # tool features extraction
        self.fc2 = nn.Linear(2048,512)
        self.fc3 = nn.Linear(512, 7)
        self.relu = nn.ReLU()
        # phase to tool features
        self.phase2tool_proj = nn.Linear(512,7)

        torch.nn.init.xavier_normal(self.lstm.all_weights[0][0])
        torch.nn.init.xavier_normal(self.lstm.all_weights[0][1])
        torch.nn.init.xavier_uniform(self.fc.weight)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        torch.nn.init.xavier_uniform(self.fc3.weight)
        torch.nn.init.xavier_uniform(self.phase2tool_proj.weight)
    
    def forward(self,x):
        shared_features = self.share.forward(x)
        shared_features = shared_features.view(-1, 2048)
        z = self.fc2(shared_features)
        z = self.relu(z)
        tool_logits = self.fc3(z)

        clip_feature = shared_features.view(-1, self.clip_size, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(clip_feature)
        y = y.contiguous().view(-1, 512)
        phase_logits = self.fc(self.relu(y))

        phase2tool = self.phase2tool_proj(y)
        return tool_logits,phase_logits, phase2tool