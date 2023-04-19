# -*- coding: utf-8 -*-
# @Author  : You.P.J
# @Time    : 2023/4/3 21:54
# @Function:

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import Bottleneck, BasicBlock

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ResNet_CSRA(ResNet):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, num_heads, lam, num_classes, depth=101, input_dim=2048, cutmix=None):
        self.block, self.layers = self.arch_settings[depth]
        self.depth = depth
        super(ResNet_CSRA, self).__init__(self.block, self.layers)
        self.cls = MultiHeadAttention(num_heads, lam, input_dim, num_classes)

    def backbone(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        x = self.backbone(x)
        logit = self.cls(x)
        return logit

    def init_weights(self, pretrained=True, cutmix=None):
        if cutmix is not None:
            print("backbone params inited by CutMix pretrained model")
            state_dict = torch.load(cutmix)
        elif pretrained:
            print("backbone params inited by Pytorch official model")
            model_url = model_urls["resnet{}".format(self.depth)]
            state_dict = model_zoo.load_url(model_url)


class MultiHeadAttention(nn.Module):
    temp_settings = {  # softmax temperature settings
        1: [1],
        2: [1, 99],
        4: [1, 2, 4, 99],
        6: [1, 2, 3, 4, 5, 99],
        8: [1, 2, 3, 4, 5, 6, 7, 99]
    }

    def __init__(self, num_heads, lam, input_dim, num_classes):
        super(MultiHeadAttention, self).__init__()
        self.temp_list = self.temp_settings[num_heads]  # 多头注意力的temp设置
        self.multi_head = nn.ModuleList([
            CSRA(input_dim, num_classes, self.temp_list[i], lam) for i in range(num_heads)
        ])

    def forward(self, x):
        logit = 0
        for head in self.multi_head:
            logit += head(x)

        return logit


class CSRA(nn.Module):
    # class-specific residual attention
    def __init__(self, input_dim, num_classes, T, lam):
        super(CSRA, self).__init__()
        self.T = T  # temperature
        self.lam = lam  # Lambda
        self.head = nn.Conv2d(input_dim, num_classes, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        score = self.head(x)  # (bs,num_classes,h,w)
        norm = torch.norm(self.head.weight, dim=1, keepdim=True).transpose(0, 1)  # (1,num_classes,1,1)
        score = score / norm
        score = score.flatten(2)  # (bs,num_classes,h*w)
        base_logit = torch.mean(score, dim=2)  # (bs,num_classes)

        if self.T == 99:  # max-pooling
            attn_logit = torch.max(score, dim=2)[0]
        else:
            score_softmax = self.softmax(score * self.T)
            attn_logit = torch.sum(score * score_softmax, dim=2)

        return base_logit + self.lam * attn_logit

        return score
