#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : sampler.py
# @Time     : 2023/4/22 下午3:05
# @Author   : YouPingJie
# @Function : 采样器

from torch.utils.data import Sampler

class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)
