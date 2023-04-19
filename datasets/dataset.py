# -*- coding: utf-8 -*-
# @Author  : You.P.J
# @Time    : 2023/3/29 21:08
# @Function: 数据集

from PIL import Image
from torch.utils.data import Dataset


class Cholec80Dataset(Dataset):
    def __init__(self, img_path_list, tool_list, phase_list, transform=None):
        self.img_path_list = img_path_list
        self.tool_list = tool_list
        self.phase_list = phase_list
        self.transform = transform

    def __getitem__(self, index):
        tool = self.tool_list[index]
        phase = self.phase_list[index]

        img_path = self.img_path_list[index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img, tool, phase

    def __len__(self):
        return len(self.img_path_list)


