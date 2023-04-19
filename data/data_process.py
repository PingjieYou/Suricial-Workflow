# -*- coding: utf-8 -*-
# @Author  : You.P.J
# @Time    : 2023/3/28 15:41
# @Function: 数据的处理

import os
import cv2
import configs
from PIL import Image
from tqdm import tqdm


def extract_content(image):
    """
    提取内容区域，过滤背景区域

    :param image: 图像
    :return:
    """
    ## 提取内容区域并去除噪声
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 15, 255, cv2.THRESH_BINARY)
    binary_image = cv2.medianBlur(binary_image, 19)
    w, h = binary_image.shape[0], binary_image.shape[1]
    ## 获取内容区域的坐标
    edges_x = []
    edges_y = []
    for i in range(w):
        for j in range(10, h - 10):
            if binary_image.item(i, j) != 0:
                edges_x.append(i)
                edges_y.append(j)
    ## 全为内容，直接返回
    if not edges_x:
        return image
    ## 提取内容区域
    left = min(edges_x)  # left border
    right = max(edges_x)  # right
    width = right - left
    bottom = min(edges_y)  # bottom
    top = max(edges_y)  # top
    height = top - bottom
    content_image = image[left:left + width, bottom:bottom + height]

    return content_image


def preprocessing(img_path, save_path):
    """
    图像预处理

    :param img_path: 图像路径
    """
    image = cv2.imread(img_path)
    ## 缩放成300的比例
    dim = (int(image.shape[1] / image.shape[0] * 300), 300)
    image = cv2.resize(image, dim)
    ## 提取内容区域，并缩放成固定大小
    image = extract_content(image)
    image = cv2.resize(image, (configs.WIDTH, configs.HEIGHT))
    ## 转化为RGB后保存
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image.save(save_path)


count = 1
## 视频帧的预处理
for videoNum in os.listdir(configs.Cholec80Frames):
    for frameNum in tqdm(os.listdir(configs.Cholec80Frames + "/" + videoNum), desc='preprocessing', unit_scale=True):
        if (os.path.exists(configs.Cholec80ProcessedFrames + "/" + videoNum) == False):
            os.mkdir(configs.Cholec80ProcessedFrames + "/" + videoNum)
        if count > 45:
            preprocessing(configs.Cholec80Frames + "/" + videoNum + "/" + frameNum, configs.Cholec80ProcessedFrames + "/" + videoNum + "/" + frameNum)
    count += 1
