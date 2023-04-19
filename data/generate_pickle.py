# -*- coding: utf-8 -*-
# @Author  : You.P.J
# @Time    : 2023/3/28 18:39
# @Function: 生成序列化数据，用于训练和测试

import os
import pickle
import configs


def get_img_dir_list():
    """
    获取图像的绝对路径

    :return: 图像路径文件夹集合
    """
    folder_list = os.listdir(configs.Cholec80ProcessedFrames)
    img_dir_list = [configs.Cholec80ProcessedFrames + "/" + folderNum for folderNum in folder_list]

    return img_dir_list


def get_img_path_list(img_dir):
    """
    获取图像绝对路径

    :param img_dir: 图像的路径文件夹
    :return:
    """
    img_name_list = os.listdir(img_dir)
    img_path_list = [img_dir + "/" + img_name for img_name in img_name_list]

    return img_path_list


def get_tool_path_list():
    """
    获取手术器具txt绝对路径

    :return:
    """
    tool_name_list = os.listdir(configs.Cholec80ToolAnnotations)
    tool_path_list = [configs.Cholec80ToolAnnotations + "/" + tool_name for tool_name in tool_name_list]

    return tool_path_list


def get_phase_path_list():
    """
    获取手术阶段txt绝对路径

    :return:
    """
    phase_name_list = os.listdir(configs.Cholec80PhaseAnnotations)
    phase_path_list = [configs.Cholec80PhaseAnnotations + "/" + phase_name for phase_name in phase_name_list]

    return phase_path_list


## 手术图像处理
img_dir_list = sorted(get_img_dir_list())

## 手术器具处理
tool_path_list = sorted(get_tool_path_list())

## 手术阶段数据处理
phase_path_list = sorted(get_phase_path_list())
index2phase = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting', 'GallbladderDissection',
               'GallbladderPackaging', 'CleaningCoagulation', 'GallbladderRetraction']
phase2index = {index2phase[index]: index for index in range(len(index2phase))}

## 所需序列化数据的集合
img_list = []
tool_list = []
phase_list = []

## 开始序列化数据
for i in range(len(img_dir_list)):
    img_list_tmp = []
    tool_list_tmp = []
    phase_list_tmp = []
    img_dir = img_dir_list[i]
    img_path_list = sorted(get_img_path_list(img_dir))  # 图像绝对路径集合
    tool_file = open(tool_path_list[i])  # 手术工具文件
    phase_file = open(phase_path_list[i])  # 手术阶段文件

    ## 手术器具提取，并添加手术图像
    tool_count = 0
    for tool_line in tool_file:
        tool_count += 1
        ## 去除第一行标注，且注意图像从第0张开始，tool从第二行开始
        if tool_count > 1:
            img_list_tmp.append(img_path_list[tool_count - 2])
            tool_split = tool_line.split()
            ## 去掉第一列标注，添加手术器具向量
            tool_vector = []
            for col in range(1, len(tool_split)):
                tool_vector.append(int(tool_split[col]))
            tool_list_tmp.append(tool_vector)
            # print(img_list_tmp)
            # print(tool_list_tmp)
            # assert 0
    ## 手术阶段提取
    phase_count = 0
    for phase_line in phase_file:
        phase_count += 1
        ## 每25帧取一次，最初有效行为2
        if phase_count % 25 == 2 and (phase_count // 25) < len(tool_list_tmp):
            phase_split = phase_line.split()
            phase_list_tmp.append(phase2index[phase_split[1]])
    ## 数据拼接
    img_list += img_list_tmp
    tool_list += tool_list_tmp
    phase_list += phase_list_tmp

with open("." + configs.ImagePicklePath, "wb") as f:
    pickle.dump(img_list, f)

with open("." + configs.ToolPicklePath, "wb") as f:
    pickle.dump(tool_list, f)

with open("." + configs.PhasePicklePath, "wb") as f:
    pickle.dump(phase_list, f)

print("数据序列化完成！")
