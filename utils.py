# -*- coding: utf-8 -*-
# @Author  : You.P.J
# @Time    : 2023/4/3 20:20
# @Function: 常用函数工具

import time
import torch
import pickle
import configs
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


def get_date_label():
    """
    获取数据，将tool和phase合并

    :return:
    """
    with open(configs.ImagePicklePath, "rb") as f:
        img_list = pickle.load(f)

    with open(configs.ToolPicklePath, "rb") as f:
        tool_list = pickle.load(f)

    with open(configs.PhasePicklePath, "rb") as f:
        phase_list = pickle.load(f)

    tool_list = np.asarray(tool_list, dtype=np.int64)
    phase_list = np.asarray(phase_list, dtype=np.int64).reshape(-1, 1)
    combined_label_list = np.hstack((tool_list, phase_list))

    return img_list, combined_label_list


def train_test_valid_split(data, label, test_size=0.6, fix_modal=True):
    """
    划分训练集、测试集和验证集

    :param test_size: 测试集和验证集占总数据的比例
    :param fix_modal: 为True比例固定为4:3:3；False按设置比率划分
    :return:
    """
    if fix_modal:
        data_len = len(data)
        train_data, train_label = data[:int(data_len * 0.4)], label[:int(data_len * 0.4)]
        test_data, test_label = data[int(data_len * 0.4):int(data_len * 0.7)], label[int(data_len * 0.4):int(data_len * 0.7)]
        val_data, val_label = data[int(data_len * 0.7):], label[int(data_len * 0.7):]
        return train_data, test_data, val_data, train_label, test_label, val_label

    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=test_size, random_state=configs.seed)
    test_data, val_data, test_label, val_label = train_test_split(test_data, test_label, test_size=0.5, random_state=configs.seed)
    return train_data, test_data, val_data, train_label, test_label, val_label

def train_val_split(data, label, test_size=0.6, fix_modal=True):
    """
    划分训练集、测试集和验证集

    :param test_size: 测试集和验证集占总数据的比例
    :param fix_modal: 为True比例固定为4:3:3；False按设置比率划分
    :return:
    """
    if fix_modal:
        data_len = len(data)
        train_data, train_label = data[:int(data_len * 0.6)], label[:int(data_len * 0.6)]
        val_data, val_label = data[int(data_len * 0.6):int(data_len)], label[int(data_len * 0.6):int(data_len)]
        return train_data, val_data, train_label, val_label

    train_data, val_data, train_label, val_label = train_test_split(data, label, test_size=test_size, random_state=configs.seed)
    return train_data, val_data, train_label, val_label


def load_pretrained_swin_transformer(model,checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            print(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            print(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            print("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            print(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)

    del checkpoint
    torch.cuda.empty_cache()




    