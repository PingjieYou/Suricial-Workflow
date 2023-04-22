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


def get_date_label_num():
    """
    获取数据，将tool和phase合并和每个视频的帧数

    :return:
    """
    with open(configs.ImagePicklePath, "rb") as f:
        img_list = pickle.load(f)

    with open(configs.ToolPicklePath, "rb") as f:
        tool_list = pickle.load(f)

    with open(configs.PhasePicklePath, "rb") as f:
        phase_list = pickle.load(f)

    with open(configs.NumPicklePath, "rb") as f:
        num_list = pickle.load(f)

    tool_list = np.asarray(tool_list, dtype=np.int64)
    phase_list = np.asarray(phase_list, dtype=np.int64).reshape(-1, 1)
    combined_label_list = np.hstack((tool_list, phase_list))  # 0～6表示tool，7表示phase

    return img_list, combined_label_list, num_list


def get_clip_start_idx(clip_size, num_list):
    """
    获取clip的起始索引

    :param clip_size: clip大小
    :param num_frames: 视频帧数
    :return:
    """
    idx = 0
    clip_idx = []

    for i in range(len(num_list)):
        for j in range(idx, idx + (num_list[i] + 1 - clip_size)):
            clip_idx.append(j)
        idx += num_list[i]

    return clip_idx


def get_clip_idx(start_idx_list, clip_size):
    """
    获取视频片段数据

    @param start_idx_list: 视频片段起始索引
    @param clip_size: 视频片段大小
    @return:
    """
    idx = []

    for i in range(len(start_idx_list)):
        for j in range(clip_size):
            idx.append(start_idx_list[i] + j)

    return idx


def train_test_valid_split(data, label, num_list):
    """
    划分训练集、测试集和验证集

    @param data: 所有图片
    @param label: 所有标签
    @param num_list: 每个视频的帧数
    @return:
    """
    train_list_num = 40
    test_list_num = 8
    val_list_num = 32

    train_num = sum(num_list[:train_list_num])
    test_num = sum(num_list[train_list_num:train_list_num + test_list_num])
    val_num = sum(num_list[train_list_num + test_list_num:])

    train_data, train_label = data[:train_num], label[:train_num]
    test_data, test_label = data[train_num:train_num + test_num], label[train_num:train_num + test_num]
    val_data, val_label = data[train_num + test_num:], label[train_num + test_num:]

    return train_data, test_data, val_data, train_label, test_label, val_label


def train_val_split(data, label, num_list):
    """
    划分训练集和验证集

    @param data: 所有图片
    @param label: 所有标签
    @param num_list: 每个视频的帧数
    @return:
    """
    train_list_num = 50
    val_list_num = 30

    train_num = sum(num_list[:train_list_num])
    val_num = sum(num_list[train_list_num:])

    train_data, train_label = data[:train_num], label[:train_num]
    val_data, val_label = data[train_num:], label[train_num:]

    return train_data, val_data, train_label, val_label


def load_pretrained_swin_transformer(model, checkpoint_path):
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


def train_one_epoch(opts, model, criterions, optimizer, data_loader, device, epoch):
    """
    训练一轮模型

    :param opts: 参数集合
    :param model: 模型
    :param criterions: 损失函数集合
    :param optimizer: 优化器
    :param data_loader: 数据加载器
    :param device: 设备，"gpu"或者"cpu"
    :param epoch: 当前轮数
    :return:
    """
    model.train()

    tool_criterion, phase_criterion = criterions

    train_tool_loss = 0.0
    train_phase_loss = 0.0
    train_tool_correct_num = 0
    train_phase_correct_num = 0
    train_start_time = time.time()

    with tqdm(enumerate(data_loader), total=len(data_loader)) as pbar:
        for idx, (imgs, tools, phases) in pbar:
            imgs = torch.autograd.Variable(imgs.cuda()) if device == "gpu" else torch.autograd.Variable(imgs)
            tools = torch.autograd.Variable(tools.cuda()) if device == "gpu" else torch.autograd.Variable(tools)
            phases = torch.autograd.Variable(phases.cuda()) if device == "gpu" else torch.autograd.Variable(phases)

            optimizer.zero_grad()

            model(imgs)
            assert 0

            tool_logits, phase_logits = model(imgs)

            tool_loss = tool_criterion(tool_logits, tools.data.float())
            phase_loss = phase_criterion(phase_logits, phases)
            total_loss = tool_loss + phase_loss
            total_loss.backward()
            optimizer.step()

            tool_logits = F.sigmoid(tool_logits.data)
            tool_pred = (tool_logits.cpu() > 0.5).to(torch.long)
            phase_pred = torch.max(phase_logits.data, 1)[1]

            train_tool_correct_num += torch.sum(tool_pred.data == tools.data.cpu())
            train_phase_correct_num += torch.sum(phase_pred == phases.data)

            pbar.set_description(f'Train Epoch [{epoch}/{opts.epochs}]')
            pbar.set_postfix(loss=total_loss.item(),
                             tool_acc=train_tool_correct_num.item() / ((idx + 1) * opts.batch_size * 7),
                             phase_acc=train_phase_correct_num.item() / ((idx + 1) * opts.batch_size))

    train_duration_time = time.time() - train_start_time
    train_tool_accuracy = train_tool_correct_num / len(data_loader) / 7
    train_phase_accuracy = train_phase_correct_num / len(data_loader)
    train_tool_average_loss = train_tool_loss / len(data_loader) / 7
    train_phase_average_loss = train_phase_loss / len(data_loader)

    print(
        "train use time: {}, tool accuracy: {}, phase accuracy: {}, average tool loss: {}, average phases loss: {}".format(
            train_duration_time,
            train_tool_accuracy,
            train_phase_accuracy,
            train_tool_average_loss,
            train_phase_average_loss))

    return train_duration_time, train_tool_accuracy, train_phase_accuracy, train_tool_average_loss, train_phase_average_loss


@torch.no_grad()
def validate_one_epoch(opts, model, data_loader, device, epoch):
    """
    验证一轮模型

    :param opts: 参数集合
    :param model: 模型
    :param losses: 损失函数集合
    :param data_loader: 数据加载器
    :param device: 设备，"gpu"或者"cpu"
    :param epoch: 当前轮数
    :return:
    """
    model.eval()

    if opts.model == "mtrcnn" or opts.model == "endotnet":
        val_tool_correct_num = 0
        val_phase_correct_num = 0
        val_start_time = time.time()

        with tqdm(enumerate(data_loader), total=len(data_loader)) as pbar:
            for idx, (imgs, tools, phases) in enumerate(data_loader):
                imgs = torch.autograd.Variable(imgs.cuda()) if device == "gpu" else torch.autograd.Variable(imgs)
                tools = torch.autograd.Variable(tools.cuda()) if device == "gpu" else torch.autograd.Variable(tools)
                phases = torch.autograd.Variable(phases.cuda()) if device == "gpu" else torch.autograd.Variable(phases)

                tool_logits, phase_logits = model(imgs)

                tool_logits = F.sigmoid(tool_logits.data)
                tool_pred = (tool_logits.cpu() > 0.5).to(torch.long)
                phase_pred = torch.max(phase_logits.data, 1)[1]

                val_tool_correct_num += torch.sum(tool_pred.data == tools.data.cpu())
                val_phase_correct_num += torch.sum(phase_pred == phases.data)

                pbar.set_description(f'Validation Epoch [{epoch}/{opts.epochs}]')
                pbar.set_postfix(tool_acc=val_tool_correct_num.item() / ((idx + 1) * opts.batch_size * 7),
                                 phase_acc=val_phase_correct_num.item() / ((idx + 1) * opts.batch_size))

        val_duration_time = time.time() - val_start_time
        val_tool_accuracy = val_tool_correct_num / len(data_loader) / 7
        val_phase_accuracy = val_phase_correct_num / len(data_loader)

        print("validation use time: {}, tool accuracy: {}, phase accuracy: {}".format(val_duration_time,
                                                                                      val_tool_accuracy,
                                                                                      val_phase_accuracy))

        return val_duration_time, val_tool_accuracy, val_phase_accuracy
