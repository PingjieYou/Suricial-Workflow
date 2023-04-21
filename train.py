# -*- coding: utf-8 -*-
# @Author  : You.P.J
# @Time    : 2023/3/31 13:34
# @Function: 训练模型

import os
import time
import copy

from tqdm import tqdm

import utils
import torch
import configs
import argparse
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.nn import DataParallel
from collections import OrderedDict
from torch.utils.data import DataLoader
from datasets.dataset import Cholec80Dataset
from models.endotnet import EndotNet
from models.mtrcnn import MultiTaskRCNN
from models.swin_transformer import SwinTransformer, build_swin_transformer
from models.video_swin_transfromer import SwinTransformer3D, build_video_swin_transformer


def main():
    parser = argparse.ArgumentParser(description="model training")
    parser.add_argument('--clip_size', default=configs.clip_size, type=int, help='sequence length, default 4')
    parser.add_argument('--batch_size', default=configs.batch_size, type=int, help='batch size, default 8')
    parser.add_argument('--epochs', default=configs.epochs, type=int, help='epochs to train and val, default 25')
    parser.add_argument('--lr', default=configs.learning_rate, type=float, help='learning rate for optimizer, default 1e-3')
    parser.add_argument('--model', default="mtrcnn", help="model to train, default mtrcnn")
    parser.add_argument('--optimizer', default="adam", help="optimizer for training, default adam")
    parser.add_argument('--momentum', default=configs.momentum, type=float, help='momentum for sgd, default 0.9')
    parser.add_argument('--weight_decay', default=configs.weigth_decay, type=float, help='weight decay for sgd, default 0')
    parser.add_argument('--dampening', default=configs.dampenning, type=float, help='dampening for sgd, default 0')
    parser.add_argument('--nesterov', default=configs.nesterov, type=bool, help='nesterov momentum, default False')
    parser.add_argument('--sgd_adjust', default=configs.sgd_adjust, type=int, help='sgd method adjust lr 0 for step 1 for min, default 1')
    parser.add_argument('--sgd_step', default=configs.sgd_step, type=int, help='number of steps to adjust lr for sgd, default 5')
    parser.add_argument('--sgd_gamma', default=configs.sgd_gamma, type=float, help='gamma of steps to adjust lr for sgd, default 0.1')
    parser.add_argument('--alpha', default=configs.alpha, type=float, help='kl loss ratio, default 1.0')

    # 命令行参数
    opts = parser.parse_args()
    print(opts)

    # 训练设备
    num_gpu = torch.cuda.device_count()
    use_gpu = torch.cuda.is_available()
    if use_gpu is not False:
        print("The number of gpu we use is {}.".format(num_gpu))

    # 训练、测试和验证数据
    img_list, combined_label_list = utils.get_date_label()
    train_img_list, test_img_list, val_img_list, train_label, test_label, val_label = utils.train_test_valid_split(
        img_list, combined_label_list, fix_modal=False)
    train_tool_list, test_tool_list, val_tool_list, train_phase_list, test_phase_list, val_phase_list = train_label[:, :-1], test_label[:, :-1], \
        val_label[:, :-1], train_label[:, -
                                       1], test_label[:, -1], val_label[:, -1]

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            [0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
    ])
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            [0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
    ])
    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            [0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
    ])

    train_dataset = Cholec80Dataset(
        train_img_list, train_tool_list, train_phase_list, train_transforms)
    test_dataset = Cholec80Dataset(
        test_img_list, test_tool_list, test_phase_list, test_transforms)
    val_dataset = Cholec80Dataset(
        val_img_list, val_tool_list, val_phase_list, val_transforms)

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=opts.batch_size, num_workers=0, pin_memory=False, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=opts.batch_size, num_workers=0, pin_memory=False, shuffle=False, drop_last=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=opts.batch_size,
                                num_workers=0, pin_memory=False, shuffle=False, drop_last=True)

    # 模型
    video_swim_transformer = build_video_swin_transformer()
    swin_transformer = build_swin_transformer()
    endotnet = EndotNet().cuda()
    mtrcnn = MultiTaskRCNN(opts.clip_size)
    model = mtrcnn.cuda() if use_gpu else mtrcnn
    model = DataParallel(model)

    # 损失函数
    tools_criterion = nn.BCEWithLogitsLoss(size_average=False).cuda(
    ) if use_gpu else nn.BCEWithLogitsLoss(size_average=False)
    phases_criterion = nn.CrossEntropyLoss(size_average=False).cuda(
    ) if use_gpu else nn.CrossEntropyLoss(size_average=False)
    criterions = [tools_criterion, phases_criterion]

    # 优化器
    if opts.model == 'mtrcnn':
        if opts.optimizer == 'sgd':
            optimizer = torch.optim.SGD([
                {"params": model.module.share.parameters()},
                {"params": model.module.lstm.parameters(), "lr": opts.lr},
                {"params": model.module.fc.parameters(), "lr": opts.lr},
                {"params": model.module.fc2.parameters(), "lr": opts.lr}
            ], lr=opts.lr / 10, momentum=opts.momentum, dampening=opts.dampening, weight_decay=opts.weight_decay, nesterov=opts.nesterov)
        elif opts.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.sgd_step, gamma=opts.sgd_gamma)

    elif opts.model == 'endotnet':
        if opts.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.module.parameters(), lr=opts.lr / 10, momentum=opts.momentum,
                                        dampening=opts.dampening, weight_decay=opts.weight_decay, nesterov=opts.nesterov)
        elif opts.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.module.parameters(), lr=opts.lr)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.sgd_step, gamma=opts.sgd_gamma)

    # 训练精度等记录
    best_model_weight = copy.deepcopy(model.state_dict())
    best_tool_val_accuracy = 0.0
    best_phase_val_accuracy = 0.0
    correspond_tool_train_accuracy = 0.0
    correspond_phase_train_accuracy = 0.0

    # 训练模型
    for epoch in range(opts.epochs):
        train_duration_time,train_tool_accuracy,train_phase_accuracy,train_tool_average_loss,train_phase_average_loss = utils.train_one_epoch(opts, model, criterions, optimizer, train_dataloader, "gpu" if use_gpu else "cpu", epoch+1)
        val_duration_time,val_tool_accuracy,val_phase_accuracy = utils.validate_one_epoch(opts, model, val_dataloader, "gpu" if use_gpu else "cpu", "mtrcnn", epoch + 1)

        scheduler.step()

        # 保存最好的模型
        if val_phase_accuracy > best_phases_val_accuracy and val_tool_accuracy > 0.95:
            best_phases_val_accuracy = val_phase_accuracy
            best_tools_val_accuracy = val_tool_accuracy
            correspond_tool_train_accuracy = train_tool_accuracy
            correspond_train_accuracy = train_phase_accuracy
            best_model_weight = copy.deepcopy(model.state_dict())
        elif val_phase_accuracy == best_phases_val_accuracy and val_tool_accuracy > 0.95:
            if val_tool_accuracy > best_tools_val_accuracy:
                correspond_tool_train_accuracy = train_tool_accuracy
                correspond_phase_train_accuracy = train_phase_accuracy
                best_model_weight = copy.deepcopy(model.state_dict())
            elif val_tool_accuracy == best_tool_val_accuracy:
                if train_phase_accuracy > correspond_phase_train_accuracy:
                    correspond_phase_train_accuracy = train_phase_accuracy
                    correspond_tool_train_accuracy = train_tool_accuracy
                    best_model_wts = copy.deepcopy(model.state_dict())
                elif train_phase_accuracy == correspond_phase_train_accuracy:
                    if train_tool_accuracy > best_tool_val_accuracy:
                        correspond_tool_train_accuracy = train_tool_accuracy
                        best_model_wts = copy.deepcopy(model.state_dict())
        
        print('best tool accuracy: {:.4f} correspond tool train accuracy: {:.4f}'.format(best_tool_val_accuracy,correspond_tool_train_accuracy))
        print('best phase accuracy: {:.4f} correspond phase train accuracy: {:.4f}'.format(best_phases_val_accuracy,correspond_phase_train_accuracy))
        save_tool_val = int("{:4.0f}".format(best_tool_val_accuracy * 10000))
        save_phase_val = int("{:4.0f}".format(best_phases_val_accuracy * 10000))
        save_tool_train = int("{:4.0f}".format(correspond_tool_train_accuracy * 10000))
        save_phase_train = int("{:4.0f}".format(correspond_phase_train_accuracy * 10000))
        public_name = opts.model \
                    + "_epoch_" + str(opts.epochs) \
                    + "_length_" + str(opts.clip_size) \
                    + "_opt_" + str(opts.optimizer) \
                    + "_batch_" + str(opts.batch_size) \
                    + "_tool_train_" + str(correspond_tool_train_accuracy) \
                    + "_phase_train_" + str(correspond_phase_train_accuracy) \
                    + "_tool_val_" + str(best_tool_val_accuracy) \
                    + "_phase_val_" + str(best_phases_val_accuracy)
        model_name = configs.model_path + public_name + ".pth"
        torch.save(best_model_wts, model_name)


if __name__ == '__main__':
    main()
