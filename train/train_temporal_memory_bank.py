#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : train_temporal_memory_bank.py
# @Time     : 2023/4/22 下午2:16
# @Author   : YouPingJie
# @Function : 训练Temporal Memory Bank模型，用于提取时序特征，为TMRNet提供外部特征

import copy
import time
import torch
import utils
import torchvision
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from models.svrcnet import SVRCNet
from torch.utils.data import DataLoader
from datasets.sampler import SeqSampler
from datasets.dataset import Cholec80Dataset
from torch.utils.tensorboard import SummaryWriter


def train(opts):
    writer = SummaryWriter('./runs/lr5e-4_do/')

    # 训练设备
    num_gpu = torch.cuda.device_count()
    use_gpu = num_gpu > 0
    if use_gpu is not False:
        print("Using {} GPUs".format(num_gpu))

    # 训练、测试和验证数据
    img_list, combined_label_list, num_list = utils.get_date_label_num()
    train_img_list, val_img_list, train_label, val_label = utils.train_val_split(img_list, combined_label_list, num_list)
    train_phase_list, val_phase_list = train_label[:, -1], val_label[:, -1]

    train_start_idx = utils.get_clip_start_idx(opts.clip_size, num_list[:50])
    val_statr_idx = utils.get_clip_start_idx(opts.clip_size, num_list[50:])
    train_idx = utils.get_clip_idx(train_start_idx, opts.clip_size)
    val_idx = utils.get_clip_idx(val_statr_idx, opts.clip_size)
    train_num = len(train_idx)
    val_num = len(val_idx)

    print("train_num: {}, val_num: {}".format(train_num, val_num))

    # 数据增广
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
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

    # 数据集
    train_dataset = Cholec80Dataset(
        train_img_list, train_phase_list, train_phase_list, train_transforms)
    val_dataset = Cholec80Dataset(
        val_img_list, val_phase_list, val_phase_list, val_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size, sampler=SeqSampler(train_dataset, train_idx), num_workers=8, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=opts.batch_size, sampler=SeqSampler(val_dataset, val_idx), num_workers=8, pin_memory=False)


    # 模型
    model = SVRCNet(opts.clip_size)
    model = torch.nn.DataParallel(model)
    model = model.cuda() if use_gpu else model
    model = torch.compile(model)

    # 损失函数
    phase_criterion = torch.nn.CrossEntropyLoss(size_average=False)

    # 优化器
    optimizer = torch.optim.SGD([
        {'params': model.module.share.parameters()},
        {'params': model.module.lstm.parameters(), 'lr': opts.lr},
        {'params': model.module.fc.parameters(), 'lr': opts.lr},
    ], lr=opts.lr / 10, momentum=opts.momentum, weight_decay=opts.weight_decay, dampening=opts.dampening, nesterov=opts.nesterov)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.sgd_step, gamma=opts.sgd_gamma)

    # 训练
    best_model_weights = copy.deepcopy(model.module.state_dict())
    best_val_accuracy_phase = 0.0
    correspond_train_accuracy_phase = 0.0
    best_epoch = 0

    for epoch in range(opts.epochs):
        torch.cuda.empty_cache()
        np.random.shuffle(train_start_idx)  # 打乱起始数据的索引
        train_idx_80 = utils.get_clip_idx(train_start_idx, opts.clip_size)
        train_dataloader_80 = DataLoader(train_dataset, batch_size=opts.batch_size, sampler=SeqSampler(train_dataset, train_idx_80), num_workers=8, pin_memory=False)

        model.train()
        train_loss_phase = 0.0
        train_corrects_pahse = 0
        batch_progress = 0.0
        running_loss_phase = 0.0
        minibatch_correct_phase = 0.0
        train_start_time = time.time()

        with tqdm(total=len(train_dataloader),dynamic_ncols=True) as pbar:
            for (i, data) in enumerate(train_dataloader_80):
                optimizer.zero_grad()

                imgs, phases, _ = data
                imgs = imgs.cuda() if use_gpu else imgs
                imgs = imgs.view(-1, opts.clip_size, 3, 224, 224)
                phases = phases.cuda() if use_gpu else phases
                phases = phases[(opts.clip_size - 1)::opts.clip_size]  # 要预测的目标phase

                phase_logits = model(imgs)
                phase_logits = phase_logits[(opts.clip_size - 1)::opts.clip_size]
                phase_loss = phase_criterion(phase_logits, phases)
                phase_loss.backward()
                optimizer.step()

                _, phase_pred = torch.max(phase_logits.data, 1)  # 预测phase的下标

                running_loss_phase += phase_loss.data.item()
                train_loss_phase += phase_loss.data.item()

                batch_correct_phase = torch.sum(phase_pred == phases.data)
                train_corrects_pahse += batch_correct_phase
                minibatch_correct_phase += batch_correct_phase

                if i % 100 == 99:
                    batch_iters = epoch * train_num + i * opts.batch_size / opts.clip_size  # 已经训练的batch数
                    writer.add_scalar('training phase loss', running_loss_phase / (opts.batch_size * 100 / opts.clip_size), batch_iters)
                    writer.add_scalar('training phase accuracy', float(minibatch_correct_phase) / (opts.batch_size * 100 / opts.clip_size), batch_iters)

                    running_loss_phase = 0.0
                    minibatch_correct_phase = 0.0

                if (i + 1) * opts.batch_size >= train_num:
                    running_loss_phase = 0.0
                    minibatch_correct_phase = 0.0

                batch_progress += 1
                if batch_progress * opts.batch_size >= train_num:
                    percent = 100.0
                    print('Batch progress: %s [%d/%d]' % (str(percent) + '%', train_num, train_num), end='\n')
                else:
                    percent = round(batch_progress * opts.batch_size / train_num * 100, 2)
                    print('Batch progress: %s [%d/%d]' % (str(percent) + '%', batch_progress * opts.batch_size, train_num), end='\r')

                pbar.set_description(f'Train Epoch [{epoch}/{opts.epochs}]')
                pbar.set_postfix(loss=phase_loss.item())

        train_duration = time.time() - train_start_time
        train_phase_accuracy = float(train_corrects_pahse) / float(train_num) * opts.clip_size
        train_phase_average_loss = train_loss_phase / float(train_num) * opts.clip_size

        # 模型测试
        model.eval()
        val_loss_phase = 0.0
        val_corrects_phase = 0
        val_start_time = time.time()
        val_process= 0
        val_all_pred_phase = []
        val_all_label_phase = []

        with torch.no_grad():
            for data in val_dataloader:
                imgs, phases, _ = data
                imgs = imgs.cuda() if use_gpu else imgs
                imgs = imgs.view(-1, opts.clip_size, 3, 224, 224)
                phases = phases.cuda() if use_gpu else phases
                phases = phases[(opts.clip_size - 1)::opts.clip_size]

                phase_logits = model(imgs)
                phase_logits = phase_logits[(opts.clip_size - 1)::opts.clip_size]
                phase_loss = phase_criterion(phase_logits, phases)

                _, phase_pred = torch.max(phase_logits.data, 1)

                val_loss_phase += phase_loss.data.item()
                val_corrects_phase += torch.sum(phase_pred == phases.data)

                for i in range(len(phase_pred)):
                    val_all_pred_phase.append(phase_pred[i].item())
                for i in range(len(phases)):
                    val_all_label_phase.append(phases[i].item())

                val_process += 1
                if val_process * opts.batch_size >= val_num:
                    percent = 100.0
                    print('Batch progress: %s [%d/%d]' % (str(percent) + '%', val_num, val_num), end='\n')
                else:
                    percent = round(val_process * opts.batch_size / val_num * 100, 2)
                    print('Batch progress: %s [%d/%d]' % (str(percent) + '%', val_process * opts.batch_size, val_num), end='\r')

        val_duration = time.time() - val_start_time
        val_phase_accuracy = float(val_corrects_phase) / float(val_num) * opts.clip_size
        val_phase_average_loss = val_loss_phase / float(val_num) * opts.clip_size
        val_recall_phase = metrics.recall_score(val_all_label_phase, val_all_pred_phase, average='macro')
        val_precision_phase = metrics.precision_score(val_all_label_phase, val_all_pred_phase, average='macro')
        val_jaccard_phase = metrics.jaccard_similarity_score(val_all_label_phase, val_all_pred_phase)
        val_precision_each_phase = metrics.precision_score(val_all_label_phase, val_all_pred_phase, average=None)
        val_recall_each_phase = metrics.recall_score(val_all_label_phase, val_all_pred_phase, average=None)

        writer.add_scalar('validation acc epoch phase',
                          float(val_phase_accuracy), epoch)
        writer.add_scalar('validation loss epoch phase',
                          float(val_phase_average_loss), epoch)

        print('epoch: {:4d}'
              ' train in: {:2.0f}m{:2.0f}s'
              ' train loss(phase): {:4.4f}'
              ' train accu(phase): {:.4f}'
              ' valid in: {:2.0f}m{:2.0f}s'
              ' valid loss(phase): {:4.4f}'
              ' valid accu(phase): {:.4f}'
              .format(epoch,
                      train_duration // 60,
                      train_duration % 60,
                      train_phase_average_loss,
                      train_phase_accuracy,
                      val_duration // 60,
                      val_duration % 60,
                      val_phase_average_loss,
                      val_phase_accuracy))

        print("val_precision_each_phase:", val_precision_each_phase)
        print("val_recall_each_phase:", val_recall_each_phase)
        print("val_precision_phase", val_precision_phase)
        print("val_recall_phase", val_recall_phase)
        print("val_jaccard_phase", val_jaccard_phase)

        scheduler.step()

        if val_phase_accuracy > best_val_accuracy_phase:
            best_val_accuracy_phase = val_phase_accuracy
            correspond_train_acc_phase = train_phase_accuracy
            best_model_wts = copy.deepcopy(model.module.state_dict())
            best_epoch = epoch
        if val_phase_accuracy == best_val_accuracy_phase:
            if train_phase_accuracy > correspond_train_acc_phase:
                correspond_train_acc_phase = train_phase_accuracy
                best_model_wts = copy.deepcopy(model.module.state_dict())
                best_epoch = epoch

        save_val_phase = int("{:4.0f}".format(best_val_accuracy_phase * 10000))
        save_train_phase = int("{:4.0f}".format(correspond_train_acc_phase * 10000))
        base_name = "lstm" \
                    + "_epoch_" + str(best_epoch) \
                    + "_length_" + str(opts.clip_size) \
                    + "_opt_" + str(opts.optimizer) \
                    + "_batch_" + str(opts.batch_size) \
                    + "_train_" + str(save_train_phase) \
                    + "_val_" + str(save_val_phase)

        torch.save(best_model_wts, "./best_model/lr5e-4_do/" + base_name + ".pth")
        print("best_epoch", str(best_epoch))

        torch.save(model.module.state_dict(), "./temp/lr5e-4_do/latest_model_" + str(epoch) + ".pth")

