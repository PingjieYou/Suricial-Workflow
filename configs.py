# -*- coding: utf-8 -*-
# @Author  : You.P.J
# @Time    : 2023/3/28 14:57
# @Function: 常用配置

## 数据集路径
Cholec80 = "/home/payton/Dataset/Cholec80"  # 数据集根路径
Cholec80Videos = Cholec80 + "/videos"  # 数据集视频路径
Cholec80Frames = Cholec80 + "/frames"  # 数据集帧路径
Cholec80ProcessedFrames = Cholec80 + "/processed_frames"  # 预处理后的帧路径
Cholec80ToolAnnotations = Cholec80 + "/tool_annotations"  # 器具标签路径
Cholec80PhaseAnnotations = Cholec80 + "/phase_annotations"  # 阶段标签路径
Cholec80PhaseAnticipations = Cholec80 + "/phase_anticipation_annotations"  # 阶段预测标签路径

## 数据处理模块
FPS = 25  # 视频取帧率
WIDTH = 250  # 图像宽度
HEIGHT = 250  # 图像高度

## 总数据、训练集数据和测试集数据
NumPicklePath = "./pkl/num.pkl"  # 数量
ImagePicklePath = "./pkl/image.pkl"  # image路径数据
ToolPicklePath = "./pkl/tool.pkl"  # tool数据
PhasePicklePath = "./pkl/phase.pkl"  # phase数据
AnticipationPicklePath = "./pkl/anticipation.pkl"  # anticipation数据

## 训练配置
seed = 3407
clip_size = 8  # 视频帧长度
batch_size = 16  # 训练批量
epochs = 25  # 训练轮数
learning_rate = 1e-4  # 学习率

## 优化器配置
momentum = 0.9  # 动量
weigth_decay = 0  # 权重衰减
dampenning = 0  # 震荡性
nesterov = False  # 加速梯度
sgd_adjust = 1  # SGD学习率调整方法
sgd_step = 5  # SGD更新步数
sgd_gamma = 0.1
alpha = 1.0

## 模型保存路径
model_path = "./model/"  # 模型保存路径
