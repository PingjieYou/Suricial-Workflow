# -*- coding: utf-8 -*-
# @Author  : You.P.J
# @Time    : 2023/3/28 14:45
# @Function: 从视频中按照25fps或1fps提取帧

import os.path
import configs


def extractor(video):
    """
    利用ffmpeg从视频中抽取图像

    :param video: 视频路径
    """
    videoNumber = video[5:7]
    os.chdir(configs.Cholec80Frames)  # 切换到frames
    os.system("mkdir " + videoNumber)
    os.chdir(configs.Cholec80Videos)
    com_str = 'ffmpeg -i ' + video + ''' -vf "select='not(mod(n\,25))',setpts=N/TB" -vsync 0 -q:v 1 -start_number 0 ''' + configs.Cholec80Frames + "./" + videoNumber + "/%08d.png"
    os.system(com_str)


video_files = sorted([f for f in os.listdir(configs.Cholec80Videos) if ".mp4" in f])  # mp4文件集合

for video in video_files:
    extractor(video)
    ## 删除最后一帧，因为最后一帧在tool范围外，总体视频多了一帧
    frame_path = configs.Cholec80Frames + "/" + video[5:7]
    frame_list = os.listdir(frame_path)
    os.remove(frame_path + "/" + frame_list[-1])
