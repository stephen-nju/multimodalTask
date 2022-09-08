import torch
import numpy as np
from mmaction.apis import init_recognizer, inference_recognizer

config_file = './task/c3d/config/c3d_sports1m_video_feature_extraction.py'
# 从模型库中下载检测点，并把它放到 `checkpoints/` 文件夹下
checkpoint_file = 'D:/Users/Downloads/c3d_sports1m_16x1x1_45e_ucf101_rgb_20201021-26655025.pth'

# 指定设备
device = 'cpu'  # or 'cpu'
device = torch.device(device)
# 根据配置文件和检查点来建立模型
model = init_recognizer(config_file, checkpoint_file, device=device)

# 测试单个视频并显示其结果
# video = './task/c3d/v_PlayingGuitar_g03_c05'
video='./task/c3d/v_PlayingGuitar_g03_c05.avi'
results, returned_features = inference_recognizer(model, video, outputs="backbone")

print(results)
# feat = returned_features.numpy()
