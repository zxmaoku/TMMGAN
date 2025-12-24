import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.io import savemat
# 设置文件夹路径
modal1_path = 'BraTS2020_TrainingData/t1/'
modal2_path = 'BraTS2020_TrainingData/t2/'
modal3_path = 'BraTS2020_TrainingData/flair/'

# 获取文件夹中的所有图像文件
modal1_files = [f for f in os.listdir(modal1_path) if f.endswith('.png')]
modal2_files = [f for f in os.listdir(modal2_path) if f.endswith('.png')]
modal3_files = [f for f in os.listdir(modal3_path) if f.endswith('.png')]
# print(len(modal1_files), len(modal2_files), len(modal3_files))
# 确保所有文件夹中的文件数量相同
assert len(modal1_files) == len(modal2_files) == len(modal3_files), "所有文件夹中的文件数量必须相同"

# 创建一个列表来存储图像数据
image_data = []

# 读取图像数据
for i in range(len(modal1_files)):
    # 读取modal1的图像
    modal1_image = Image.open(os.path.join(modal1_path, modal1_files[i]))
    modal1_tensor = torch.from_numpy(np.array(modal1_image)).float().unsqueeze(0)  # 添加一个批次维度

    # 读取modal2的图像
    modal2_image = Image.open(os.path.join(modal2_path, modal2_files[i]))
    modal2_tensor = torch.from_numpy(np.array(modal2_image)).float().unsqueeze(0)  # 添加一个批次维度

    # 读取modal3的图像
    modal3_image = Image.open(os.path.join(modal3_path, modal3_files[i]))
    modal3_tensor = torch.from_numpy(np.array(modal3_image)).float().unsqueeze(0)  # 添加一个批次维度

    # 将图像数据添加到列表中
    image_data.append({'t1': modal1_tensor, 't2': modal2_tensor, 'flair': modal3_tensor})
# print(image_data[0])

# 将图像数据保存到MAT文件
# savemat('imageData.mat', {'imageData': image_data})


# 定义一个自定义的数据集类
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]