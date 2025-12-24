# import os
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import  torch
# import torchvision.transforms as transforms
# # 定义文件夹路径
# folder_path = 'BraTS2020_TrainingData//t1/'
#
# # 获取文件夹中的所有文件名
# file_names = os.listdir(folder_path)
#
# # 创建一个空的数据集
# data = []
#
# # 遍历所有文件
# for file_name in file_names:
#     # 构建图像文件名
#     filename = os.path.join(folder_path, file_name)
#
#     # 读取图像
#     image = Image.open(filename)
#
#
#     # 将图像调整为相同的尺寸
#     image = image.resize((256, 256))  # 你可以根据需要调整尺寸
#
#     # 将图像转换为数组
#     image_array = np.array(image)
#
#     # 将图像添加到数据集中
#     data.append(image_array)
#
# # 将数据集转换为 NumPy 数组
# data = np.array(data)
#
# # 打印数据集的形状
# # print(data.shape)
# # for k in data:
# #     print(k[120][120])
# # num_images_to_display = 5  # 你可以调整这个数字来显示更多或更少的图像
# # for i in range(num_images_to_display):
# #     plt.imshow(data[i], cmap='gray')  # 假设图像是灰度图像
# #     plt.axis('off')  # 不显示坐标轴
# #     plt.show()
# # 将数据集转换为 NumPy 数组
# data = np.array(data)
#
# # 打印数据集的形状
# # print(data.shape)
#
# # 将数据集转换为 Tensor
# data_tensor = torch.from_numpy(data)
# data_tensor = data_tensor.float()
# # 打印 Tensor 的形状
# # print(data_tensor.shape)
#
# # 遍历 Tensor 并打印每个像素的值
# # for k in data_tensor:
# #     print(k[120][120])
#
#
# transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#
# # 归一化数据
# # data_tensor = transform(data_tensor)
# # 256*256*4 3*256*256
# for  k  in data_tensor:
#     k=transform(k)
#     print(k[120][120])
#
#
# # # 打印归一化后的 Tensor 的形状
# # print(data_tensor.shape)
# #
# # # 遍历 Tensor 并打印每个像素的值
# # for k in data_tensor:
# #     print(k[120][120])
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

def dataloader(path, list, views):
    path = path
    list = list
    views = views
    data = []
    data_total = []
    # 定义文件夹路径
    for i in range(views):
        folder_path = path + list[i] + '//'
        # 获取文件夹中的所有文件名
        file_names = os.listdir(folder_path)
        # 遍历所有文件
        for file_name in file_names:
            # 构建图像文件名
            filename = os.path.join(folder_path, file_name)

            # 读取图像
            image = Image.open(filename)

            # 将图像调整为相同的尺寸
            image = image.resize((256, 256))  # 你可以根据需要调整尺寸

            # 将图像转换为 Tensor
            image_tensor = transforms.functional.to_tensor(image)

            # 将图像添加到数据集中
            data.append(image_tensor)
            data_tensor = torch.stack(data)
        data_total.append(data_tensor)

    # 将数据集转换为 Tensor


    # # 打印 Tensor 的形状
    # print(data_tensor.shape)

    # 定义归一化函数
    # transform = transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.226])

    # 归一化数据
    # data_tensor = transform(data_tensor)

    # 打印归一化后的 Tensor 的形状
    # print(data_tensor.shape)

    # # 遍历 Tensor 并打印每个像素的值
    # for k in data_tensor:
    #     print(k[1])
    return data_total

