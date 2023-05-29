import torch

# 导入 PyTorch 中的一些模块
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 定义一些常量
cnn_normalization_mean = [0.485, 0.456, 0.406]
cnn_normalization_std = [0.229, 0.224, 0.225]
tensor_normalizer = transforms.Normalize(mean=cnn_normalization_mean, std=cnn_normalization_std)
epsilon = 1e-5


def preprocess_image(image, target_width=None):
    """输入 PIL.Image 对象，输出标准化后的四维 tensor"""
    if target_width:
        t = transforms.Compose([
            transforms.Resize(target_width),
            transforms.CenterCrop(target_width),
            # 将图像在中心处裁剪到指定大小。在这种情况下，target_width 参数用于指定正方形裁剪的大小。如果未指定 target_width，则图像不会被裁剪，而是被调整为默认大小。
            transforms.ToTensor(),
            tensor_normalizer,
        ])
    else:
        t = transforms.Compose([
            transforms.ToTensor(),
            tensor_normalizer,
        ])
    return t(image).unsqueeze(0)


def image_to_tensor(image, target_width=None):
    """输入 OpenCV 图像，范围 0~255，BGR 顺序，输出标准化后的四维 tensor"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return preprocess_image(image, target_width)


def read_image(path, target_width=None):
    """输入图像路径，输出标准化后的四维 tensor"""
    image = Image.open(path)
    return preprocess_image(image, target_width)


def recover_image(tensor):
    """输入 GPU 上的四维 tensor，输出 0~255 范围的三维 numpy 矩阵，RGB 顺序"""
    """recover_image 函数的作用是将 GPU 上的四维 tensor 转换为 0~255 范围的三维 numpy 矩阵，RGB 顺序。
    具体实现是先将 tensor 转换为 numpy 数组，然后进行标准化还原，最后将数组转换为图片格式。
    其中标准化还原的过程是将 tensor 中的每个像素点乘以标准化的标准差并加上均值，以还原原始像素值。"""
    image = tensor.detach().cpu().numpy()
    image = image * np.array(cnn_normalization_std).reshape((1, 3, 1, 1)) + \
            np.array(cnn_normalization_mean).reshape((1, 3, 1, 1))
    return (image.transpose(0, 2, 3, 1) * 255.).clip(0, 255).astype(np.uint8)[0]


def recover_tensor(tensor):
    """recover_tensor 函数的作用是将经过标准化后的 tensor 还原为原始的 tensor。
    具体实现是将 tensor 中的每个像素点乘以标准化的标准差并加上均值，以还原原始像素值。最后，将还原后的 tensor 的像素值限制在 0 到 1 之间。
    还原原始的 tensor 的原因是为了在经过神经网络处理后获得原始图像。
    神经网络通过标准化图像进行处理，这涉及到减去均值并除以标准差。这样做是为了确保神经网络的输入具有与训练数据类似的分布。
    但是，为了获得原始图像，我们需要通过将每个像素乘以标准差并加上均值来反转此过程。这正是 recover_tensor 函数所做的。
    它接收一个已经标准化的 tensor，并返回一个已经恢复为其原始像素值的 tensor。恢复后的 tensor 然后用于使用 imshow 函数显示原始图像。
    clamp 函数用于确保像素值在 0 到 1 的范围内。"""
    m = torch.tensor(cnn_normalization_mean).view(1, 3, 1, 1).to(tensor.device)
    s = torch.tensor(cnn_normalization_std).view(1, 3, 1, 1).to(tensor.device)
    tensor = tensor * s + m
    return tensor.clamp(0, 1)


def imshow(tensor, title=None):
    """输入 GPU 上的四维 tensor，然后绘制该图像"""
    image = recover_image(tensor)
    # print(image.shape)
    plt.imshow(image)
    if title is not None:
        plt.title(title)


def mean_std(features):
    """
    输入 VGG16 计算的四个特征，输出每张特征图的均值和标准差，长度为1920
    mean_std 函数的作用是计算 VGG16 计算的四个特征的均值和标准差，长度为 1920。
    具体实现是将每个特征图的像素点展平，然后计算每个特征图的均值和标准差，最后将所有特征图的均值和标准差拼接在一起。
    """
    mean_std_features = []
    for x in features:
        x = x.view(*x.shape[:2], -1)
        x = torch.cat([x.mean(-1), torch.sqrt(x.var(-1) + epsilon)], dim=-1)
        n = x.shape[0]
        x2 = x.view(n, 2, -1).transpose(2, 1).contiguous().view(n, -1)  # 【mean, ..., std, ...] to [mean, std, ...]
        mean_std_features.append(x2)
    mean_std_features = torch.cat(mean_std_features, dim=-1)
    return mean_std_features


class Smooth:
    # 对输入的数据进行滑动平均
    """这个类是一个滑动平均器，用于对输入的数据进行滑动平均。它的构造函数接受一个窗口大小参数，用于指定滑动窗口的大小。在每次调用 __iadd__ 方法时，它会将输入的数据加入到滑动窗口中，并返回滑动窗口的平均值。这个类的实现非常简单，它使用了一个循环数组来存储滑动窗口中的数据，并使用了 numpy 的 mean 方法来计算平均值。如果你需要对输入的数据进行平滑处理，这个类可能会很有用。
    """

    def __init__(self, windowsize=100):
        self.window_size = windowsize
        self.data = np.zeros((self.window_size, 1), dtype=np.float32)
        self.index = 0

    def __iadd__(self, x):
        if self.index == 0:
            self.data[:] = x
        self.data[self.index % self.window_size] = x
        self.index += 1
        return self

    def __float__(self):
        return float(self.data.mean())

    def __format__(self, f):
        return self.__float__().__format__(f)

