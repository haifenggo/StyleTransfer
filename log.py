import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from utils import *
# from models import *

import numpy as np

from tqdm import tqdm
import random
import cv2

def ConvLayer(in_channels, out_channels, kernel_size=3, stride=1,
    upsample=None, instance_norm=True, relu=True):
    """ ConvLayer函数用于创建卷积神经网络的层列表。它接受输入通道数、输出通道数、卷积核大小、步长、上采样因子、实例归一化标志和ReLU激活标志作为参数。

该函数首先检查是否提供了上采样因子，如果是，则将上采样层添加到层列表中。然后添加一个反射填充层，其内核大小为kernel_size // 2，接着是一个具有指定输入和输出通道数、卷积核大小和步长的卷积层。如果实例归一化标志设置为True，则将实例归一化层添加到层列表中。最后，如果ReLU激活标志设置为True，则将ReLU激活层添加到层列表中。

该函数返回层列表，可以用于创建卷积神经网络。"""
    layers = []
    if upsample:
        layers.append(nn.Upsample(mode='nearest', scale_factor=upsample))
    layers.append(nn.ReflectionPad2d(kernel_size // 2))
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
    if instance_norm:
        layers.append(nn.InstanceNorm2d(out_channels))
    if relu:
        layers.append(nn.ReLU())
    return layers

class ResidualBlock(nn.Module):
    """ResidualBlock是一个继承自nn.Module的类，用于实现残差块。在神经网络中，残差块是一种常用的结构，用于解决梯度消失和梯度爆炸等问题。ResidualBlock包含一个卷积层序列self.conv，其中包含两个卷积层，每个卷积层都是由ConvLayer函数创建的。这两个卷积层的输入和输出通道数都是channels，卷积核大小都是3，步长都是1。第一个卷积层后面跟着一个ReLU激活函数，第二个卷积层没有激活函数。在前向传播过程中，输入张量x首先通过self.conv进行卷积操作，然后将卷积结果与输入张量相加，得到残差块的输出。"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            *ConvLayer(channels, channels, kernel_size=3, stride=1),
            *ConvLayer(channels, channels, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.conv(x) + x


class TransformNet(nn.Module):
    """TransformNet是一个继承自nn.Module的类，用于实现图像风格转换。
    它包含三个子模块：downsampling、residuals和upsampling。
    downsampling模块包含三个卷积层，用于将输入图像进行下采样。
    residuals模块包含五个残差块，用于提取图像的特征。
    upsampling模块包含三个卷积层，用于将特征图进行上采样，并生成最终的输出图像。
    在前向传播过程中，输入张量X首先通过downsampling模块进行下采样，然后通过residuals模块提取特征，
    最后通过upsampling模块进行上采样，得到输出图像。"""
    def __init__(self, base=32):
        super(TransformNet, self).__init__()
        self.downsampling = nn.Sequential(
            *ConvLayer(3, base, kernel_size=9),
            *ConvLayer(base, base * 2, kernel_size=3, stride=2),
            *ConvLayer(base * 2, base * 4, kernel_size=3, stride=2),
        )
        self.residuals = nn.Sequential(*[ResidualBlock(base * 4) for i in range(5)])
        self.upsampling = nn.Sequential(
            *ConvLayer(base * 4, base * 2, kernel_size=3, upsample=2),
            *ConvLayer(base * 2, base, kernel_size=3, upsample=2),
            *ConvLayer(base, 3, kernel_size=9, instance_norm=False, relu=False),
        )

    def forward(self, X):
        y = self.downsampling(X)
        y = self.residuals(y)
        y = self.upsampling(y)
        return y

def tensor_to_image(tensor):
    """
    将PyTorch张量转换为PIL图像。
    """
    image = tensor.clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
    return image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform_net = TransformNet(32).to(device)
transform_net.load_state_dict(torch.load('transform_net.pth'))

transform_net.eval()

style_path = "style1.jpg"
style_img = read_image(style_path).to(device)

# ("video1.mp4")

# 读取视频
cap = cv2.VideoCapture("video2.mp4")

# 获取视频宽度和高度
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# 创建视频编写器
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 30.0, (frame_width, frame_height))

# 逐帧处理视频
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将帧图像转换为PyTorch张量，并进行风格迁移
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = Image.fromarray(frame)
    # frame_tensor = transforms.ToTensor()(frame).unsqueeze(0).to(device)

    frame_tensor = image_to_tensor(frame)

    with torch.no_grad():
        output_tensor = transform_net(frame_tensor)
    # output_img = tensor_to_image(output_tensor.squeeze(0).cpu())
    # print(torch.equal(frame_tensor, output_tensor))

    output_img = recover_image(output_tensor)
    # 将输出图像转回OpenCV格式，并写入输出视频
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    out.write(output_img)

cap.release()
out.release()
cv2.destroyAllWindows()






