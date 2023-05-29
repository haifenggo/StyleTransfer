import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
from collections import defaultdict

from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VGG(nn.Module):

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"

            # '5': "relu1_2",
            # '10': "relu2_2",
            # '19': "relu3_4",
            # '28': "relu4_4",
            # '35': "relu5_4"
        }
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        outs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                outs.append(x)
        return outs


class MyConv2D(nn.Module):
    """MyConv2D 是一个自定义的卷积层类，继承自 nn.Module。它的作用是实现卷积操作，其中权重和偏置都是可训练的。在初始化时，权重和偏置都被初始化为全零张量。在前向传播时，它调用了 F.conv2d 函数来实现卷积操作。extra_repr 函数用于返回一个字符串，描述该层的参数信息。"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(MyConv2D, self).__init__()
        self.weight = torch.zeros((out_channels, in_channels, kernel_size, kernel_size)).to(device)
        self.bias = torch.zeros(out_channels).to(device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        return s.format(**self.__dict__)


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


#
# class ResidualBlock(nn.Module):
#     """ConvLayer 是一个卷积层的工厂函数，用于创建一个卷积层的序列。在 ResidualBlock 中，ConvLayer 被调用两次，分别用于创建两个卷积层。第一个卷积层对输入进行处理，第二个卷积层不使用激活函数，用于将处理后的结果与输入相加。最终，ResidualBlock 的输出是输入和处理后的结果的和。"""
#
#     def __init__(self, channels):
#         super(ResidualBlock, self).__init__()
#         self.conv = nn.Sequential(
#             *ConvLayer(channels, channels, kernel_size=3, stride=1),
#             *ConvLayer(channels, channels, kernel_size=3, stride=1, relu=False)
#         )
#
#     def forward(self, x):
#         return self.conv(x) + x
#
#
# def ConvLayer(in_channels, out_channels, kernel_size=3, stride=1,
#               upsample=None, instance_norm=True, relu=True, trainable=False):
#     """ConvLayer 是一个工厂函数，用于创建卷积层的序列。它接受一些参数，包括输入通道数、输出通道数、卷积核大小、步长、上采样因子、是否使用实例归一化、是否使用 ReLU 激活函数以及是否可训练。它返回一个包含多个层的列表，这些层按顺序组成了一个卷积层。具体来说，它首先根据上采样因子添加上采样层，然后添加反射填充层，接着添加卷积层（如果可训练则使用 nn.Conv2d，否则使用自定义的 MyConv2D），然后添加实例归一化层和 ReLU 激活函数层。最终返回这些层的列表。"""
#     layers = []
#     if upsample:
#         layers.append(nn.Upsample(mode='nearest', scale_factor=upsample))
#     layers.append(nn.ReflectionPad2d(kernel_size // 2))
#     if trainable:
#         layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
#     else:
#         layers.append(MyConv2D(in_channels, out_channels, kernel_size, stride))
#     if instance_norm:
#         layers.append(nn.InstanceNorm2d(out_channels))
#     if relu:
#         layers.append(nn.ReLU())
#     return layers
#
#
# class TransformNet(nn.Module):
#     """TransformNet 类是一个 PyTorch 模块，用于实现图像风格转换的神经网络。它接受一个输入图像并对其应用学习到的变换，从而产生具有给定参考图像风格的输出图像。
#
# 该网络的架构由三个主要部分组成：下采样、残差和上采样。下采样部分减少输入图像的空间分辨率，同时增加通道数。残差部分由多个残差块组成，每个残差块包含两个具有相同通道数的卷积层。上采样部分增加图像的空间分辨率，同时减少通道数。
#
# TransformNet 类的方法如下：
#
# ### __init__(self, base=8)
# 构造函数，初始化 TransformNet 类的实例。base 参数指定网络中的通道数基数，默认为 8。
#
# ### get_param_dict(self)
# 计算网络中每个 MyConv2D 层所需的权重数量，并返回一个字典，其中键是层的名称，值是权重数量。
#
# ### set_weights(self, weights)
# 基于给定的一组权重设置网络中所有 MyConv2D 层的权重。weights 参数是一个包含所有权重的一维张量。"""
#
#     def __init__(self, base=8):
#         super(TransformNet, self).__init__()
#         self.base = base
#         self.weights = []
#         self.downsampling = nn.Sequential(
#             *ConvLayer(3, base, kernel_size=9, trainable=True),
#             *ConvLayer(base, base * 2, kernel_size=3, stride=2),
#             *ConvLayer(base * 2, base * 4, kernel_size=3, stride=2),
#         )
#         self.residuals = nn.Sequential(*[ResidualBlock(base * 4) for i in range(5)])
#         self.upsampling = nn.Sequential(
#             *ConvLayer(base * 4, base * 2, kernel_size=3, upsample=2),
#             *ConvLayer(base * 2, base, kernel_size=3, upsample=2),
#             *ConvLayer(base, 3, kernel_size=9, instance_norm=False, relu=False, trainable=True),
#         )
#         self.get_param_dict()
#
#     def forward(self, X):
#         y = self.downsampling(X)
#         y = self.residuals(y)
#         y = self.upsampling(y)
#         return y
#
#     def get_param_dict(self):
#         """找出该网络所有 MyConv2D 层，计算它们需要的权值数量"""
#         param_dict = defaultdict(int)
#
#         def dfs(module, name):
#             for name2, layer in module.named_children():
#                 dfs(layer, '%s.%s' % (name, name2) if name != '' else name2)
#             if module.__class__ == MyConv2D:
#                 param_dict[name] += int(np.prod(module.weight.shape))
#                 param_dict[name] += int(np.prod(module.bias.shape))
#
#         dfs(self, '')
#         return param_dict
#
#     def set_my_attr(self, name, value):
#         # 下面这个循环是一步步遍历类似 residuals.0.conv.1 的字符串，找到相应的权值
#         target = self
#         for x in name.split('.'):
#             if x.isnumeric():
#                 target = target.__getitem__(int(x))
#             else:
#                 target = getattr(target, x)
#
#         # 设置对应的权值
#         n_weight = np.prod(target.weight.shape)
#         target.weight = value[:n_weight].view(target.weight.shape)
#         target.bias = value[n_weight:].view(target.bias.shape)
#
#     def set_weights(self, weights, i=0):
#         """输入权值字典，对该网络所有的 MyConv2D 层设置权值"""
#         for name, param in weights.items():
#             self.set_my_attr(name, weights[name][i])


class MetaNet(nn.Module):
    """MetaNet 类接受一个字典，其中包含神经网络中每个层的参数数量，并创建一个元模型，该模型根据输入特征的均值和标准差预测每个层的滤波器。

在构造函数中，计算参数数量并将其存储在 self.param_num 中。然后创建一个具有 128self.paramnum 个输出单元的隐藏层，后跟一个字典，其中包含网络中每个层的全连接层 (nn.Linear)。字典的键是层的名称，值是相应的全连接层。

在 forward 方法中，将输入特征传递到隐藏层，并将输出拆分为大小为 128 的块，每个块对应于网络中的一个层。然后将每个层的全连接层应用于其相应的隐藏层输出块，从而得到每个层的滤波器字典。"""

    def __init__(self, param_dict):
        super(MetaNet, self).__init__()
        self.param_num = len(param_dict)
        self.hidden = nn.Linear(1920, 128 * self.param_num)
        self.fc_dict = {}
        for i, (name, params) in enumerate(param_dict.items()):
            self.fc_dict[name] = i
            setattr(self, 'fc{}'.format(i + 1), nn.Linear(128, params))

    def forward(self, mean_std_features):
        hidden = F.relu(self.hidden(mean_std_features))
        filters = {}
        for name, i in self.fc_dict.items():
            fc = getattr(self, 'fc{}'.format(i + 1))
            filters[name] = fc(hidden[:, i * 128:(i + 1) * 128])
        return filters
