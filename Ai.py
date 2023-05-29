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
from models import *

import numpy as np

from tqdm import tqdm
import random

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


style_path = "style1.jpg"
style_img = read_image(style_path).to(device)
plt.figure()
imshow(style_img, title='Style Image')

# vgg16 = models.vgg16(pretrained=True)
# models.vgg19(weights='DEFAULT')
vgg16 = models.vgg16(weights='DEFAULT')
vgg16 = VGG(vgg16.features[:23]).to(device).eval()

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

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w) #features是一个形状为(b, ch, h*w)的张量，features_t是features的转置，形状为(b, h*w, ch)。因此，features.bmm(features_t)计算了features的Gram矩阵，它是一个形状为(b, ch, ch)的矩阵。最后，Gram矩阵通过除以ch * h * w进行归一化。
    return gram


# !rm -rf /home/ypw/COCO/*/.AppleDouble

batch_size = 4
width = 256

data_transform = transforms.Compose([
    transforms.Resize(width),
    transforms.CenterCrop(width),
    transforms.ToTensor(),
    tensor_normalizer,
])
"""数据集是使用PyTorch内置的ImageFolder类加载的，该类从目录中加载图像数据。它假设目录结构是这样的，即每个子目录表示一个类，并包含该类的图像。ImageFolder类根据它们的目录名称自动为图像分配标签。在这种情况下，目录/home/ypw/COCO/是数据集的根目录，每个子目录包含特定类别的图像。

ImageFolder构造函数中的transform参数指定要应用于每个图像的数据预处理步骤。在这种情况下，data_transform变量是一个Compose对象，它将几个转换链接在一起，包括调整大小、中心裁剪和归一化。

DataLoader类用于批量加载数据集进行训练或测试。它以dataset对象为输入，以及batch_size和shuffle参数。batch_size参数指定每个批次中的样本数，shuffle参数指定是否在每个epoch之前对样本进行洗牌。"""
dataset = torchvision.datasets.ImageFolder('D:/PYproject/datasets/one/', transform=data_transform)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)




style_features = vgg16(style_img)
style_grams = [gram_matrix(x) for x in style_features] #这段代码的作用是计算风格图像的Gram矩阵，并将其存储在一个列表中。Gram矩阵是一个用于描述特征之间相关性的矩阵，它可以用于捕捉图像的纹理信息。在这里，我们使用VGG16模型提取风格图像的特征，并计算每个特征图的Gram矩阵。这些Gram矩阵将用于计算风格损失。在第二行代码中，我们使用detach()方法将计算出的Gram矩阵从计算图中分离出来，以便在后续的优化过程中不会对它们进行梯度更新。
style_grams = [x.detach() for x in style_grams]



def tensor_to_array(tensor):
    x = tensor.cpu().detach().numpy()
    x = (x * 255).clip(0, 255).transpose(0, 2, 3, 1).astype(np.uint8)
    return x


def save_debug_image(style_images, content_images, transformed_images, filename):
    """这个方法的作用是将风格迁移的结果保存为一张图片，其中包括原始风格图像、原始内容图像和风格迁移后的图像。
    具体来说，它接受四个参数：styleimages、contentimages、transformedimages和filename。
    其中，styleimages是一个形状为(3, H, W)的张量，表示风格图像；contentimages是一个形状为(N, 3, H, W)的张量，表示N个内容图像；transformedimages是一个形状为(N, 3, H, W)的张量，表示N个风格迁移后的图像；filename是保存结果图片的文件名。在方法内部，它首先将styleimages转换为PIL图像，然后将contentimages和transformedimages中的每个图像都恢复为0-255的像素值，并转换为PIL图像。接下来，它创建一个新的RGB图像，将styleimage放在左上角，然后将每个内容图像和对应的风格迁移后的图像放在右侧。最后，它将结果图像保存到指定的文件中。"""
    style_image = Image.fromarray(recover_image(style_images))
    content_images = [recover_image(x) for x in content_images]
    transformed_images = [recover_image(x) for x in transformed_images]

    new_im = Image.new('RGB', (style_image.size[0] + (width + 5) * 4, max(style_image.size[1], width * 2 + 5)))
    new_im.paste(style_image, (0, 0))

    x = style_image.size[0] + 5
    for i, (a, b) in enumerate(zip(content_images, transformed_images)):
        new_im.paste(Image.fromarray(a), (x + (width + 5) * i, 0))
        new_im.paste(Image.fromarray(b), (x + (width + 5) * i, width + 5))

    new_im.save(filename)

transform_net = TransformNet(32).to(device)

"""verbose_batch在代码块中没有被使用，但它很可能在程序的其他地方用于控制训练过程中进度更新的频率。

style_weight是一个超参数，它控制了总损失函数中风格损失项的相对重要性。style_weight值越高，生成图像中保留风格图像风格的强调就越强。

content_weight是一个超参数，它控制了总损失函数中内容损失项的相对重要性。content_weight值越高，生成图像中保留内容图像内容的强调就越强。

tv_weight是一个超参数，它控制了总损失函数中总变差损失项的相对重要性。总变差损失通过惩罚像素值的剧烈变化来鼓励生成图像的平滑性。tv_weight值越高，生成图像就越平滑。"""
verbose_batch = 800
style_weight = 1e5
content_weight = 1
tv_weight = 1e-6
# 这行代码的作用是创建一个Adam优化器，并将其绑定到transformnet模型的参数上。
# Adam是一种常用的优化算法，它可以自适应地调整每个参数的学习率，从而加速模型的收敛。
# 在这里，我们使用Adam优化器来更新transformnet模型的参数，以最小化生成图像与风格图像和内容图像之间的差异。优化器的学习率被设置为1e-3。
optimizer = optim.Adam(transform_net.parameters(), 1e-3)
# 这行代码的作用是将transformnet模型设置为训练模式。在PyTorch中，模型有两种模式：训练模式和评估模式。
# 在训练模式下，模型会保留所有中间变量的梯度信息，以便在反向传播时更新模型的参数。
# 而在评估模式下，模型不会保留梯度信息，从而减少内存消耗并加速模型的推理速度。
# 在这里，我们需要将transformnet模型设置为训练模式，以便在后续的优化过程中更新模型的参数。
transform_net.train()

transform_net.load_state_dict(torch.load('transform_net.pth')) # add
# print(transform_net.state_dict())

n_batch = len(data_loader)

print("start epoch..")
for epoch in range(10):
    print('Epoch: {}'.format(epoch + 1))
    smooth_content_loss = Smooth()
    smooth_style_loss = Smooth()
    smooth_tv_loss = Smooth()
    smooth_loss = Smooth()
    with tqdm(enumerate(data_loader), total=n_batch) as pbar:
        for batch, (content_images, _) in pbar:
            optimizer.zero_grad()

            # 使用风格模型预测风格迁移图像
            content_images = content_images.to(device)
            transformed_images = transform_net(content_images)
            transformed_images = transformed_images.clamp(-3, 3)

            # 使用 vgg16 计算特征
            content_features = vgg16(content_images)
            transformed_features = vgg16(transformed_images)

            # content loss
            content_loss = content_weight * F.mse_loss(transformed_features[1], content_features[1])

            # total variation loss
            y = transformed_images
            tv_loss = tv_weight * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
                                   torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

            # style loss
            style_loss = 0.
            transformed_grams = [gram_matrix(x) for x in transformed_features]
            for transformed_gram, style_gram in zip(transformed_grams, style_grams):
                style_loss += style_weight * F.mse_loss(transformed_gram,
                                                        style_gram.expand_as(transformed_gram))

            # 加起来
            loss = style_loss + content_loss + tv_loss

            loss.backward()
            optimizer.step()

            smooth_content_loss += content_loss.item()
            smooth_style_loss += style_loss.item()
            smooth_tv_loss += tv_loss.item()
            smooth_loss += loss.item()

            s = f'Content: {smooth_content_loss:.2f} '
            s += f'Style: {smooth_style_loss:.2f} '
            s += f'TV: {smooth_tv_loss:.4f} '
            s += f'Loss: {smooth_loss:.2f}'
            if batch % verbose_batch == 0:
                s = '\n' + s + '\n'
                save_debug_image(style_img, content_images, transformed_images,
                                 f"debug/s2_{epoch}_{batch}.jpg")

            pbar.set_description(s)
    torch.save(transform_net.state_dict(), 'transform_net.pth')


content_img = random.choice(dataset)[0].unsqueeze(0).to(device)
output_img = transform_net(content_img)

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
imshow(style_img, title='Style Image')

plt.subplot(1, 3, 2)
imshow(content_img, title='Content Image')

plt.subplot(1, 3, 3)
imshow(output_img.detach(), title='Output Image')



