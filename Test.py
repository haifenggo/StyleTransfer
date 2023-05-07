import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

from torchvision.utils import save_image
from datetime import datetime

from PIL import Image
import matplotlib.pyplot as plt

import copy

# 添加全局变量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判断是否有GPU
unloader = transforms.ToPILImage()  # 将张量转换为PIL图像


class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()
        # self.mean = torch.tensor(mean).view(-1, 1, 1)
        # self.std = torch.tensor(std).view(-1, 1, 1)
        # 一个张量被克隆（clone）后，会生成一个新的张量，其在内存中与原张量没有任何关系，但是克隆张量保留了原始张量的历史操作信息，
        # 并与原始张量共享相同的梯度信息。而detach 操作则是用来从计算图中分离出张量的操作，它可以创建一个新的张量，并且保留与原来张量相同的数值，
        # 但是不再跟踪原先张量的梯度信息，也就是说，分离后的张量只能用于前向传播操作，无法进行反向传播求导操作。
        cnn_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        cnn_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.mean = cnn_mean.clone().detach().view(-1, 1, 1)
        self.std = cnn_std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


def load_image(img_path, imsize):
    image = Image.open(img_path)  # 打开图片
    # 将图片调整为指定大小
    loader = transforms.Compose([
        transforms.Resize(imsize),  # 调整图片大小
        transforms.CenterCrop(imsize),
        transforms.ToTensor(),  # 将图片转换为张量
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化图片的每一个通道
    ])
    # 添加一个维度，使图片符合VGG19输入维度要求
    image = loader(image).unsqueeze(0)  # 在第0维添加一个维度
    return image.to(device, torch.float)  # 将图片转换为指定类型


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # 将张量克隆到CPU上
    image = image.squeeze(0)  # 去掉第0维
    image = unloader(image)  # 将张量转换为PIL图像
    plt.imshow(image)  # 显示图像
    if title is not None:
        plt.title(title)  # 设置标题
    plt.pause(1)  # 暂停


def gram_matrix(input):
    batch_size, depth, height, width = input.size()  # 获取输入的形状
    features = input.view(batch_size * depth, height * width)  # 将输入的形状转换为(batch_size * depth, height * width)
    G = torch.mm(features, features.t())  # 计算Gram矩阵
    return G.div(batch_size * depth * height * width)  # 返回标准化后的Gram矩阵


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()  # 分离目标，detach()函数可以将一个张量从计算图中分离出来，这样就可以保证在反向传播时不会修改这个张量。

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)  # 计算损失
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()  # 分离目标

    def forward(self, input):
        G = gram_matrix(input)  # 计算Gram矩阵
        self.loss = nn.functional.mse_loss(G, self.target)  # 计算损失
        return input


def calculate_loss(model, content_losses, style_losses, content_weight, style_weight):
    content_loss = 0  # 内容损失
    style_loss = 0  # 风格损失

    for cl in content_losses:
        content_loss += cl.loss  # 计算内容损失
    for sl in style_losses:
        style_loss += sl.loss  # 计算风格损失

    content_loss = content_weight * content_loss
    style_loss = style_weight * style_loss
    # print(len(content_losses))
    # print("content_loss = {:.4f}, style_loss = {:.4f}".format(content_loss.item(), style_loss.item()))
    total_loss = content_loss + style_loss  # 计算总损失
    return content_loss, style_loss, total_loss


def model_build(content_img, style_img):
    # 定义VGG19模型 IMAGENET1K_V1 DEFAULT
    vgg = models.vgg19(weights='DEFAULT').features.to(device).eval()  # 提取vgg模型的features网络层部分
    # PyTorch的VGG模型实现被分为了两个字Sequential模型：features（包含卷积层和池化层）和classifier（包含全连接层）。
    # 我们将使用features模型，因为我们需要每一层卷积层的输出来计算内容和风格损失。
    # 在训练的时候有些层会有和评估不一样的行为，所以我们必须用.eval()将网络设置成评估模式。

    content_losses = []  # 内容损失
    style_losses = []  # 风格损失

    # 提取图像的内容特征
    content_layers = ['conv_4']  # 内容层
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']  # 风格层

    # 在这个代码段中，我们创建了一个自定义的VGG19模型，并将其保存在变量 model 中。在循环中，我们向模型中添加了各种网络层，包括卷积层、ReLU激活层、最大池化层和批量归一化层。通过逐层向模型中添加网络层，我们可以构建一个完整的神经网络。同时，我们还在循环中对每个卷积层执行一些处理，其中包括添加内容损失和风格损失。这些损失将用于计算训练过程中的损失函数。
    #
    # 在添加完所有的内容损失和风格损失之后，我们要将自定义的VGG模型中添加的最后一个损失层删除。原因是这个模型的最后一个损失层不是我们添加的内容损失层或者风格损失层，而是VGG模型本身的最后一个全连接层。如果不将其删除，将会影响到我们后面计算损失的过程。因此，我们使用切片操作 model[:-(len(content_losses) + len(style_losses))] 将其删除，使得模型与内容损失和风格损失正确连接。
    normalization = Normalization().to(device)
    model = nn.Sequential(normalization)  # 利用nn.Sequential() 自定义自己的网络层
    # 具体来说，归一化层可以将输入数据中每个维度的数值减去该维度的均值，并且除以该维度的标准差，从而使得每个维度的数值都在类似于标准正态分布的范围内。这样做可以使得网络在学习时更加稳定，加速训练过程，提高模型的精度和泛化能力，同时避免梯度消失或爆炸等问题。另外，归一化层还可以避免神经元的饱和和死亡现象，有效地解决了神经网络训练过程中出现的一些问题。

    i = 0
    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)  # 获取层的名称
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)  # 获取层的名称
            layer = nn.ReLU(inplace=False)  # 将inplace设置为False
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)  # 获取层的名称
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)  # 获取层的名称
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))  # 抛出异常

        model.add_module(name, layer)  # 添加模块

        if name in content_layers:
            target = model(content_img).detach()  # 分离目标
            content_loss = ContentLoss(target)  # 计算内容损失，这行代码添加的content_loss是一个网络层。在这行代码之前，我们定义了一个名为ContentLoss的类，该类继承自nn.Module。
            # 在该类中，我们定义了一个forward函数，该函数计算输入和目标之间的均方误差损失，并将其存储在self.loss中。
            # 在这行代码中，我们实例化了ContentLoss类，并将其添加到我们的模型中。这样，我们就可以在训练过程中计算内容损失。
            model.add_module("content_loss_{}".format(i), content_loss)  # 添加内容损失 ，
            content_losses.append(content_loss)  # 添加到内容损失列表

        if name in style_layers:
            target_feature = model(style_img).detach()  # 分离目标
            style_loss = StyleLoss(target_feature)  # 计算风格损失
            model.add_module("style_loss_{}".format(i), style_loss)  # 添加风格损失
            style_losses.append(style_loss)  # 添加到风格损失列表

    # 删除模型的最后一个损失层
    # print("pre")
    # print(model)
    # print(len(model)) # 44
    # 当调整模型时，这些额外的层需要从模型中排除以保持模型结构的正确性。
    # 在计算内容损失和风格损失时，我们只需要利用前面的一些层进行计算即可。因此，为了减少计算量，需要将不必要的层删去。
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss): break
    model = model[:(i + 1)]
    # model = model[:-(len(content_losses) + len(style_losses))]

    # print("after")
    # print(model)
    return model, content_losses, style_losses


def train(model, content_losses, style_losses, input_img, optimizer, num_steps=2, content_weight=1,
          style_weight=100000):
    for i in range(num_steps):
        # print("current is {}".format(i))

        # closure()函数是一个闭包函数，它被传递给optimizer.step()函数，用于计算损失并执行反向传播。
        # 在这个函数中，我们首先将content_img张量的值限制在0和1之间，然后将梯度清零。
        # 接下来，我们计算模型的输出，并计算总损失。最后，我们执行反向传播并返回损失。
        # optimizer.step()函数将使用这个函数来更新模型的参数。
        # 在这段代码中，我们使用了PyTorch提供的自动求导机制来计算损失值，然后在闭包函数（closure）中使用backward()函数进行反向传播，计算模型参数的梯度，
        # 并且将梯度传递给优化器（optimizer）进行更新。
        # 所以在每次迭代中，我们传递损失值给优化器，让它能够根据损失值来更新模型的参数，使得模型能够更好地适应训练数据。
        """
        为了计算内容和风格损失，代码分别使用了ContentLoss和StyleLoss类。
        在ContentLoss中，目标内容被分离，然后计算输入和目标之间的均方误差（MSE）损失。这个损失被作为前向传递的输出返回。
        在StyleLoss中，目标特征被通过gram_matrix函数传递（该函数计算输入的Gram矩阵），然后结果被分离。然后计算输入的Gram矩阵和目标Gram矩阵之间的MSE损失，并将其作为前向传递的输出返回。
        calculate_loss函数将模型、内容和风格损失、内容权重和风格权重作为输入。它通过分别对每个ContentLoss和StyleLoss实例的损失求和，然后将它们分别乘以它们的权重来计算内容和风格损失。总损失是内容损失和风格损失的总和。
        最后，train函数使用calculate_loss函数计算损失，并执行反向传播以使用优化器更新模型参数。closure函数用于在训练循环的每次迭代中计算损失并执行反向传播。然后，优化器使用计算出的梯度更新模型参数。
        """

        def closure():
            input_img.data.clamp_(0, 1)  # 将数据限制在0和1之间
            optimizer.zero_grad()  # 梯度清零
            model(input_img)  # 计算模型，将content_img张量传递给模型
            content_loss, style_loss, loss = calculate_loss(model, content_losses, style_losses, content_weight,
                                                            style_weight)  # 计算损失
            loss.backward()  # 反向传播
            return loss

        # 在训练神经网络时，我们通常使用梯度下降算法来更新模型的参数，求得优化后的模型参数。
        # 在每次迭代过程中，我们都会计算损失值，并在闭包函数中执行反向传播，计算模型参数的梯度。
        # 接着，这些梯度将被传递给优化器，优化器使用这些梯度来更新模型参数，使得模型能够更好地适应训练数据。
        optimizer.step(closure)  # 优化器更新
        input_img.data.clamp_(0, 1)

        if i % 1 == 0:
            content_loss, style_loss, total_loss = calculate_loss(model, content_losses, style_losses, content_weight,
                                                                  style_weight)  # 计算总损失
            # print("Step {}: Total Loss = {:.4f}".format(i, total_loss.item()))
            print("Step {}: content_loss = {:.4f}, style_loss = {:.4f}".format(i, content_loss.item(),
                                                                              style_loss.item()))  # 打印总损失
            # imshow(input_img, title='Output Image')  # 显示输出图像

    return input_img


def main():
    # 加载图像
    imsize = 512  # 128
    style_img = load_image("style5.jpg", imsize)  # 加载风格图片
    content_img = load_image("content5.jpg", imsize)  # 加载内容图片
    # input_img = content_img.clone()
    input_img = load_image("output.jpg", imsize)
    # input_img = load_image("content3.jpg", imsize)
    # print(input_img is content_img)
    # 显示图像
    plt.ion()  # 打开交互模式

    # plt.figure()
    # imshow(style_img, title='Style Image')  # 显示风格图片
    # plt.figure()
    # imshow(content_img, title='Content Image')  # 显示内容图片

    model, content_losses, style_losses = model_build(content_img, style_img)

    # load model param
    # Step 0:content_loss = 18.9210, style_loss = 72.1728
    # Step 1:content_loss = 18.1085, style_loss = 24.9845

    # 设置设备
    model.to(device)

    # 定义优化器，这个optimizer是优化器，它的作用是用来更新模型的参数，使得模型的输出更加接近于目标值。在这个代码中，我们使用了LBFGS优化器，它是一种基于拟牛顿法的优化器，可以在不计算完整的Hessian矩阵的情况下近似计算。在这个代码中，我们将content_img张量传递给优化器，并将其设置为需要梯度计算。在训练过程中，优化器将使用反向传播算法计算梯度，并使用这些梯度来更新模型的参数，以最小化损失函数。
    optimizer = optim.LBFGS([input_img.requires_grad_()], lr=1.5)

    # if os.path.exists("./model.pth"):
    #     # model.load_state_dict(torch.load('./model.pth'))
    #     checkpoint = torch.load('./model.pth')
    #     # model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     # print(checkpoint['model'])
    #     print("load model param")
    # else:
    #     print("start newly")

    # 训练模型
    num_steps = 10  # 迭代次数
    # print(len(content_losses))
    output = train(model, content_losses, style_losses, input_img, optimizer, num_steps=num_steps)  # 训练模型

    # print(optimizer)
    # state = {'optimizer': optimizer.state_dict()}
    # torch.save(state, './model.pth')
    # torch.save(model.state_dict(), './model.pth')
    # print(model.state_dict())

    # torch.save(model, "./model.pt")
    # model = torch.load("./model.pt")


    # 显示输出图像
    plt.figure()
    imshow(output, title='Output Image')  # 显示输出图像
    plt.ioff()  # 关闭交互模式
    plt.show()  # 显示图像

    # now = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    # filename = f"output_{now}.jpg"
    ask = int(input("1.保存, 其他丢弃:\n"))
    if ask == 1:
        filename = f"output.jpg"
        save_image(output, f"./{filename}")




def final_save():
    pass
    imsize = 512  # 128
    style_img = load_image("style3.jpg", imsize)  # 加载风格图片
    content_img = load_image("content4.jpg", imsize)  # 加载内容图片
    input_img = load_image("output.jpg", imsize)
    save_image(style_img, f"./saved_style.jpg")
    save_image(content_img, f"./saved_content.jpg")

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    imshow(style_img, title='Style Image')

    plt.subplot(1, 3, 2)
    imshow(content_img, title='Content Image')

    plt.subplot(1, 3, 3)
    imshow(input_img.detach(), title='Output Image')
    plt.pause(0)

if __name__ == '__main__':

    final_save()
    # main()
