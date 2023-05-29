# StyleTransfer

## 普通风格迁移

在当前`Test.py`的目录下放置风格图片,内容图片和输入图片,运行`Test.py`.

~~~python
style_img = load_image("style7.jpg", imsize)  # 加载风格图片
content_img = load_image("content6.jpg", imsize)  # 加载内容图片
input_img = load_image("output7.jpg", imsize)
~~~

## 快速风格迁移

一共包含4个`.py`文件,`Ai.py, utils.py, models.py, log.py `

### 训练模型

加入风格图片路径

~~~python
style_path = "style1.jpg"
~~~

运行`Ai.py`训练模型,得到`transform_net.pth`模型参数.

### 视频风格迁移

加入风格图片和要风格化的mp4视频

~~~python
style_path = "style1.jpg"
style_img = read_image(style_path).to(device)
# 读取视频
cap = cv2.VideoCapture("video2.mp4")
~~~

运行`log.py`风格化视频,得到`output.mp4`.