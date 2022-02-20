[toc]

# YOLOV5

## 环境配置

项目地址：[ultralytics/yolov5: YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite (github.com)](https://github.com/ultralytics/yolov5)

1.选择特定tags

2.在pycharm配置对应conda环境

3.作者提供requirements.txt文件

- 可以利用pycharm自带智能提示安装

- 利用pip install -r requirements.txt

4.若作者没有提供requirements.txt

- 根据报错信息 百度 安装缺少的库



## issue

### 使用低版本会遇到的错误

:heavy_exclamation_mark:warning：![68f23a4da99e4be79e7683650b5d7e7d-20220122220310835](YOLOV5.imgs/68f23a4da99e4be79e7683650b5d7e7d-20220122220310835.png)

说在models.py中找不到SPPF这个类，解决方法如下：

如果你用的是Tags5的话,就去Tags6里面的models/common.py里面去找到这个SPPF的类,把它拷过来到你这个Tags5的models/common.py里面,这样你的代码就也有这个类了,还要在Tags5的models/common.py里引入一个warnings包就可以了。
<img src="YOLOV5.imgs/ef0cb87bcb9646d3a7c5e5dec439eb9c.png" style="zoom:100%;" />

![在这里插入图片描述](YOLOV5.imgs/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAUm9vS2lDaGVu,size_18,color_FFFFFF,t_70,g_se,x_16.png)

![在这里插入图片描述](YOLOV5.imgs/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAUm9vS2lDaGVu,size_18,color_FFFFFF,t_70,g_se,x_16-20220122220509507.png)

![在这里插入图片描述](YOLOV5.imgs/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAUm9vS2lDaGVu,size_20,color_FFFFFF,t_70,g_se,x_16.png)

![在这里插入图片描述](YOLOV5.imgs/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAUm9vS2lDaGVu,size_20,color_FFFFFF,t_70,g_se,x_16-20220122220525427.png)



### macos环境报错

在detect.py头文件中

``` python
import os
os.environ['KMP_DUPLICATE_OK'] = 'True'
```







## detect参数

```python
parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
```

### --weights

- 权重



### --source

- 来源



### --conf-thres

- 是该分类的概率是多少才给予显示



### --iou-thres

- NMS-non maximum suppression（非极大值抑制）

![Selecting the Right Bounding Box Using Non-Max Suppression (with  implementation)](YOLOV5.imgs/graphic4-20220123143136331.jpg)

- iou- 交并比

 计算公式：

![Intersection Over Union (IOU). | Download Scientific Diagram](YOLOV5.imgs/Intersection-Over-Union-IOU.ppm)





eg：当iou大于某个阈值，就会从多个框中选择一个合适的框

<img src="YOLOV5.imgs/1*kK0G-BmCqigHrc1rXs7tYQ.jpeg" alt="Intersection over union (IoU) calculation for evaluating an image  segmentation model | by Oleksii Sheremet | Towards Data Science" style="zoom:40%;" />



- iou= 1

<img src="YOLOV5.imgs/image-20220123145851736.png" alt="image-20220123145851736" style="zoom:25%;" />



- Iou = 0.45

  <img src="YOLOV5.imgs/image-20220123145556144.png" alt="image-20220123145556144" style="zoom:25%;" />



- Iou = 0（iou>0,就会认为多个框是同一个目标）

<img src="YOLOV5.imgs/image-20220123145612626.png" alt="image-20220123145612626" style="zoom:25%;" />





### --view-img

- 画面实时显示

1. 在terminal中：

​		python detect.py -- view-img



2. 在项目主修改参数

![image-20220123161722537](YOLOV5.imgs/image-20220123161722537.png)

<img src="YOLOV5.imgs/image-20220123161836227.png" alt="image-20220123161836227" style="zoom:20%;" />

可以实时查看视频图片识别情况

<img src="YOLOV5.imgs/image-20220123172129347.png" alt="image-20220123172129347" style="zoom:33%;" />





### --classes

- 筛选检测

--classes 0 只检测人



### --argument

增强检测



## detect默认值

<img src="YOLOV5.imgs/image-20220123173454554.png" alt="image-20220123173454554" style="zoom:50%;" />





##  训练YOLOv5神经网络

### 本地上训练

- ![image-20220123180501935](YOLOV5.imgs/image-20220123180501935.png)

在别人模型的基础上进行训练

Yolov5s.pt

Yolov5m.pt

Yolov5l.pt

Yolov5x.pt



- ![image-20220123180621412](YOLOV5.imgs/image-20220123180621412.png)

<img src="YOLOV5.imgs/image-20220123180814481.png" alt="image-20220123180814481" style="zoom:50%;" />





- ![image-20220123180859245](YOLOV5.imgs/image-20220123180859245.png)

<img src="YOLOV5.imgs/image-20220123181030753.png" alt="image-20220123181030753" style="zoom:50%;" />



- ![image-20220123181147893](YOLOV5.imgs/image-20220123181147893.png)

<img src="YOLOV5.imgs/image-20220123181300799.png" alt="image-20220123181300799" style="zoom:50%;" />





- ![image-20220123181704941](YOLOV5.imgs/image-20220123181704941.png)

矩阵填充

old：<img src="YOLOV5.imgs/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70-20220123181838046.png" alt="Yolo训练技巧：Rectangular training/inference | 码农家园" style="zoom:50%;" />



new：<img src="YOLOV5.imgs/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0REX1BQX0pK,size_16,color_FFFFFF,t_70-20220123181847974.png" alt="Yolo训练技巧：Rectangular training/inference | 码农家园" style="zoom:50%;" />





- ![image-20220123182127733](YOLOV5.imgs/image-20220123182127733.png)

断点续训练：

<img src="YOLOV5.imgs/image-20220123182228410.png" alt="image-20220123182228410" style="zoom:50%;" />





- ![image-20220123184518759](YOLOV5.imgs/image-20220123184518759.png)

锚点

锚框

:star:（自学）



- ![image-20220123204853335](YOLOV5.imgs/image-20220123204853335.png)

净化参数



- ![image-20220123205324775](YOLOV5.imgs/image-20220123205324775.png)

adam 优化器

不开启默认梯度下降





### 云端训练

压缩项目，上传云端

``` 
!unzip /content/yolov5-5.0.zip -d /content/yolov5
```

``` 
!rm -rf /content/yolov5/__MACOSX
```

``` 
%cd /content/yolov5/yolov5-5.0
```

```
!pip install -r requirements.txt
```

``` 
%load_ext tensorboard
```

``` 
%tensorboard --logdir=runs/train
```

```
!python train.py --rect
```

```
!python train.py --rect --data=data/coco.yaml
```



查询显卡

```
!/opt/bin/nvidia-smi
```







## 制作训练数据集



![image-20220126123024578](YOLOV5.imgs/image-20220126123024578.png)
