# TRT-YOLOv8-Seg
使用TensorRT加速YOLOv8-Seg，完整的后端框架，包括Http服务器，Mysql数据库，ffmpeg视频推流等。

- YOLOv8 实例分割
- TensorRT INT8量化 模型部署
- Http Sever 
- Mysql 数据库
- FFmpeg 推流

## 一、YOLOv8 实例分割

[YOLOv8](https://github.com/ultralytics/ultralytics)  是一个 SOTA 模型，它建立在以前 YOLO 版本的成功基础上，并引入了新的功能和改进，以进一步提升性能和灵活性。具体创新包括一个新的骨干网络、一个新的 Ancher-Free 检测头和一个新的损失函数。

具体到 YOLOv8 算法，其核心特性和改动可以归结为如下：

1. **提供了一个全新的 SOTA 模型，包括 P5 640 和 P6 1280 分辨率的目标检测网络和基于 [YOLACT](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1904.02689) 的实例分割模型。和 YOLOv5 一样，基于缩放系数也提供了 N/S/M/L/X 尺度的不同大小模型，用于满足不同场景需求**
2. **骨干网络和 Neck 部分可能参考了 YOLOv7 ELAN 设计思想，将 YOLOv5 的 C3 结构换成了梯度流更丰富的 C2f 结构，并对不同尺度模型调整了不同的通道数，属于对模型结构精心微调，不再是无脑一套参数应用所有模型，大幅提升了模型性能。不过这个 C2f 模块中存在 Split 等操作对特定硬件部署没有之前那么友好了**
3. **Head 部分相比 YOLOv5 改动较大，换成了目前主流的解耦头结构，将分类和检测头分离，同时也从 Anchor-Based 换成了 Anchor-Free**
4. **Loss 计算方面采用了 TaskAlignedAssigner 正样本分配策略，并引入了 Distribution Focal Loss**
5. **训练的数据增强部分引入了 YOLOX 中的最后 10 epoch 关闭 Mosiac 增强的操作，可以有效地提升精度**

![](https://blog-1300216920.cos.ap-nanjing.myqcloud.com/243418644-7df320b8-098d-47f1-85c5-26604d761286.png)

## 二、TensorRT 量化

