# pytorch_yolov5s_ncnn_android

## 环境配置
该工程主要在win10环境下完成，主要涉及
1：ncnn在win10下配置
2：yolov5s.pt模型转换onnx
3：onnx2ncnn
4：ncnn在Android下配置

### ncnn在win10下配置
[Windows下ncnn环境配置（VS2019）](https://blog.csdn.net/qq_36890370/article/details/104966786)，出现以下界面就代表安装成功了。
![成功界面](https://github.com/zlszhonglongshen/YOLO_NCNN_Android_demo/blob/master/pytorch_yolov5s_ncnn_android/ncnn%E9%85%8D%E7%BD%AE.png)

### pytorch框架下模型训练
略，网上资料比较多，自行白嫖。
一定是要在**yoloV5S**结构下训练!!!

### yolov5s.pt模型转换onnx
[yolov5](https://github.com/ultralytics/yolov5) 
作者已经为我们准备好了从pt到onnx转换的脚本

1： **转换过程中主要需求修改两个地方，[参考链接](https://github.com/sunnyden/YOLOV5_NCNN_Android/issues/3#issuecomment-668028381)**
```
【1】在导出onnx模型时，common.py删除切片操作还是采用的return self.conv(torch.cat([x,x,x,x], 1))替代。然后export.py导出模型时，输入大小改为训练模型时的一半，如输入大小640时为[320, 320]

【2】预测框不对是因为YoloV5.h文件中，预测层没有根据自己模型指定对，原来的是“394，“375，“output”这3层输出，可以根据Netron查看自己模型的3个预测层修改，我的是“output”，“423”，“442”，然后预测框正确。
```
![结果](https://github.com/zlszhonglongshen/YOLO_NCNN_Android_demo/blob/master/pytorch_yolov5s_ncnn_android/ncnn%E8%BD%AC%E6%8D%A2%E6%88%90%E5%8A%9F%E5%9B%BE.png)

2：修改完以后开始模型转换：
```
cd yolov5
export PYTHONPATH="$PWD"  # add path
python models/export.py --weights yolov5s.pt --img 320 --batch 1  # export
```
3：**这里记录下属于自己模型output维度，因为在推理的时候需要更换成自己的**

### onnx2ncnn
转换的过程也比较简单，主要有两步

第一步:安装onnx-simplifier
```
pip install -U onnx-simplifier --user
```
第二步：模型转换
```
1：python -m onnxsim  yolov5s.onnx yolov5s-sim.onnx
2：onnx2ncnn  yolov5s-sim.onnx yolov5.param yolov5.bin
```

### ncnn在Android下配置
第一步：在CPP/yoloV5.h中替换成自己的labels

第二步：在CPP/yoloV5.h中替换成自己的labels数目

第三步：在CPP/yoloV5.h中替换成自己的output维度，比如我的是(543,424)

参考链接：

* [YOLOv5_NCNN](https://github.com/WZTENG/YOLOv5_NCNN)

* [YOLOV5_NCNN_Android](https://github.com/sunnyden/YOLOV5_NCNN_Android)

## 效果图，标签没有改，暂时先这样子
![demo](https://github.com/zlszhonglongshen/YOLO_NCNN_Android_demo/blob/master/pytorch_yolov5s_ncnn_android/yolo5s.png)