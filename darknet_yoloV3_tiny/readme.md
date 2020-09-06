# darknet_yolov3_tiny
该目录下主要记载了，在darknet框架下训练的yoloV3-tiny模型，如何转换为ncnn以及在Android下如何部署
## 模型转换
### 模型训练
略，网上的资料很多，自行查阅
### darknet2ncnn环境安装
[darknet2ncnn](https://github.com/xiangweizeng/darknet2ncnn) 框架支持多种基于darknet训练的模型转换为ncnn，
配置过程主要遇到了结果问题，具体问题忘记截图了，解决方法我放在下方，自行配置的过程中可以按照error进行修改
```
CFLAGS+=-std=c++11

#include "opencv2/imgcodecs/legacy/constants_c.h"
```
### 模型转换
依据[tiny_demo](https://zhuanlan.zhihu.com/p/99904596) 中的提示，一步一步的运行命令，得到bin以及param文件代表转换成功。


## Android 部署
参考了以下案例

1：[Android_MobileNetV2-YOLOV3-Nano-NCNN](https://github.com/dog-qiuqiu/Android_MobileNetV2-YOLOV3-Nano-NCNN)

2：[darknet-ncnn-android](https://github.com/paleomoon/darknet-ncnn-android)

**真机下测试，需要注意的点**
```
1:在makelists文件夹添加如下代码

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -static-openmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -static-openmp")

2：替换掉自己在上一步中转换的模型
```

##结果展示
