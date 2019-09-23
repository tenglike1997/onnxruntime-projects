# onnxruntime projects
## Introduction
This repository include codes for some onnxruntime projects,such as classification, segmentation, detection, style transfer and super resolution.
## Onnxruntime
ONNX Runtime is a performance-focused complete scoring engine for Open Neural Network Exchange (ONNX) models, with an open extensible architecture to continually address the latest developments in AI and Deep Learning. 
In my repository,onnxruntime.dll have been compiled. You can download it and see specific information about onnxruntime in https://github.com/microsoft/onnxruntime.

## Projects
The programming language is C++ and The platform is Visual Studio. I have finished some projects based on onnxruntime official samples. The link have been mentioned afore. Also, you can download some onnx models in https://github.com/onnx/models. If necessary,you can see the structure onnx models in https://lutzroeder.github.io/netron/.

##### Windows

|  Network  | Classes | Input resolution | Batch size | Iterations | CPU Running time | GPU Running time | TRT Running time* |
| :-------: | :-----: | :--------------: | :--------: | :--------: | :--------------: | :--------------: | :---------------: |
| MobileNet |  1000   |     224x224      |     1      |    1000    |      19.56s      |      4.15s       |       1.05s       |
|  ERFNet   |    4    |     640x480      |     1      |    1000    |      >100s       |      12.93s      |       5.6s        |

##### Ubuntu

|  Network  | Classes | Input resolution | Batch size | Iterations | CPU Running time | GPU Running time | TRT Running time* |
| :-------: | :-----: | :--------------: | :--------: | :--------: | :--------------: | :--------------: | :---------------: |
| MobileNet |  1000   |     224x224      |     1      |    1000    |      20.09s      |      4.24s       |       0.79s       |
|  ERFNet   |    4    |     640x480      |     1      |    1000    |      >100s       |      13.56s      |       4.90s       |

*The TensorRT engine is compiled with FP16 settings. Just add "trt_builder->setFp16Mode(true);" to 339 line of tensorrt_execution_provider.cc, if you build libonnxruntime yourself.

### Classification
---
The onnx model is moblienet. You can download it in the link mentioned afore.
### Segmentation 
---
The onnx model is our trained erfnet. We use specific datasets to train erfnet.
### Detection
---
The onnx model is Tiny YOLOv2.You can download it in the link mentioned afore.
### Style transfer
---
The onnx model is Fast Neural Style Transfer. You can download it in the link mentioned afore.
### Super resolution
---
The onnx model is Super Resolution with sub-pixel CNN. You can download it in the link mentioned afore.
