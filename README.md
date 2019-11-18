# keras-yolo3

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).


---

## Quick Start(改完不确定能用了)

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).  注：已经下载到本目录下
2. Convert the Darknet YOLO model to a Keras model. 注：现已经转化了识别两类目标的keras模型，在model—data文件夹下（.h5文件）
3. Run YOLO detection.

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
python yolo.py   OR   python yolo_video.py
```

For Tiny YOLOv3, just do in a similar way. And modify model path and anchor path in yolo.py.

---

## Training

1. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    For VOC dataset, try `python voc_annotation.py`  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```
    1.2 也可以用data2label.py
    
        （1）将标注文件.xml和图片.jpg分别放在  data/label/ 和 data/image/
        
        （2）运行data2label.py ， 在data/文件夹可看到生成的trainlabel.txt 和 test_train.txt


2. Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`  
    The file model_data/yolo_weights.h5 is used to load pretrained weights.

    现有的.h5文件是针对两类目标的，其他情况需要更改.cfg文件 ，然后重新运行 2 中指令

3. 数据文件夹格式

        ```
        data
        ├── Readme.md                   // help
        ├── code                        // 数据处理脚本
        ├── datax.xx                    // 数据目录,以时间命名
        │   ├── image                   // 图片目录
        │   ├── label                   // 标注目录
        │   ├── *.txt                   // 转化后的标注格式
        │   └── 其他文件                 // 其他文件,如其他未标记数据,临时文件
        ├── datax.xx
        ├── ...
        └── 
        
        ```
iarc mission8 数据下载地址: 
        
        链接: https://pan.baidu.com/s/1NObYblKPmtIXyj55oDblJw 
        提取码: 2sir 
    
或 [iarc2019data](http://219.217.235.37/gitlab/NewYinbao/iarc2019data)
3. Modify train.py and start training.  
    `python train.py`  
    Use your trained weights or checkpoint weights in yolo.py.  
    Remember to modify class path or anchor path.

If you want to use original pretrained weights for YOLOv3:  
    1. `wget https://pjreddie.com/media/files/darknet53.conv.74`  
    2. rename it as darknet53.weights  
    3. `python convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5`  
    4. use model_data/darknet53_weights.h5 in train.py

---

## Some issues to know

1. The test environment is
    - Python 3.6.8
    - Keras 2.2.4
    - tensorflow 1.13.0
    - 版本更高没问题, 但tf是1.x

2. Default anchors are used. If you use your own anchors, probably some changes are needed.
    
    -已更新,见 kmaen/findeanchors.py
    

3. The inference result is not totally the same as Darknet but the difference is small.

4. The speed is slower than Darknet. Replacing PIL with opencv may help a little.

5. Always load pretrained weights and freeze layers in the first stage of training. Or try Darknet training. It's OK if there is a mismatch warning.

6. The training strategy is for reference only. Adjust it according to your dataset and your goal. And add further strategy if needed.

7. 关于代码理解：可以从train.py 开始阅读，根据函数实现反推理解

