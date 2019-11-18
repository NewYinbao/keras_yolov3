#!/usr/bin/env python
# -- coding: utf-8 --# @Author: newyinbao
# @Date: 2019-09-25 22:40:53
# @Function: 
# @TODO: 
# @Last Modified by:   newyinbao
# @Last Modified time: 2019-09-25 22:40:53


"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/7/4
"""
import os
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.backend import mean
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import plot_model

from yolo3.model import preprocess_true_boxes, yolo_body_mobilenet3s_5_15, yolo_body, tiny_yolo_body, yolo_loss, yolo_body_mobilenetv2, yolo_body_mobilenetv2_5_9, yolo_body_5_14, yolo_body_mobilenet3_5_14
from yolo3.utils import get_classes, get_anchors, data_generator_wrapper
import matplotlib.pyplot as plt
from yolo3.mobilenetv3 import yolo_body_mobilenet3s_5_15 as my_yolo_body


def _main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    from keras import backend as K
    config = tf.ConfigProto()

    # 怀疑是gpu版本用的
    # config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    K.set_session(sess)

    # annotation_path = 'dataset/WIDER_train.txt'  # 数据
    # annotation_path = 'VOCdevkit/VOC2010/2010_train_label.txt'
    annotation_path = 'data/all.txt'
    classes_path = 'model_data/my_classes.txt'  # 类别

    log_dir = 'logs/384/'  # 日志文件夹

    pretrained_path = 'logs/000/trained_weights_final.h5'  # 预训练模型
    # pretrained_path = 'logs/000/trained_weights_final.h5'  # 预训练模型
    anchors_path = 'model_data/yolo_anchors.txt'  # anchors

    class_names = get_classes(classes_path)  # 类别列表
    num_classes = len(class_names)  # 类别数
    anchors = get_anchors(anchors_path)  # anchors列表

    input_shape = (288, 384)        # 32的倍数，输入图像
    model = create_model(input_shape, anchors, num_classes,
                         freeze_body=2, load_pretrained=0,
                         weights_path=pretrained_path)  # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True,
                                 save_best_only=True, period=3)  # 只存储weights，
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=3, verbose=1)  # 当评价指标不在提升时，减少学习率
    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=1)  # 测试集准确率，下降前终止

    val_split = 0.1  # 训练和验证的比例
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(47)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)  # 验证集数量
    num_train = len(lines) - num_val  # 训练集数量

    """
    把目标当成一个输入，构成多输入模型，把loss写成一个层，作为最后的输出，搭建模型的时候，
    就只需要将模型的output定义为loss，而compile的时候，
    直接将loss设置为y_pred（因为模型的输出就是loss，所以y_pred就是loss），
    无视y_true，训练的时候，y_true随便扔一个符合形状的数组进去就行了。
    """
    if 0:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # 使用定制的 yolo_loss Lambda层
            'yolo_loss': lambda y_true, y_pred: y_pred})  # 损失函数

        batch_size = 32  # batch尺寸
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(
            num_train, num_val, batch_size))
        history = model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                                      steps_per_epoch=max(
                                          1, num_train // batch_size),
                                      validation_data=data_generator_wrapper(
            lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val // batch_size),
            epochs=500,
            initial_epoch=0,
            callbacks=[logging, checkpoint])
        # 存储最终的参数，再训练过程中，通过回调存储
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    if 1:  # 全部训练
        for i in range(len(model.layers)):
            model.layers[i].trainable = True

        # 训练初期学习率可以适当大点, 后期可以减小
        model.compile(optimizer=Adam(lr=1e-3),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 4  # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(
            num_train, num_val, batch_size))

        history = model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                                      steps_per_epoch=max(
                                          1, num_train // batch_size),
                                      validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors,
                                                                             num_classes),
                                      validation_steps=max(
                                          1, num_val // batch_size),
                                      epochs=20,
                                      initial_epoch=0,
                                      callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                 weights_path='model_data/yolo_weights.h5'):
    K.clear_session()  # 清除session
    h, w = input_shape  # 尺寸
    image_input = Input(shape=(h, w, 3))  # 图片输入格式
    num_anchors = len(anchors)  # anchor数量

    # YOLO的三种尺度，每个尺度的anchor数，类别数+边框4个+置信度1
    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],
                           3, num_classes + 5)) for l in range(num_anchors//3)]

    model_body = yolo_body(image_input, 3, num_classes)

    print('Create YOLOv3 model with {} anchors and {} classes.'.format(
        num_anchors, num_classes))

    if load_pretrained:  # 加载预训练模型
        model_body.load_weights(
            weights_path, by_name=True, skip_mismatch=True)  # 加载参数，跳过错误
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num):
                model_body.layers[i].trainable = False  # 将其他层的训练关闭
            print('Freeze the first {} layers of total {} layers.'.format(
                num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss,
                        output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors,
                                   'num_classes': num_classes,
                                   'ignore_thresh': 0.5}
                        )(model_body.output + y_true)
    model = Model(inputs=[model_body.input] + y_true,
                  outputs=model_loss)  # 模型，inputs和outputs
    model.summary()
    return model


if __name__ == '__main__':
    _main()
