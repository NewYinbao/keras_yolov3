# @Author: newyinbao
# @Date: 2019-09-25 22:31:27
# @Function:
# @TODO:
# @Last Modified by:   newyinbao
# @Last Modified time: 2019-09-25 22:31:27


"""
Retrain the tiny YOLO model for your own dataset.
you can chenge the backbone network with the networks defined in yolo3.model . 
"""

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss, yolo_body_mobilenetv2_5_9
from yolo3.utils import get_classes, get_anchors, data_generator_wrapper
import matplotlib.pyplot as plt

from yolo3.mobilenet import yolo_body_mobilenet
from yolo3.mobilenetv2 import yolo_body_mobilenetv2


def _main():

    # 这些路径可以自定义
    fig_path = 'fig/fackoff823.png'
    log_dir = 'logs/fackoff823/'

    # 训练数据及模型参数路径
    classes_path = 'model_data/my_classes.txt'
    anchors_path = 'model_data/320_224.txt'
    annotation_path = 'data/all.txt'

    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (224, 320)  # multiple of 32, hw

    is_tiny_version = len(anchors) == 6  # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
                                  freeze_body=2, load_pretrained=0, weights_path='others/detect8.5.h5')
    else:
        model = create_model(input_shape, anchors, num_classes,
                             freeze_body=2, weights_path='logs/yolo_body_mobilenetv2_5_93843/trained_weights_final.h5')  # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(21552)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if 0:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 32
        model.summary()
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(
            num_train, num_val, batch_size))
        history = model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                                      steps_per_epoch=max(
                                          1, num_train//batch_size),
                                      validation_data=data_generator_wrapper(
                                          lines[num_train:], batch_size, input_shape, anchors, num_classes),
                                      validation_steps=max(
                                          1, num_val//batch_size),
                                      epochs=100,
                                      initial_epoch=0,
                                      callbacks=[logging, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if 1:
        model.summary()
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        # recompile to apply the change
        model.compile(optimizer=Adam(lr=1e-3),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        print('Unfreeze all of the layers.')

        batch_size = 32  # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(
            num_train, num_val, batch_size))
        history = model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                                      steps_per_epoch=max(
                                          1, num_train//batch_size),
                                      validation_data=data_generator_wrapper(
            lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=100,
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
    plt.savefig(fig_path)
    plt.show()
    # Further training if needed.


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                 weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0: 32, 1: 16, 2: 8}[l], w//{0: 32, 1: 16, 2: 8}[l],
                           num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body_mobilenetv2_5_9(
        image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(
        num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            # 185 是对应 darknet, 自定义的模型需要修改, 但是不改也没啥事, 出事了可以过来看看
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(
                num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    return model


def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                      weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session()  # get a new session

    h, w = input_shape
    image_input = Input(shape=(h, w, 3))
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0: 32, 1: 16}[l], w//{0: 32, 1: 16}[l],
                           num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = yolo_body_mobilenetv2_5_9(
        image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(
        num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            # 20 也有隐患
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(
                num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    return model


if __name__ == '__main__':
    _main()
