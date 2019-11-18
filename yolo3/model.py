"""YOLO_v3 Model Defined in Keras."""

from yolo3.efficientnet  import EfficientNetB0
from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, SeparableConv2D,Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
# from tensorflow import ReLU6
from tensorflow import nn

from yolo3.utils import compose

from keras.applications.mobilenet import MobileNet
from keras.applications import MobileNetV2

from yolo3 import mobilenetv3
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])
    return x

def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x

def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1,1)))(x)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1,y2,y3])

def tiny_yolo_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    # x1 = compose(
    #         DarknetConv2D_BN_Leaky(16, (3,3)),
    #         MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
    #         DarknetConv2D_BN_Leaky(32, (3,3)),
    #         MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
    #         DarknetConv2D_BN_Leaky(64, (3,3)),
    #         MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
    #         DarknetConv2D_BN_Leaky(128, (3,3)),
    #         MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
    #         DarknetConv2D_BN_Leaky(256, (3,3)))(inputs)
    # x2 = compose(
    #         MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
    #         DarknetConv2D_BN_Leaky(512, (3,3)),
    #         MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
    #         DarknetConv2D_BN_Leaky(1024, (3,3)),
    #         DarknetConv2D_BN_Leaky(256, (1,1)))(x1)
    # # spp
    # # sp3 = MaxPooling2D(pool_size=(3,3),strides=1,padding='same')(x2)
    # # sp5 = MaxPooling2D(pool_size=(5,5),strides=1,padding='same')(x2)
    # # x2 = compose(
    # #         Concatenate(),
    # #         DarknetConv2D_BN_Leaky(256, (1,1)))([sp3,sp5,x2])
    # # end
    # y1 = compose(
    #         DarknetConv2D_BN_Leaky(512, (3,3)),
    #         DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    # x2 = compose(
    #         DarknetConv2D_BN_Leaky(128, (1,1)),
    #         UpSampling2D(2))(x2)
    # y2 = compose(
    #         Concatenate(),
    #         DarknetConv2D_BN_Leaky(256, (3,3)),
    #         DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D_BN_Leaky(512, (3,3)))(inputs)
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D_BN_Leaky(128, (1,1)))(x1)
    y1 = compose(
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(64, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.  ???
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))

    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])
    # H-sigmoid
    # box_xy = (H_sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    # box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    
    # box_confidence = H_sigmoid(feats[..., 4:5])
    # box_class_probs = H_sigmoid(feats[..., 5:])
    if calc_loss == True:
        
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [0,1,2]] # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []

    # 非最大抑制
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [0,1,2]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true


def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou

def H_sigmoid(x):
    return tf.nn.relu6(x + 3) / 6

def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=True):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [0,1,2]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        # xy_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_xy-K.sigmoid(raw_pred[...,0:2]))
        xy_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_xy-K.sigmoid(raw_pred[...,0:2]))
        # K.binary_crossentropy(raw_true_xy, (raw_pred[...,0:2]), from_logits=True)
        # wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
        confidence_loss = 2*K.square(1-K.sigmoid(raw_pred[...,4:5])) *object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            2*K.square(K.sigmoid(raw_pred[...,4:5]))*(1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])


        # confidence_loss = K.square(1-H_sigmoid(raw_pred[...,4:5]))*object_mask * 0.5* K.square(object_mask-H_sigmoid(raw_pred[...,4:5]))+ \
        #     K.square(H_sigmoid(raw_pred[...,4:5])) * (1-object_mask) * K.square(object_mask-H_sigmoid(raw_pred[...,4:5])) * 0.5 * ignore_mask

        # class_loss = object_mask * K.square(true_class_probs-H_sigmoid(raw_pred[...,5:]))

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
    if print_loss:
        loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss


def yolo_body_mobilenet(inputs, num_anchors, num_classes):
    """Create YOLO_V3_mobilenet model CNN body in Keras."""
    mobilenet = MobileNet(input_tensor=inputs,weights='imagenet')

    # input: 416 x 416 x 3
    # conv_pw_13_relu :13 x 13 x 1024
    # conv_pw_11_relu :26 x 26 x 512
    # conv_pw_5_relu : 52 x 52 x 256
    mobilenet.summary()
    f1 = mobilenet.get_layer('conv_pw_13_relu').output
    # f1 :13 x 13 x 1024
    
    # spp
    sp3 = MaxPooling2D(pool_size=(3,3),strides=1,padding='same')(f1)
    sp5 = MaxPooling2D(pool_size=(5,5),strides=1,padding='same')(f1)
    f1 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (1,1)))([sp3,sp5,f1])
    # end
    
    x, y1 = make_last_layers(f1, 512, num_anchors * (num_classes + 5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)

    f2 = mobilenet.get_layer('conv_pw_11_relu').output
    # f2: 26 x 26 x 512
    x = Concatenate()([x,f2])

    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)

    f3 = mobilenet.get_layer('conv_pw_5_relu').output
    # f3 : 52 x 52 x 256
    x = Concatenate()([x, f3])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs = inputs, outputs=[y1,y2,y3])


def yolo_body_mobilenetlite(inputs, num_anchors, num_classes):
    """Create YOLO_V3_mobilenet model CNN body in Keras."""
    mobilenet = MobileNet(input_tensor=inputs,weights='imagenet')

    # input: 416 x 416 x 3
    # conv_pw_13_relu :13 x 13 x 1024
    # conv_pw_11_relu :26 x 26 x 512
    # conv_pw_5_relu : 52 x 52 x 256
    # mobilenet.summary()
    f1 = mobilenet.get_layer('conv_pw_13_relu').output
    # f1 :13 x 13 x 1024
    
    # spp
    sp3 = MaxPooling2D(pool_size=(3,3),strides=1,padding='same')(f1)
    sp5 = MaxPooling2D(pool_size=(5,5),strides=1,padding='same')(f1)
    f1 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(512, (1,1)))([sp3,sp5,f1])
    # end
    
    f1 = DarknetSeparableConv2D_BN_Leaky(256,(3,3))(f1)

    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(f1)
    
    f1 = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(f1)
 
    f2 = mobilenet.get_layer('conv_pw_11_relu').output
    # f2: 26 x 26 x 512
    f2 = compose(
                Concatenate(),
                DarknetSeparableConv2D_BN_Leaky(256,(3,3))
                )([f1,f2])
    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(f2)

    
    f2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(f2)

    f3 = mobilenet.get_layer('conv_pw_5_relu').output
    # f3 : 52 x 52 x 256
    f3 = compose(
                Concatenate(),
                DarknetSeparableConv2D_BN_Leaky(128,(3,3))
                )([f2,f3])
    y3 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(f3)

    return Model(inputs = inputs, outputs=[y1,y2,y3])

# mobilenetv2 lite
def yolo_body_mobilenetv2(inputs, num_anchors, num_classes):
    """Create YOLO_V3_mobilenet model CNN body in Keras."""
    mobilenet2 = MobileNetV2(input_tensor=inputs,weights='imagenet')

    # input: 416 x 416 x 3
    # out_relu : 32 13 x 13 x 1280
    # block_13_expand_relu :16 26 x 26 x 576
    # block_6_expand_relu :8 52 x 52 x 192
    # mobilenet2.summary()
    f1 = mobilenet2.get_layer('out_relu').output
    # f1 :13 x 13 x 1280
    
    # spp
#     sp3 = MaxPooling2D(pool_size=(3,3),strides=1,padding='same')(f1)
#     sp5 = MaxPooling2D(pool_size=(5,5),strides=1,padding='same')(f1)
#     f1 = compose(
#             Concatenate(),
#             DarknetConv2D_BN_Leaky(512, (1,1)))([sp3,sp5,f1])
    # end
#     f1 = DarknetConv2D_BN_Leaky(256,(3,3))(f1)
    f1 = DarknetConv2D_BN_Leaky(256,(3,3))(f1)

    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(f1)
    
    f1 = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2, interpolation='bilinear'))(f1)
 
    f2 = mobilenet2.get_layer('block_13_expand_relu').output

#     f2 = DarknetConv2D_BN_Leaky(256,(1,1))(f2) # 周二添加
    # f2: 26 x 26 x 576
    f2 = compose(
                Concatenate(),
                DarknetConv2D_BN_Leaky(256,(3,3))
                )([f1,f2])
    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(f2)

    
    f2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2,interpolation='bilinear'))(f2)

    f3 = mobilenet2.get_layer('block_6_expand_relu').output
    # f3 : 52 x 52 x 192
    f3 = compose(
                Concatenate(),
                DarknetConv2D_BN_Leaky(128,(3,3))
                )([f2,f3])
    y3 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(f3)

    return Model(inputs = inputs, outputs=[y1,y2,y3])

# mobilenetv2 lite 精简网络
# # 5.9 
# 
# 
# 

def yolo_body_mobilenetv2_5_9(inputs, num_anchors, num_classes):
    """Create YOLO_V3_mobilenet model CNN body in Keras."""
    mobilenet2 = MobileNetV2(input_tensor=inputs,weights='imagenet')

    # input: 416 x 416 x 3
    # out_relu : 32 13 x 13 x 1280
    # block_13_expand_relu :16 26 x 26 x 576
    # block_6_expand_relu :8 52 x 52 x 192
    # mobilenet2.summary()
    f1 = mobilenet2.get_layer('block_15_expand_relu').output
    # f1 :13 x 13 x 960
    
    # f1 = DarknetConv2D_BN_Leaky(256,(3,3))(f1)

    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(f1)
    
    f1 = compose(
                DarknetSeparableConv2D_BN_Leaky(256, (3,3)),
                UpSampling2D(2, interpolation='bilinear'))(f1)
 
    f2 = mobilenet2.get_layer('block_13_expand_relu').output

    # f2: 26 x 26 x 576
    f2 = compose(
                Concatenate(),
                DarknetSeparableConv2D_BN_Leaky(128,(3,3))
                )([f1,f2])
    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(f2)

    
    # f2 = compose(
    #         DarknetSeparableConv2D_BN_Leaky(64, (1,1)),
    #         UpSampling2D(2,interpolation='bilinear'))(f2)

    # f3 = mobilenet2.get_layer('block_6_expand_relu').output
    # f3 : 52 x 52 x 192
    # f3 = compose(
    #             Concatenate(),
    #             DarknetSeparableConv2D_BN_Leaky(64,(3,3))
    #             )([f2,f3])
    # y3 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(f3)

    return Model(inputs = inputs, outputs=[y1,y2])
'''
将resblock中的卷积用深度可分离卷积代替
'''


@wraps(SeparableConv2D)
def DarknetSeparableConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return SeparableConv2D(*args, **darknet_conv_kwargs)

def DarknetSeparableConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetSeparableConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

# def DarknetSeparableConv2D_BN_Leaky(*args, **kwargs):
#     """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
#     no_bias_kwargs = {'use_bias': False}
#     no_bias_kwargs.update(kwargs)
#     # x = compose(
#     #     DarknetSeparableConv2D(*args, **no_bias_kwargs),
#     #     BatchNormalization())
    
#     return tf.nn.relu6(compose(
#         DarknetSeparableConv2D(*args, **no_bias_kwargs),
#         BatchNormalization()))

def Separable_resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
                #DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetSeparableConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])
    return x

def Separable_darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = Separable_resblock_body(x, 64, 1)
    x = Separable_resblock_body(x, 128, 2)
    x = Separable_resblock_body(x, 256, 8)
    x = Separable_resblock_body(x, 512, 8)
    x = Separable_resblock_body(x, 1024, 4)
    return x


def Separable_yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, Separable_darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[95].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[59].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1,y2,y3])


def darknet_body_test(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 4)
    x = resblock_body(x, 512, 2)
    x = resblock_body(x, 1024, 1)
    return x

def yolo_body_test(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body_test(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[82].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[64].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1,y2,y3])


def yolo_body_mobilenet3_5_14(inputs, num_anchors, num_classes):
    """Create YOLO_V3_mobilenet model CNN body in Keras."""
    mobilenet = mobilenetv3.my_mobilenet(inputs=inputs)

    # input: 416 x 416 x 3
    # out_relu : 32 13 x 13 x 1280
    # block_13_expand_relu :16 26 x 26 x 576
    # block_6_expand_relu :8 52 x 52 x 192
    # mobilenet2.summary()
    f1 = mobilenet.get_layer('block_7_dw_hs').output
    # f1 :13 x 13 x 512
    
    # f1 = DarknetConv2D_BN_Leaky(256,(3,3))(f1)

    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(f1)
    
    f1 = compose(
                DarknetSeparableConv2D_BN_Leaky(256, (3,3)),
                UpSampling2D(2, interpolation='bilinear'))(f1)
    # f2: 26 x 26 x 256
    f2 = mobilenet.get_layer('block_6_expand_hs').output
    f2 = DarknetConv2D(256, (1,1))(f2)
    
    f2 = compose(
                Concatenate(),
                DarknetSeparableConv2D_BN_Leaky(128,(3,3))
                )([f1,f2])
    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(f2)

    
#     f2 = compose(
#             DarknetConv2D_BN_Leaky(32, (1,1)),
#             UpSampling2D(2,interpolation='bilinear'))(f2)

#     f3 = mobilenet2.get_layer('block_5_add').output
#     # f3 : 52 x 52 x 192
#     f3 = compose(
#                 Concatenate(),
#                 DarknetConv2D_BN_Leaky(32,(3,3))
#                 )([f2,f3])
#     y3 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(f3)

    return Model(inputs = inputs, outputs=[y1,y2])



def yolo_body_5_14(inputs, num_anchors, num_classes):
    """Create YOLO_V3_mobilenet model CNN body in Keras."""
    mobilenet = mobilenetv3.build_mobilenet_v3(inputs=inputs,model_type='large')

    # input: 416 x 416 x 3
    # out_relu : 32 13 x 13 x 1280
    # block_13_expand_relu :16 26 x 26 x 576
    # block_6_expand_relu :8 52 x 52 x 192
    # mobilenet2.summary()
    f1 = mobilenet.get_layer('block_14_dw_hs').output
    # f1 :13 x 13 x 512
    
    # f1 = DarknetConv2D_BN_Leaky(256,(3,3))(f1)

    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(f1)
    
    f1 = compose(
                DarknetSeparableConv2D_BN_Leaky(256, (3,3)),
                UpSampling2D(2, interpolation='bilinear'))(f1)
    # f2: 26 x 26 x 576
    f2 = mobilenet.get_layer('block_14_expand_hs').output
    # f2 = DarknetConv2D(256, (1,1))(f2)
    
    f2 = compose(
                Concatenate(),
                DarknetSeparableConv2D_BN_Leaky(128,(3,3))
                )([f1,f2])
    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(f2)

    
#     f2 = compose(
#             DarknetConv2D_BN_Leaky(32, (1,1)),
#             UpSampling2D(2,interpolation='bilinear'))(f2)

#     f3 = mobilenet2.get_layer('block_5_add').output
#     # f3 : 52 x 52 x 192
#     f3 = compose(
#                 Concatenate(),
#                 DarknetConv2D_BN_Leaky(32,(3,3))
#                 )([f2,f3])
#     y3 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(f3)

    return Model(inputs = inputs, outputs=[y1,y2])


def yolo_body_mobilenet3s_5_15(inputs, num_anchors, num_classes):
    """Create YOLO_V3_mobilenet model CNN body in Keras."""
    mobilenet = mobilenetv3.build_mobilenet_v3(inputs=inputs, num_classes=10, model_type='large', pooling_type='avg', include_top=False)

    # input: 416 x 416 x 3
    # out_relu : 32 13 x 13 x 1280
    # block_13_expand_relu :16 26 x 26 x 576
    # block_6_expand_relu :8 52 x 52 x 192
    # mobilenet2.summary()
    f1 = mobilenet.get_layer('conv2d1_3__hs').output
    # f1 :13 x 13 x 960
    
    # f1 = DarknetConv2D_BN_Leaky(256,(3,3))(f1)

    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(f1)
    
    f1 = compose(
                DarknetSeparableConv2D_BN_Leaky(256, (3,3)),
                UpSampling2D(2, interpolation='bilinear'))(f1)
    # f2: 26 x 26 x 256
    f2 = mobilenet.get_layer('block_9_expand_hs').output
#     f2 = DarknetConv2D(256, (1,1))(f2)
    
    f2 = compose(
                Concatenate(),
                DarknetSeparableConv2D_BN_Leaky(256,(3,3))
                )([f1,f2])
    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(f2)

    
#     f2 = compose(
#             DarknetConv2D_BN_Leaky(32, (1,1)),
#             UpSampling2D(2,interpolation='bilinear'))(f2)

#     f3 = mobilenet2.get_layer('block_5_add').output
#     # f3 : 52 x 52 x 192
#     f3 = compose(
#                 Concatenate(),
#                 DarknetConv2D_BN_Leaky(32,(3,3))
#                 )([f2,f3])
#     y3 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(f3)

    return Model(inputs = inputs, outputs=[y1,y2])


def yolo_body_EfficientNetB0(inputs, num_anchors, num_classes):

    """Create YOLO_V3_mobilenet model CNN body in Keras."""
    mobilenet = EfficientNetB0(include_top=False,input_shape=(224,320,3))

    f1 = mobilenet.get_layer('multiply_14').output
    # f1 :13 x 13 x 576
    
    # spp
    # sp3 = MaxPooling2D(pool_size=(3,3),strides=1,padding='same')(f1)
    # sp5 = MaxPooling2D(pool_size=(5,5),strides=1,padding='same')(f1)
    # f1 = compose(
    #         Concatenate(),
    #         DarknetConv2D_BN_Leaky(512, (1,1)))([sp3,sp5,f1])
    # end
    
    f1 = DarknetSeparableConv2D_BN_Leaky(512,(3,3))(f1)

    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(f1)
    
    f1 = compose(
            Conv2D(256, (1,1)),
            UpSampling2D(2))(f1)
 
    f2 = mobilenet.get_layer('swish_34').output
    # f2: 26 x 26 x 512
    f2 = compose(
                Concatenate(),
                DarknetSeparableConv2D_BN_Leaky(256,(3,3))
                )([f1,f2])
    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(f2)

    return Model(inputs = mobilenet.input, outputs=[y1,y2])