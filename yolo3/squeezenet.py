from keras.applications import MobileNetV2
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, SeparableConv2D,Input
from yolo3.utils import compose
from yolo3.model import DarknetConv2D, DarknetConv2D_BN_Leaky, DarknetSeparableConv2D_BN_Leaky
from keras.models import Model



from keras_applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.utils import get_file
from keras.utils import layer_utils


sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"

# WEIGHTS_PATH = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels.h5"
# WEIGHTS_PATH_NO_TOP = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5"

# Modular function for Fire Node

def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    
    x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x


# Original SqueezeNet from paper.

def SqueezeNet(inputs = Input(shape=(32,32,3)),include_top=True, weights='imagenet',
               input_tensor=None, input_shape=None,
               pooling=None,
               classes=1000):
    """Instantiates the SqueezeNet architecture.
    """
        
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')


    input_shape = _obtain_input_shape(input_shape,
                                      default_size=227,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=(32,32,3))
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor


    x = Convolution2D(64, (3, 3), strides=(2, 2), padding='same', name='conv1')(inputs)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(2,2), padding='same', strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(2,2),padding='same',  strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(2,2), padding='same', strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    x = MaxPooling2D(pool_size=(2,2), padding='same', strides=(2, 2), name='pool9')(x)
    # if include_top:
    #     # It's not obvious where to cut the network... 
    #     # Could do the 8th or 9th layer... some work recommends cutting earlier layers.
    
    #     x = Dropout(0.5, name='drop9')(x)

    #     x = Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
    #     x = Activation('relu', name='relu_conv10')(x)
    #     x = GlobalAveragePooling2D()(x)
    #     x = Activation('softmax', name='loss')(x)
    # else:
    #     if pooling == 'avg':
    #         x = GlobalAveragePooling2D()(x)
    #     elif pooling=='max':
    #         x = GlobalMaxPooling2D()(x)
    #     elif pooling==None:
    #         pass
    #     else:
    #         raise ValueError("Unknown argument for 'pooling'=" + pooling)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    # if input_tensor is not None:
    #     inputs = get_source_inputs(input_tensor)
    # else:
    #     inputs = img_input

    model = Model(inputs, x, name='squeezenet')

    # load weights
    # if weights == 'imagenet':
    #     if include_top:
    #         weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels.h5',
    #                                 WEIGHTS_PATH,
    #                                 cache_subdir='models')
    #     else:
    #         weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5',
    #                                 WEIGHTS_PATH_NO_TOP,
    #                                 cache_subdir='models')
            
        # model.load_weights(weights_path)
        # if K.backend() == 'theano':
        #     layer_utils.convert_all_kernels_in_model(model)

        # if K.image_data_format() == 'channels_first':

        #     if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model

def yolo_body_squeezenet(inputs, num_anchors, num_classes):
    """Create YOLO_V3_mobilenet model CNN body in Keras."""
    mobilenet = SqueezeNet(inputs=inputs)

    # input: 416 x 416 x 3
    # conv_pw_13_relu :13 x 13 x 1024
    # conv_pw_11_relu :26 x 26 x 512
    # conv_pw_5_relu : 52 x 52 x 256
    # mobilenet.summary()
    f1 = mobilenet.get_layer('pool9').output
    # f1 :13 x 13 x 960
    
    # spp
    # sp3 = MaxPooling2D(pool_size=(3,3),strides=1,padding='same')(f1)
    # sp5 = MaxPooling2D(pool_size=(5,5),strides=1,padding='same')(f1)
    # f1 = compose(
    #         Concatenate(),
    #         DarknetConv2D_BN_Leaky(512, (1,1)))([sp3,sp5,f1])
    # end
    
    f1 = DarknetSeparableConv2D_BN_Leaky(256,(3,3))(f1)

    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(f1)
    
    f1 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(f1)
 
    f2 = mobilenet.get_layer('fire6/concat').output
    # f2: 26 x 26 x 576
    f2 = compose(
                Concatenate(),
                DarknetSeparableConv2D_BN_Leaky(256,(3,3))
                )([f1,f2])
    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(f2)

    return Model(inputs = inputs, outputs=[y1,y2])
if __name__ == "__main__":
    # model = yolo_body_mobilenetv2(Input(shape = (224,320,3)),3,2)
    model = SqueezeNet(include_top=False)
    model.summary()