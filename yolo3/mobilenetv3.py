import tensorflow as tf

from keras import backend as K
from keras.models import Model

from keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, Activation, Input, Add, UpSampling2D, Concatenate
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Softmax, Flatten

# ** to update custom Activate functions
from keras.utils.generic_utils import get_custom_objects
# from yolo3 import model as md

""" Define layers block functions """
def Hswish(x):
    return x * tf.nn.relu6(x + 3) / 6

def H_sigmoid(x):
    return tf.nn.relu6(x + 3) / 6

# ** update custom Activate functions
get_custom_objects().update({'custom_activation': Activation(Hswish)})

def __conv2d_block(_inputs, filters, kernel, strides, name, is_use_bias=False, padding='same', activation='RE'):
    
    x = Conv2D(filters, kernel, name = name+'_conv', strides= strides, padding=padding, use_bias=is_use_bias)(_inputs)
    x = BatchNormalization(name = name + '_bn')(x)
    if activation == 'RE':
        x = ReLU(name = name + '_relu')(x)
    elif activation == 'HS':
        x = Activation(Hswish,name = name + '_hs')(x)
    else:
        raise NotImplementedError
    return x

def __depthwise_block(_inputs, name, kernel=(3, 3), strides=(1, 1), activation='RE', is_use_se=True):
    x = DepthwiseConv2D(kernel_size=kernel, strides=strides, depth_multiplier=1, padding='same', name = name)(_inputs)
    x = BatchNormalization(name = name + '_bn')(x)
    if is_use_se:
        x = __se_block(x,name = name +'_se_')
    if activation == 'RE':
        x = ReLU(name = name +'_relu')(x)
    elif activation == 'HS':
        x = Activation(Hswish,name = name +'_hs')(x)
    else:
        raise NotImplementedError
    return x

def __global_depthwise_block(_inputs,name):
    assert _inputs._keras_shape[1] == _inputs._keras_shape[2]
    kernel_size = _inputs._keras_shape[1]
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(1, 1), depth_multiplier=1, padding='valid',name=name)(_inputs)
    return x

def __se_block(_inputs, name, ratio=4, pooling_type='avg'):
    filters = _inputs._keras_shape[-1]
    se_shape = (1, 1, filters)
    if pooling_type == 'avg':
        se = GlobalAveragePooling2D(name = name + 'avg')(_inputs)
    elif pooling_type == 'depthwise':
        se = __global_depthwise_block(_inputs, name = name+'dwpooling')
    else:
        raise NotImplementedError
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='hard_sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    return multiply([_inputs, se], name = name+'multip')

def __bottleneck_block(_inputs, out_dim, kernel, strides, expansion_ratio, bnblock_id, is_use_bais=False, shortcut=True, is_use_se=True, activation='RE', *args):
    # ** to high dim 
    # bottleneck_dim = K.int_shape(_inputs)[-1] * expansion_ratio
    prefix = 'block_{}_'.format(bnblock_id)

    # ** pointwise conv 
    x = __conv2d_block(_inputs, expansion_ratio, name = prefix + 'expand', kernel=(1, 1), strides=(1, 1), is_use_bias=is_use_bais, activation=activation)

    # ** depthwise conv
    x = __depthwise_block(x, name = prefix + 'dw',kernel=kernel, strides=strides, is_use_se=is_use_se, activation=activation)

    # ** pointwise conv
    x = Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same',name = prefix + 'project')(x)
    x = BatchNormalization(name = prefix + 'project_bn')(x)

    if shortcut and strides == (1, 1):
        in_dim = K.int_shape(_inputs)[-1]
        if in_dim != out_dim:
            ins = Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same',name = prefix + 'shortcut_conv1_1')(_inputs)
            x = Add(name = prefix + 'add')([x, ins])
        else:
            x = Add(name = prefix + 'add',)([x, _inputs])
    return x

def build_mobilenet_v3(inputs = Input(shape=(288, 512, 3)), num_classes=1000, model_type='small', pooling_type='avg', include_top=False):
    # ** input layer
    

    # ** feature extraction layers
    net = __conv2d_block(inputs, 16, name = 'conv2d0_3_',kernel=(3, 3), strides=(2, 2), is_use_bias=False, padding='same', activation='HS') 

    if model_type == 'large':
        config_list = large_config_list
    elif model_type == 'small':
        config_list = small_config_list
    else:
        raise NotImplementedError
        
    for config in config_list:
        net = __bottleneck_block(net, *config)
    
    # ** final layers
    net = __conv2d_block(net, 960, name = 'conv2d1_3_',kernel=(3, 3), strides=(1, 1), is_use_bias=True, padding='same', activation='HS')

    if pooling_type == 'avg':
        net = GlobalAveragePooling2D(name = 'final_avgpooling')(net)
    elif pooling_type == 'depthwise':
        net = __global_depthwise_block(net,name = 'final_dwpooling')
    else:
        raise NotImplementedError

    # ** shape=(None, channel) --> shape(1, 1, channel) 
    pooled_shape = (1, 1, net._keras_shape[-1])

    net = Reshape(pooled_shape)(net)
    net = Conv2D(1280, (1, 1), strides=(1, 1), padding='valid', use_bias=True)(net)
    
    if include_top:
        net = Conv2D(num_classes, (1, 1), strides=(1, 1), padding='valid', use_bias=True)(net)
        net = Flatten()(net)
        net = Softmax()(net)

    model = Model(inputs=inputs, outputs=net)

    return model

""" define bottleneck structure """
# ** 
# **             
global large_config_list    
global small_config_list

large_config_list = [[16,  (3, 3), (1, 1), 16, 1, False, False, False, 'RE'],
                     [24,  (3, 3), (2, 2), 64, 2, False, False, False, 'RE'],
                     [24,  (3, 3), (1, 1), 72, 3, False, True,  False, 'RE'],
                     [40,  (5, 5), (2, 2), 72, 4, False, False, True,  'RE'],
                     [40,  (5, 5), (1, 1), 120, 5,False, True,  True,  'RE'],
                     [40,  (5, 5), (1, 1), 120, 6,False, True,  True,  'RE'],
                     [80,  (3, 3), (2, 2), 240, 7,False, False, False, 'HS'],
                     [80,  (3, 3), (1, 1), 200, 8,False, True,  False, 'HS'],
                     [80,  (3, 3), (1, 1), 184, 9,False, True,  False, 'HS'],
                     [80,  (3, 3), (1, 1), 184, 10,False, True,  False, 'HS'],
                     [112, (3, 3), (1, 1), 360, 11,False, False, True,  'HS'],
                     [112, (3, 3), (1, 1), 360, 12,False, True,  True,  'HS'],
                     [160, (5, 5), (1, 1), 480, 13,False, False, True,  'HS'],
                     [160, (5, 5), (2, 2), 672, 14,False, True,  True,  'HS'],
                     [160, (5, 5), (1, 1), 960, 15,False, True,  True,  'HS']]

# small_config_list = [[16,  (3, 3), (2, 2), 16,  False, False, True,  'RE'],
#                      [24,  (3, 3), (2, 2), 72,  False, False, False, 'RE'],
#                      [24,  (3, 3), (1, 1), 88,  False, True,  False, 'RE'],
#                      [40,  (5, 5), (1, 1), 96,  False, False, True,  'HS'],
#                      [40,  (5, 5), (1, 1), 240, False, True,  True,  'HS'],
#                      [40,  (5, 5), (1, 1), 240, False, True,  True,  'HS'],
#                      [48,  (5, 5), (1, 1), 120, False, False, True,  'HS'],
#                      [48,  (5, 5), (1, 1), 144, False, True,  True,  'HS'],
#                      [96,  (5, 5), (2, 2), 288, False, False, True,  'HS'],
#                      [96,  (5, 5), (1, 1), 576, False, True,  True,  'HS'],
#                      [96,  (5, 5), (1, 1), 576, False, True,  True,  'HS']]

# small_config_list = [[16,  (3, 3), (2, 2), 16, 1, False, False, True,  'RE'],
#                      [24,  (3, 3), (2, 2), 72, 2, False, False, False, 'RE'],
#                      [24,  (3, 3), (1, 1), 88, 3, False, True,  False, 'RE'],
#                      [40,  (5, 5), (1, 1), 96, 4, False, False, True,  'HS'],
#                     #  [40,  (5, 5), (1, 1), 240, False, True,  True,  'HS'],
#                      [40,  (3, 3), (2, 2), 240, 5, False, False, False, 'HS'],
#                      [40,  (5, 5), (1, 1), 240, 6,False, True,  True,  'HS'],
#                      [48,  (5, 5), (1, 1), 120, 7,False, False, True,  'HS'],
#                      [48,  (5, 5), (1, 1), 144, 8,False, True,  True,  'HS'],
#                      [96,  (5, 5), (2, 2), 256, 9,False, False, True,  'HS'],
#                     #  [96,  (5, 5), (1, 1), 576, 10,False, True,  True,  'HS'],
#                      [96,  (5, 5), (1, 1), 576, 10,False, True,  True,  'HS']]
# small_config_list = [[16,  (3, 3), (2, 2), 16, 1, False, False, True,  'RE'],
#                      [24,  (3, 3), (2, 2), 72, 2, False, False, False, 'RE'],
#                      [24,  (3, 3), (1, 1), 88, 3, False, True,  False, 'RE'],
#                      [40,  (5, 5), (1, 1), 96, 4, False, False, True,  'RE'],
#                     #  [40,  (5, 5), (1, 1), 240, False, True,  True,  'HS'],
#                      [40,  (3, 3), (2, 2), 240, 5, False, False, False, 'RE'],
#                      [40,  (5, 5), (1, 1), 240, 6,False, True,  True,  'RE'],
#                      [48,  (5, 5), (1, 1), 120, 7,False, False, True,  'RE'],
#                      [48,  (5, 5), (1, 1), 144, 8,False, True,  True,  'RE'],
#                      [96,  (5, 5), (2, 2), 256, 9,False, False, True,  'RE'],
#                     #  [96,  (5, 5), (1, 1), 576, 10,False, True,  True,  'HS'],
#                      [96,  (5, 5), (1, 1), 576, 10,False, True,  True,  'RE']]
small_config_list = [[16,  (3, 3), (2, 2), 16, 1, False, False, True,  'RE'],
                     [24,  (3, 3), (2, 2), 72, 2, False, False, False, 'RE'],
                     [24,  (3, 3), (1, 1), 88, 3, False, True,  False, 'RE'],
                     [40,  (5, 5), (1, 1), 96, 4, False, False, True,  'HS'],
                     [40,  (3, 3), (2, 2), 240, 5, False, False, False, 'HS'],
                     [40,  (5, 5), (1, 1), 240, 6,False, True,  True,  'HS'],
                     [48,  (5, 5), (1, 1), 120, 7,False, False, True,  'HS'],
                     [48,  (5, 5), (1, 1), 144, 8,False, True,  True,  'HS'],
                     [96,  (5, 5), (2, 2), 256, 9,False, False, True,  'RE'],
                     [96,  (5, 5), (1, 1), 576, 10,False, True,  True,  'HS']]

def my_mobilenet(inputs = Input(shape=(288,512,3))):
    # h,w = inputs
    # inputs = Input(shape=(h, w, 3))
    # net = Conv2D(16,kernel_size=(3,3),stride=(2,2),padding='same',)
    net = __conv2d_block(inputs, 16,name = 'conv2d0_3', kernel=(3, 3), strides=(2, 2), is_use_bias=False, padding='same', activation='HS')
    
    # out_dim, kernel, strides, expansion_ratio, bnblock_id, is_use_bais=False, shortcut=True, is_use_se=True, activation='RE',
    config_list = [ [16,  (3, 3), (2, 2), 16, 1, False, False, True,  'RE'],

                    [24,  (3, 3), (2, 2), 72, 2, False, False, False, 'RE'],
                    [24,  (3, 3), (1, 1), 88, 3, False, True,  True, 'RE'],

                    [40,  (3, 3), (2, 2), 96, 4, False, False, True,  'HS'],
                    [40,  (5, 5), (1, 1), 256, 5, False, True,  True,  'HS'],

                    # [48,  (5, 5), (1, 1), 120, False, False, True,  'HS'],
                    # [48,  (5, 5), (1, 1), 144, False, True,  True,  'HS'],
                    [96,  (5, 5), (2, 2), 256, 6,False, False, True,  'HS'],
                    [96,  (3, 3), (1, 1), 512, 7,False, True,  True,  'HS']]
                    # [96,  (5, 5), (1, 1), 576, False, True,  True,  'HS']]

    for config in config_list:
        net = __bottleneck_block(net, *config)

    net = Conv2D(512,kernel_size=(1,1),padding='same')(net)
    model = Model(inputs=inputs, outputs=net)
    return model

def __yolo_block(_inputs, out_dim, kernel, strides, expansion_ratio, bnblock_id, is_use_bais=False, shortcut=True, is_use_se=True, activation='HS', *args):
    prefix = 'block_{}_'.format(bnblock_id)

    # ** pointwise conv 
    x = __conv2d_block(_inputs, expansion_ratio, name = prefix + 'expand', kernel=(1, 1), strides=(1, 1), is_use_bias=is_use_bais, activation=activation)
    f1 = x
    # ** depthwise conv
    x = __depthwise_block(x, name = prefix + 'dw',kernel=kernel, strides=strides, is_use_se=is_use_se, activation=activation)

    # ** pointwise conv
    x = Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same',name = prefix + 'project')(x)
    x = BatchNormalization(name = prefix + 'project_bn')(x)

    if shortcut and strides == (1, 1):
        in_dim = K.int_shape(_inputs)[-1]
        if in_dim != out_dim:
            ins = Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same',name = prefix + 'shortcut_conv1_1')(_inputs)
            x = Add(name = prefix + 'add')([x, ins])
        else:
            x = Add(name = prefix + 'add',)([x, _inputs])
    return (x,f1)


def yolo_body_mobilenet3s_5_15(inputs, num_anchors, num_classes,model_type='small'):
    """Create YOLO_V3_mobilenet model CNN body in Keras."""


    net = __conv2d_block(inputs, 16, name = 'conv2d0_3_',kernel=(3, 3), strides=(2, 2), is_use_bias=False, padding='same', activation='HS') 

    if model_type == 'large':
        config_list = large_config_list
        for i in range(8):
            net = __bottleneck_block(net, *config_list[i])  


    elif model_type == 'small':
        config_list = small_config_list
        for i in range(8):
            net = __bottleneck_block(net, *config_list[i]) 
        
        net,f2 = __yolo_block(net, *config_list[8])
        print(f2.shape)
        for i in range(9,len(config_list)):
            net = __bottleneck_block(net, *config_list[i])
    else:
        raise NotImplementedError
    
    # ** final layers
    # net = __conv2d_block(net, 512, name = 'conv2d1_3_',kernel=(3, 3), strides=(1, 1), is_use_bias=True, padding='same', activation='RE')
    net = md.DarknetSeparableConv2D_BN_Leaky(256, (3,3))(net)

    y1 = Conv2D(num_anchors*(num_classes+5), (1,1))(net)
    
    # net = __conv2d_block(net, 256, name = 'conv2d2_3_',kernel=(3, 3), strides=(1, 1), is_use_bias=True, padding='same', activation='RE')
    # net = DarknetSeparableConv2D_BN_Leaky(256, (3,3)) (net)  
    net = UpSampling2D(2, interpolation='bilinear')(net)

    f2 = Concatenate()([f2,net])
    f2 = md.DarknetSeparableConv2D_BN_Leaky(128, (3,3))(f2)
    # f2 = __conv2d_block(net, 256, name = 'conv2d3_3_',kernel=(3, 3), strides=(1, 1), is_use_bias=True, padding='same', activation='RE')

    y2 = Conv2D(num_anchors*(num_classes+5), (1,1))(f2)

    return Model(inputs = inputs, outputs=[y1,y2])








if __name__ == "__main__":
    # model = my_mobilenet()
    """ build MobileNet V3 model """
    model = build_mobilenet_v3(inputs=Input(shape=(352,480,3)), num_classes=2, model_type='large', pooling_type='avg', include_top=True)
    # layer.out_dim('block_1_expand_conv')
    
    print(model.summary())
    # print(model.get_layer('block_1_expand_conv').out_dim())


   