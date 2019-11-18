import argparse
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,EarlyStopping
from keras.optimizers import Adam, SGD
from matplotlib import pyplot as plt
from keras.models import load_model
from os.path import isfile
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, SeparableConv2D,BatchNormalization,concatenate, Input,AveragePooling2D
from keras.layers import Conv2D, UpSampling2D, Concatenate
from keras.layers import MaxPooling2D
from keras.models import load_model, Model

from yolo3.utils import compose
from yolo3.model import DarknetConv2D, DarknetConv2D_BN_Leaky, DarknetSeparableConv2D_BN_Leaky
from keras.models import Model

import os
import keras

def _DenseLayer(x,num_input_features, growth_rate, bottleneck_width, drop_rate):

    growth_rate = int(growth_rate / 2)
    inter_channel = int(growth_rate  * bottleneck_width / 4) * 4 

    if inter_channel > num_input_features / 2:
        inter_channel = int(num_input_features / 8) * 4
        print('adjust inter_channel to ',inter_channel)

    
    x = Conv2D(inter_channel, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(growth_rate, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(inter_channel, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(growth_rate, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(growth_rate, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def basic_conv(x, output_channels, kernel_size, stride):
    x = Conv2D(output_channels, (kernel_size, kernel_size),strides=(stride,stride), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def _DenseBlock(x, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        
        for i in range(num_layers):
                x = _DenseLayer(x, num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)

        return x



def _StemBlock(x, num_init_features):

    num_stem_features = int(num_init_features/2)
    
    out = basic_conv(x, num_init_features, kernel_size=3, stride=2)
    branch2 = basic_conv(out, num_init_features, kernel_size=1, stride=1)
    branch2 = basic_conv(branch2, num_init_features, kernel_size=3, stride=2)
    branch1 = MaxPooling2D()(out)
    out = concatenate([branch1,branch2],axis=3)
    out = basic_conv(out, num_init_features, kernel_size=1, stride=1)
    
    return out
    
def PeleeNet(growth_rate=32, block_config=[3, 4, 8, 6],num_init_features=32, bottleneck_width=[1, 2, 4, 4], drop_rate=0.05, num_classes=10):


    if type(growth_rate) is list:
            growth_rates = growth_rate
            assert len(growth_rates) == 4, 'The growth rate must be the list and the size must be 4'
    else:
            growth_rates = [growth_rate] * 4

    if type(bottleneck_width) is list:
            bottleneck_widths = bottleneck_width
            assert len(bottleneck_widths) == 4, 'The bottleneck width must be the list and the size must be 4'
    else:
            bottleneck_widths = [bottleneck_width] * 4

    inp = Input((32, 32, 3))
    stem_block =  _StemBlock(inp, num_init_features)
    num_features = num_init_features

    for i, num_layers in enumerate(block_config):
            #block = _DenseLayer(stem_block,num_features,growth_rates[i], bottleneck_widths[i], drop_rate)
            block = _DenseBlock(stem_block,num_layers=num_layers, num_input_features=num_features,
                                bn_size=bottleneck_widths[i], growth_rate=growth_rates[i], drop_rate=drop_rate)
            
            #self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rates[i]

            block = basic_conv(block, num_features, kernel_size=1, stride=1)
            
            if i != len(block_config) - 1:
                block = AveragePooling2D()(block)
                num_features = num_features
                stem_block = block
            else:
                stem_block =  block
                
    #     block = MaxPooling2D(pool_size=(7,7))(block)
    block = Flatten()(block)
    block = Dropout(drop_rate)(block)
    block = Dense(num_classes, activation='softmax')(block)

    model = Model(inp,block)
    return model

def yolo_body_peleenet(inputs, num_anchors, num_classes):
    """Create YOLO_V3_mobilenet model CNN body in Keras."""
    mobilenet = PeleeNet()

    # input: 416 x 416 x 3
    # conv_pw_13_relu :13 x 13 x 1024
    # conv_pw_11_relu :26 x 26 x 512
    # conv_pw_5_relu : 52 x 52 x 256
    # mobilenet.summary()
    f1 = mobilenet.get_layer('activation_113').output
    # f1 :13 x 13 x 704
    
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
 
    f2 = mobilenet.get_layer('activation_82').output
    # f2: 26 x 26 x 512
    f2 = compose(
                Concatenate(),
                DarknetSeparableConv2D_BN_Leaky(256,(3,3))
                )([f1,f2])
    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(f2)

    return Model(inputs = inputs, outputs=[y1,y2])

if __name__ == '__main__':
        
        model = PeleeNet()
        model.summary()