from keras.applications import MobileNetV2
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, SeparableConv2D,Input
from yolo3.utils import compose
from yolo3.model import DarknetConv2D, DarknetConv2D_BN_Leaky, DarknetSeparableConv2D_BN_Leaky
from keras.models import Model
def yolo_body_mobilenetv2(inputs, num_anchors, num_classes):
    """Create YOLO_V3_mobilenet model CNN body in Keras."""
    mobilenet = MobileNetV2(input_tensor=inputs,weights='imagenet')

    # input: 416 x 416 x 3
    # conv_pw_13_relu :13 x 13 x 1024
    # conv_pw_11_relu :26 x 26 x 512
    # conv_pw_5_relu : 52 x 52 x 256
    # mobilenet.summary()
    f1 = mobilenet.get_layer('block_15_expand_relu').output
    # f1 :13 x 13 x 960
    
    # spp
    # sp3 = MaxPooling2D(pool_size=(3,3),strides=1,padding='same')(f1)
    # sp5 = MaxPooling2D(pool_size=(5,5),strides=1,padding='same')(f1)
    # f1 = compose(
    #         Concatenate(),
    #         DarknetConv2D_BN_Leaky(512, (1,1)))([sp3,sp5,f1])
    # end
    
    # f1 = DarknetSeparableConv2D_BN_Leaky(256,(3,3))(f1)

    y1 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(f1)
    
    f1 = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(f1)
 
    f2 = mobilenet.get_layer('block_13_expand_relu').output
    # f2: 26 x 26 x 576
    f2 = compose(
                Concatenate(),
                DarknetSeparableConv2D_BN_Leaky(256,(3,3))
                )([f1,f2])
    y2 = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(f2)

    return Model(inputs = inputs, outputs=[y1,y2])
if __name__ == "__main__":
    model = yolo_body_mobilenetv2(Input(shape = (224,320,3)),3,2)