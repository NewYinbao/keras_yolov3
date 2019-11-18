import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model

# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# tflite_quant_model = converter.convert()
# from tensorflow.python.keras.utils import CustomObjectScope
# def Hswish(x):
#     return x * tf.nn.relu6(x + 3) / 6
def relu6(x):
    return tf.nn.relu6(x )
# with CustomObjectScope({'Hswish': Hswish, 'relu6':relu6}):
kerasmodel = load_model('h5/yolo_body_mobilenetv2_5_9_data723.h5')
tflite_model = tf.lite.TFLiteConverter.from_keras_model(kerasmodel)
# tflite_model.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
model= tflite_model.convert()
with open('models/data7.23.tflite','wb') as f:
        f.write(model)
# model=tf.lite.TFLiteConverter.from_keras_model_file('/home/ship/tflitedemo/yolo_body_mobilenet3s_5_15_360.h5')
# # model.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# # model.post_training_quantize=True
# tfmodel = model.convert()
# open('yolo_body_mobilenet3s_5_15_360.tflite','wb').write(tfmodel)

print('successiful')

