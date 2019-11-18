#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
from timeit import default_timer as timer
import cv2
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image

class YOLO(object):
    '''def __init__(self):
        self.model_path = 'logs/000/trained_weights_stage_1.h5' # model path or trained weights path
        self.anchors_path = 'model_data/tiny_yolo_anchors.txt'
        self.classes_path = 'model_data/voc_classes.txt'''
    def __init__(self,yolo_model,model_path,anchors_path,image_size=(224,320),classes_path='model_data/my_classes.txt',score = 0.3,iou=0.45):
        self.model_path = model_path#'logs/000/trained_weights_stage_1.h5' # model path or trained weights path
        self.anchors_path = anchors_path#'model_data/tiny_yolo_anchors.txt'
        self.classes_path = classes_path#'model_data/voc_classes.txt'
        self.score = 0.3
        self.iou = 0.2
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = image_size # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate(yolo_model)

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self,yolo_model):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        # is_tiny_version = num_anchors==6 # default setting
        '''
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        '''
        try:
            self.yolo_model = load_model(model_path, compile=False)
            print('use model:'+model_path)
        except:
            self.yolo_model = yolo_model
            self.yolo_model.load_weights(self.model_path)
            print('load model:'+model_path)
        finally:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()
        label_yolo = []
        out = []
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        end = timer()
        print('resize:' + str(end - start))

        start = timer()
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        end = timer()
        print(end - start)
        start = timer()

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))


            
            
            w = int(right-left)
            h = int(bottom-top)
            x = int((right+left)*0.5)
            y = int(0.5 * (bottom+top))

            label_yolo.append([c,0.5*(right+left)/image.width, 0.5*(top+bottom)/image.height, 1.0*w/image.width, 1.0*h/image.height])
            out.append([c,left,top,w,h])

            top2 = max(top-h//2,0)
            right2 = min(right+w//2, image.size[0]) 
            bottom2 = min(bottom+h//2,image.size[1])
            left2 = max(0,left-w//2)
            
            # img = image.crop((left2, top2, right2, bottom2))
            # img.save('roi/'+label+'{},{},{},{}'.format(left2, top2, right2, bottom2)+'.png')

            

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw  
        end = timer()
        print('draw:' + str(end - start))  

        r, g, b = image.split()
        image = Image.merge('RGB', (b, g, r))
        return image,label_yolo ,out

    def label(self, image):
        
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                                image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        return out_boxes, out_scores, out_classes



    def predict(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        end = timer()
        print('resize:' + str(end - start))

        start = timer()
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        end = timer()
        print(end - start)

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        # out_classes.reshape(-1,1)
        out_classes = np.expand_dims(out_classes,1)
        if len(out_boxes) != 0:
            return np.concatenate((out_boxes, out_classes), axis = 1)
        else:
            return [[0,0,0,0,0]]

    
    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path, output_path="e:/g1.avi"):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    # kernel = np.ones((11,11),np.uint8)
    # lower_blue=np.array([60, 100, 100])
    # upper_blue=np.array([80, 240, 255])

    while True:
        return_value, frame = vid.read()
        
        
        # image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(image_hsv, lower_blue, upper_blue)
        # mask = cv2.bitwise_not(mask)
        # #closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # #opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        # res = cv2.bitwise_and(frame,frame,mask=mask)
        
        # image = rgb_to_hsv(frame)
        b,g,r = cv2.split(frame)
        clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8,8))
        # r = clahe.apply(r)
        

        # g = clahe.apply(g)
        # b = clahe.apply(b)
        frame = cv2.merge([b,g,r])
        image = Image.fromarray(frame)
        r, g, b = image.split()
        image = Image.merge('RGB', (b, g, r))

        
        

        image,l,c = yolo.detect_image(image)

        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        # cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

def detect_video_test(yolo, video_path, output_path=""):

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    # kernel = np.ones((11,11),np.uint8)
    # lower_blue=np.array([60, 100, 100])
    # upper_blue=np.array([80, 240, 255])
    cont = 0
    while True:
        return_value, frame = vid.read()
        
        if not return_value:
            break
        
    
        image = Image.fromarray(frame)
        if cont==0:
            image = yolo.detect_image(image)
            cont = 30
        else:
            yolo.track(frame)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cont -= 1
    yolo.close_session()
    out.release()

def detect_img(yolo,img=0):
    if img == 0:
        while True:
            img = input('Input image filename:')
            try:
                image = cv2.imread(img)
                image = np.array(image)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image, _, _ = yolo.detect_image(image)
                cv2.imshow('result', r_image)
                cv2.waitKey(0)
    else:
        try:
            # image = cv2.imread(img)
            # image = np.array(image)
            image = Image.open(img)
            # r, g, b = image.split()
            # image = Image.merge('RGB', (b, g, r))
        except:
            print('Open Error! Try again!')
        else:
            r_image, _, _= yolo.detect_image(image)
            r_image = np.asarray(r_image)
            # cv2.imwrite(img.split('/')[-1], r_image)
            cv2.imshow('result', r_image)
            cv2.waitKey(0)
    # yolo.close_session()