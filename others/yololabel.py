#! /usr/bin/env python
# -*- coding: utf-8 -*-
# # @Author: newyinbao
# @Date: 2019-09-25 22:17:38
# @Function:
# @TODO:
# @Last Modified by:   newyinbao
# @Last Modified time: 2019-09-25 22:17:38


"""
Run a YOLO_v3 style detection model on test images and create auto-labels.
"""
from predict.yolo import YOLO, detect_img, detect_video
from yolo3.model import yolo_body
from keras.layers import Input
import cv2
from PIL import Image, ImageFont, ImageDraw


if __name__ == '__main__':
    cont = 0
    # model path or trained weights path trained/yolo_myanchors
    model_path = 'logs/000/trained_weights_final.h5'
    anchors_path = 'model_data/anchors_f1.txt'  # 'model_data/tiny_yolo_anchors.txt'
    video_path = '/home/ship/troditional/g1.mp4'
    image_path = '/media/ship/文档/data/label_aut/image4/'
    label_path = '/media/ship/文档/data/label_aut/label4/'
    yolo = yolo_body(Input(shape=(None, None, 3)), 3, 2)
    yolo = YOLO(yolo, model_path, anchors_path, image_size=(320, 640))
    # detect_img(YOLO(yolo,model_path, anchors_path))
    # detect_video(YOLO(yolo,model_path, anchors_path,image_size=(320,640)),video_path)

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    while True:
        return_value, frame = vid.read()

        # image1.show()
        if cont % 5 == 0:
            image = Image.fromarray(frame)
            r, g, b = image.split()
            image1 = Image.merge('RGB', (b, g, r))
            image = Image.merge('RGB', (b, g, r))
            image1, label, o = yolo.detect_image(image1)
            image.save(image_path+str(cont)+'g.jpg')
            label_file = open(label_path+str(cont)+'g.txt', 'w')
            # label_file.write(' %s,%s,%s,%s,%s\n'%
            for i in label:
                label_file.write('%s %s %s %s %s\n' %
                                 (i[0], i[1], i[2], i[3], i[4]))
                # label_file.write('\n')
            label_file.close()
        cont += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    yolo.close_session()
