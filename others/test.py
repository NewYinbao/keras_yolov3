# -*- coding: utf-8 -*-
"""
查看数据是否有问题
"""
import cv2
from yolo3.utils import get_random_data
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

#%%  test get data function
annotation_path = 'data/data8.4/all.txt'

with open(annotation_path) as f:
        lines = f.readlines()
        for line in lines:
            img, boxes = get_random_data(line,(224,320),random=1)
            cv2.imshow('imgr',img)
            '''
            img[...,0] = preprocessing.scale(img[...,0])
            img[...,1] = preprocessing.scale(img[...,1]) 
            img[...,2] = preprocessing.scale(img[...,2]) 
            '''
            # img = rgb_to_hsv(img)
            # img[...,2]=cv2.equalizeHist(img[...,2])
            # img[...,2]=exposure.equalize_hist(img[...,2])
            
            # img = hsv_to_rgb(img)
            img[...,[0,2]] = img[...,[2,0]]
            print("##########################")
            for box in boxes:
                
                if box[4] in (0,1,2):
                    print((int(box[0]),int(box[1]) ), (int(box[2]),int(box[3])),box)
                    cv2.rectangle(img, (int(box[0]),int(box[1]) ), (int(box[2]),int(box[3])),(0, 255, 0),1)
                
            cv2.imshow('img',img)
            
            if cv2.waitKey(0) & 0xFF == ord('q'):
                
                break