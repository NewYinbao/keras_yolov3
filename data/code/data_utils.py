# -- coding: utf-8 --
# # @Author: newyinbao
# @Date: 2019-09-21 22:07:18
# @Function: 标注数据整理工具
# @TODO:
# @Last Modified by:   newyinbao
# @Last Modified time: 2019-09-21 22:07:18


import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np

'''
for i in image_names:
    #设置旧文件名（就是路径+文件名）
    oldname=image_path+i
    #设置新文件名
    newname=image_path+str(n+1)+'.jpg'    
    #用os模块中的rename方法对文件改名
    os.rename(oldname,newname)
    n+=1
n=0
for i in label_names:
     #设置旧文件名（就是路径+文件名）
    oldname=label_path+i
    #设置新文件名
    newname=label_path+str(n+1)+'.xml'    
    #用os模块中的rename方法对文件改名
    os.rename(oldname,newname)
    n+=1
'''


def convert_xml(file_path, file_name, classes):
    '''将xml标注文件转化成yolo格式
        输入: 标注文件路径, 标注类别
        输出: 列表 boxes(xmin,ymin,xmax,ymax,classes)
    '''
    in_file = open(file_path + file_name)
    tree = ET.parse(in_file)
    root = tree.getroot()

    b = []
    for obj in root.iter('object'):

        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b.append((float(xmlbox.find('xmin').text),  float(xmlbox.find('ymin').text), float(
            xmlbox.find('xmax').text), float(xmlbox.find('ymax').text), cls_id))
    b = list(np.array(b, dtype=np.int))
    return b


def convert_txt(image_path, image_name, label_path, label_name):
    '''将txt标注文件转化成yolo格式
        输入: 图片路径, 标注文件路径
        输出: 列表 boxes(xmin,ymin,xmax,ymax,classes)
    '''
    with open(label_path + label_name) as f:
        lines = f.readlines()
    image = Image.open(image_path + image_name)
    b = []
    for obj in lines:
        obj = obj.split()
        cls_id = int(obj[0])
        x, y, w, h = map(float, obj[1:5])
        xmin = max(0, int((x-0.5*w)*image.width))
        xmax = min(int((x+0.5*w)*image.width), image.width)
        ymin = max(0, int((y-0.5*h)*image.height))
        ymax = min(int((y+0.5*h)*image.height), image.height)
        b.append([xmin, ymin, xmax, ymax, cls_id])
    return b


def write_to_file(image_name, annotation, tofile_path):
    '''标注写入.TXT文件
        输入: 图片路径image_name, 标注信息annotation, .txt文件路径'''
    with open(tofile_path, 'a') as tofile:
        tofile.write(image_name)
        for i in range(len(annotation)):
            tofile.write(' %s,%s,%s,%s,%s' % (
                annotation[i][0], annotation[i][1], annotation[i][2], annotation[i][3], annotation[i][4]))
        tofile.write('\n')


def clear_file(file_name):
    '''clear .txt fie'''
    with open(file_name, 'w') as tofile:
        tofile.write('')


def convert_br(old_path, new_path):
    '''bgr to rgb, rgb to bgr'''
    image = Image.open(old_path)
    r, g, b = image.split()
    image = Image.merge("RGB", [b, g, r])
    image.save(new_path, quality=100)
    return image
