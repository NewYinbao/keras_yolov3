# -- coding: utf-8 --
'''标注数据集data 转 label.txt：图像地址 boxes(xmin,ymin,xmax,ymax,classes)  基于voc2label.py修改'''
import random
from os import listdir
from code.utils import convert_xml, write_to_file


if __name__ == "__main__":

    dataset_path = 'data7.23/'

    # 图片标注的位置
    image_path = dataset_path + 'image/'
    label_path = dataset_path + 'label/'

    # 图片和标注的文件名
    image_names = listdir(image_path)
    label_names = listdir(label_path)

    data_size = len(label_names)
    n=0

    train_percent = 0.9
    train_list = random.sample(image_names,int(data_size*train_percent))

    classes = ["box","helmet"]

    train_file = dataset_path + 'train.txt'
    test_file = dataset_path + 'test.txt'
    all_file =dataset_path + 'all.txt'

    for image_name in image_names:
        
        label_name = image_name[:-4] + '.xml'
        # 未标记图片
        if label_name not in label_names:
            continue
    
        annotation = convert_xml(label_path, label_name, classes)

        write_to_file(image_path+image_name, annotation, all_file)

        if image_name in train_list:
            write_to_file(image_path+image_name, annotation, train_file)
        else:
            write_to_file(image_path+image_name, annotation, test_file)
 
