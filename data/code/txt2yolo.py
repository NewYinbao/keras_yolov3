# -- coding: utf-8 --
# # @Author: newyinbao
# @Date: 2019-09-21 21:52:32
# @Function: 标注数据集data 转 label.txt：图像地址 boxes(xmin,ymin,xmax,ymax,classes)  基于voc2label.py修改
# @TODO: 自动化生成 data 目录结构
# @Last Modified by:   newyinbao
# @Last Modified time: 2019-09-21 21:52:32


import random
from os import listdir, remove
from data_utils import convert_txt, write_to_file, clear_file, convert_xml

if __name__ == "__main__":

    dataset_path = 'data/data7.23/'

    # 图片标注的位置
    image_path = dataset_path + 'image/'
    label_path = dataset_path + 'label/'

    # 图片和标注的文件名
    image_names = listdir(image_path)
    label_names = listdir(label_path)

    data_size = len(label_names)
    n = 0

    train_percent = 0.9
    train_list = random.sample(image_names, int(data_size*train_percent))

    classes = ["box", "helmet", "qrcode"]

    train_file = dataset_path + 'train.txt'
    test_file = dataset_path + 'test.txt'
    all_file = dataset_path + 'all.txt'
    clear_file(test_file)
    clear_file(train_file)
    clear_file(all_file)

    for image_name in image_names:

        # .txt 标记文件
        if image_name[:-4] + '.txt' in label_names:
            label_name = image_name[:-4] + '.txt'
            annotation = convert_txt(
                image_path, image_name, label_path, label_name)
        # .xml 标记文件
        elif image_name[:-4] + '.xml' in label_names:
            label_name = image_name[:-4] + '.xml'
            annotation = convert_xml(label_path, label_name, classes)
        # 未标记文件
        else:
            remove(image_path+image_name)

        write_to_file(image_path+image_name, annotation, all_file)

        if image_name in train_list:
            write_to_file(image_path+image_name, annotation, train_file)
        else:
            write_to_file(image_path+image_name, annotation, test_file)
