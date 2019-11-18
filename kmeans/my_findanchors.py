# @Author: newyinbao
# @Date: 2019-09-21 21:01:05
# @Function: find YOLO anchors
# @TODO:
# @Last Modified by:   newyinbao
# @Last Modified time: 2019-09-21 21:01:05


import glob
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import matplotlib.pyplot as plot
from my_kmeans import kmeans, avg_iou


ANNOTATIONS_PATH = 'data/data8.23/all.txt'

CLUSTERS = 6


def load_txt(txt_file):
  '''load_txt: 从.txt文件中读取标签边界框的长宽
  
  Args:
      txt_file (Str): 标签文件地址
  
  Returns:
      list: 标签边界框的长宽相对于图片长宽的比值,(batch,x,y)
  '''
  b = []
  with open(txt_file) as f:
      lines = f.readlines()

  for line in lines:
      line = line.split()
      image = Image.open(line[0])
      box = np.array([np.array(list(map(float, box.split(','))))
                      for box in line[1:]])
      for i in box:
          xmin, ymin, xmax, ymax, c = i
          b.append([(xmax - xmin)/image.width, (ymax - ymin)/image.height])
  return np.array(b)


if __name__ == '__main__':

    height = 224
    width = 320
    data = load_txt(ANNOTATIONS_PATH)
    plot.scatter(data[:, 0], data[:, 1], s=10, c='blue')

    out = kmeans(data, k=CLUSTERS)
    plot.scatter(out[:, 0], out[:, 1], s=40, c='red')

    for i in range(CLUSTERS):
        print("{},{}, ".format(int(out[i, 0]*width), int(out[i, 1]*height)))
        
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))

    # 长宽比
    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))
    plot.show()
