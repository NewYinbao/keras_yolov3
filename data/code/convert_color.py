# @Author: newyinbao
# @Date: 2019-09-21 22:09:03
# @Function: 图片通道转换
# @TODO: 
# @Last Modified by:   newyinbao
# @Last Modified time: 2019-09-21 22:09:03


import os
import numpy as np
from utils import convert_br
if __name__ == "__main__":
    oldpath = '/media/ship/documents/8.231/3/'
    newpath = '/media/ship/documents/8.231/3/'
    images_name = os.listdir(oldpath)
    for img in images_name:
        convert_br(oldpath+img, newpath+img)
        print('success!' + ' from ' + oldpath+img+' save to '+ newpath)