# @Author: newyinbao
# @Date: 2019-09-25 22:42:04
# @Function: 修改视频尺寸
# @TODO: 
# @Last Modified by:   newyinbao
# @Last Modified time: 2019-09-25 22:42:04


import cv2

video_path = 'e:/data/video/f3.mp4'
out_path = 'e:/data/video/f3n.avi'
vid = cv2.VideoCapture(video_path)

video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
video_fps       = vid.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID') , 30, (960,720))
while(1):
    return_value, frame = vid.read()
    if not return_value:
        break
    frame = cv2.resize(frame,(960,720))
    out.write(frame)
    print('1')
out.release()