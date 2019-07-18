import cv2
import os

fps = 15   #视频帧率
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
videoWriter = cv2.VideoWriter('F:/models/VisDronevideo/video7.avi', fourcc, fps, (2720,1530))   #(1360,480)为视频大小
for i in os.listdir('F:/models/VisDronevideo/video7/'):
    img12 = cv2.imread('F:/models/VisDronevideo/video7/' + i)
#    cv2.imshow('img', img12)
#    cv2.waitKey(1000/int(fps))
    videoWriter.write(img12)
videoWriter.release()
