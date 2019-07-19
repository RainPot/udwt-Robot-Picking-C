import os
import shutil
import cv2
i = 0
'''
for videoname in os.listdir('F:/dataset/zhangzidao/val/image'):
    for index in os.listdir('F:/dataset/zhangzidao/val/image/'+videoname):
        for filename in os.listdir('F:/dataset/zhangzidao/val/image/'+videoname+'/'+index):
            print('F:/dataset/zhangzidao/val/image/'+videoname+'/'+index+'/'+filename)
            shutil.copy('F:/dataset/zhangzidao/val/image/'+videoname+'/'+index+'/'+filename, 'F:/dataset/zhangzidaonew/val/images')
            i += 1
print(i)
'''
for name in os.listdir('F:/dataset/undertotal/train/annotations'):
    F = open('F:/dataset/undertotal/train/annotations/'+name)
    img = cv2.imread('F:/dataset/undertotal/train/images/'+name[:-4]+'.jpg')
    height, width = img.shape[0], img.shape[1]
    for line in F.readlines():
        if (int(line.split(',')[0])+int(line.split(',')[2]))>=width or (int(line.split(',')[1])+int(line.split(',')[3]))>=height:
            print(name)