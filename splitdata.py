import os
import numpy as np
import shutil
'''
total = os.listdir('F:/dataset/train_part1/image/')
np.random.shuffle(total)
val_part = total[0:740]
train_part = total[740:]

for val_name in val_part:
    shutil.copy('F:/dataset/train_part1/image/'+val_name[:-4]+'.jpg', 'F:/dataset/undertotal/val2/images')
    shutil.copy('F:/dataset/train_part1/annotations/'+val_name[:-4]+'.txt', 'F:/dataset/undertotal/val2/annotations')

for train_name in train_part:
    shutil.copy('F:/dataset/train_part1/image/'+train_name[:-4]+'.jpg', 'F:/dataset/undertotal/train/images')
    shutil.copy('F:/dataset/train_part1/annotations/'+train_name[:-4]+'.txt', 'F:/dataset/undertotal/train/annotations')
'''
for name in os.listdir('F:/dataset/undertotal/val3/annotations'):
    F=open('F:/dataset/undertotal/val3/annotations/'+name)
    if F.readline() == '':
        print(name)
        F.close()
        os.remove('F:/dataset/undertotal/val3/annotations/'+name)
        os.remove('F:/dataset/undertotal/val3/images/'+name[:-4]+'.jpg')