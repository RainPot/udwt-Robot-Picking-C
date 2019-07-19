import os
import shutil
i = 0
for videoname in os.listdir('F:/dataset/zhangzidao/val/image'):
    for index in os.listdir('F:/dataset/zhangzidao/val/image/'+videoname):
        for filename in os.listdir('F:/dataset/zhangzidao/val/image/'+videoname+'/'+index):
            print('F:/dataset/zhangzidao/val/image/'+videoname+'/'+index+'/'+filename)
            shutil.copy('F:/dataset/zhangzidao/val/image/'+videoname+'/'+index+'/'+filename, 'F:/dataset/zhangzidaonew/val/images')
            i += 1
print(i)