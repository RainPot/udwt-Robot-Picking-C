import xml.etree.ElementTree as ET
import numpy as np
import os

i = 1
classall = []
for sample in os.listdir('F:/dataset/水下目标抓取/train_part1/train/box/'):

    xml_file = 'F:/dataset/水下目标抓取/train_part1/train/box/' + sample
    txt_file = open('F:/dataset/水下目标抓取/train_part1/train/bboxtxt/' + sample[:-4] + '.txt', 'w')
    tree = ET.parse(xml_file)
    root = tree.getroot()
    img_name = root.find('frame').text
    # print(img_name)
    for object in root.findall('object'):
        cls = object.find('name').text
        Xmin = int(object.find('bndbox').find('xmin').text)
        Ymin = int(object.find('bndbox').find('ymin').text)
        Xmax = int(object.find('bndbox').find('xmax').text)
        Ymax = int(object.find('bndbox').find('ymax').text)
        w = Xmax - Xmin
        h = Ymax - Ymin
        if cls == 'echinus':
            cls = int(1)
        if cls == 'starfish':
            cls = int(2)
        if cls == 'scallop':
            cls = int(3)
        if cls == 'holothurian':
            cls = int(4)
        if cls == 'waterweeds':
            cls = int(5)
        txt_file.write('{},{},{},{},{},{},1,1\n'.format(Xmin, Ymin, w, h, 1, cls))
        #print('cls:{}, Xmin:{}, Ymin:{}, Xmax:{}, Ymax:{}'.format(cls, Xmin, Ymin, Xmax, Ymax))
        if cls not in classall:
            classall.append(cls)
    # a = np.empty([1, 5])
    # b = np.zeros([1, 5])
    # a = np.append(a, b, axis=0)
    # print(a)
    # print(a.shape)
    i += 1

print(i)
print(classall)

