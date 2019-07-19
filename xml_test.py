import xml.etree.ElementTree as ET
import numpy as np
import os

i = 1
classall = []
echinus = 0
starfish = 0
scallop = 0
holothurian = 0
waterweeds = 0

for sample in os.listdir('F:/dataset/UnderWaterDetection[UPRC2018]/test/box/'):

    xml_file = 'F:/dataset/UnderWaterDetection[UPRC2018]/test/box/' + sample
    txt_file = open('F:/dataset/UnderWaterDetection[UPRC2018]/test/annotations/' + sample[:-4] + '.txt', 'w')
    try:
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
            if cls == 'echinus' or cls == 'seaurchin':
                cls = int(1)
                echinus += 1
            if cls == 'starfish':
                cls = int(2)
                starfish += 1
            if cls == 'scallop':
                cls = int(3)
                scallop +=1
            if cls == 'holothurian' or cls == 'seacucumber':
                cls = int(4)
                holothurian += 1
            if cls == 'waterweeds':
                cls = int(5)
                waterweeds += 1

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
    except:
        print(xml_file)


print(i)
print(classall)
print(echinus, starfish, scallop, holothurian, waterweeds)

