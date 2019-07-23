import os.path as osp
import os
import glob
import imagesize
import json
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import shutil

class Convertor(object):
    def __init__(self, root_dir, output_dir, source='drones', target='coco'):
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.source = source
        self.target = target

        self.splits = ['val']
        # self.splits = ['train', 'val', 'test']
        if source == 'drones' and target == 'coco':
            self.start = self.under2coco

    def load_drones(self):
        splits_names = {}
        for split in self.splits:
            img_path = osp.join(self.root_dir, split, 'images')
            image_names = glob.glob(osp.join(img_path, '*.jpg'))
            names = [x.split('\\')[-1].split('.')[0] for x in image_names]
            splits_names[split] = names
        return splits_names



    def under2coco(self):
        splits_names = self.load_drones()
        for split in self.splits:
            coco_json = {
                "info": "",
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": [
                    {
                        "id": 0,
                        "name": "ignore",
                        "supercategory": "",
                    },
                    {
                        "id": 1,
                        "name": "echinus",
                        "supercategory": "",
                    },
                    {
                        "id": 2,
                        "name": "starfish",
                        "supercategory": "",
                    },
                    {
                        "id": 3,
                        "name": "scallop",
                        "supercategory": "",
                    },
                    {
                        "id": 4,
                        "name": "holothurian",
                        "supercategory": "",
                    },
                    {
                        "id": 5,
                        "name": "waterweeds",
                        "supercategory": "",
                    }
                ]
            }

            names = splits_names[split]
            img_id = 0
            anno_id = 0
            for name in names:
                # dir = 'F:/dataset/水下目标抓取/under/'+ split + '/images/' + '{}.jpg'.format(name)
                # dir = 'F:/dataset/水下目标抓取/under/'+ split
                # dir = 'F:/dataset/水下目标抓取/' + '{}.jpg'.format(name.replace('\\', '/'))
                # print(dir)
                # width, height = imagesize.get(dir)


                img = cv2.imread(osp.join(self.root_dir, split, 'images', '{}.jpg'.format(name)))

                print(osp.join(self.root_dir, split, 'images', '{}.jpg'.format(name)))
                width, height = img.shape[0], img.shape[1]
                image = {
                    "license": 3,
                    "file_name": "{}.jpg".format(name),
                    "coco_url": "",
                    "height": height,
                    "width": width,
                    "date_captured": "",
                    "flickr_url": "",
                    "id": img_id
                }
                coco_json["images"].append(image)
                if split is not 'test':
                    with open(osp.join(self.root_dir, split, 'annotations', '{}.txt'.format(name)), 'r') as reader:
                        annos = reader.readlines()
                    for anno in annos:
                        anno = anno.strip().split(',')
                        x, y, w, h, score, cls, trc, occ = \
                            int(anno[0]), int(anno[1]), int(anno[2]), int(anno[3]), int(1), int(anno[5]), anno[6], anno[7]
                        annotation = {
                            "id": anno_id,
                            "image_id": img_id,
                            "category_id": cls,
                            "segmentation": [],
                            "area": (w-1)*(h-1),
                            "bbox": [x, y, w-1, h-1],
                            "iscrowd": 0,
                        }
                        coco_json["annotations"].append(annotation)
                        anno_id += 1
                else:
                    annotation = {
                        "id": anno_id,
                        "image_id": img_id,
                        "category_id": 0,
                        "segmentation": [],
                        "area": 0,
                        "bbox": [0, 0, 0, 0],
                        "iscrowd": 0,
                    }
                    coco_json["annotations"].append(annotation)
                    anno_id += 1
                img_id += 1

            with open(osp.join(self.output_dir, '{}.json'.format(split)), 'w') as outfile:
                json.dump(coco_json, outfile)


def xml2txt():
    i = 1
    classall = []
    echinus = 0
    starfish = 0
    scallop = 0
    holothurian = 0
    waterweeds = 0
    for sample in os.listdir('F:/dataset/train_part1/box/'):

        xml_file = 'F:/dataset/train_part1/box/' + sample
        txt_file = open('F:/dataset/train_part1/annotations/' + sample[:-4] + '.txt', 'w')
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
                    scallop += 1
                if cls == 'holothurian' or cls == 'seacucumber':
                    cls = int(4)
                    holothurian += 1
                if cls == 'waterweeds':
                    cls = int(5)
                    waterweeds += 1

                txt_file.write('{},{},{},{},{},{},1,1\n'.format(Xmin, Ymin, w, h, 1, cls))
                # print('cls:{}, Xmin:{}, Ymin:{}, Xmax:{}, Ymax:{}'.format(cls, Xmin, Ymin, Xmax, Ymax))
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

def splitdata():
    total = os.listdir('F:/dataset/oyster/images/')
    np.random.shuffle(total)
    train_part = total[0:276]
    val_part = total[276:]
    print(len(train_part))
    print(len(val_part))

    i = 0
    print(val_part)
    for val_name in val_part:
        shutil.copy('F:/dataset/oyster/images/' + val_name[:-4] + '.jpg', 'F:/dataset/oyster/val/images/')
        shutil.copy('F:/dataset/oyster/labels/' + val_name[:-4] + '.xml', 'F:/dataset/oyster/val/annotations/')
        i += 1
    print(i)
    for train_name in train_part:
        shutil.copy('F:/dataset/oyster/images/' + train_name[:-4] + '.jpg', 'F:/dataset/oyster/train/images/')
        shutil.copy('F:/dataset/oyster/labels/' + train_name[:-4] + '.xml', 'F:/dataset/oyster/train/annotations/')


def remove_empty_file():
    name1 = os.listdir('F:/dataset/undertotal/val3/annotations')
    for name in os.listdir('F:/dataset/undertotal/val3/images'):
        F=open('F:/dataset/undertotal/val3/annotations/'+name)
        if F.readline() == '':
            print(name)
            F.close()
            os.remove('F:/dataset/undertotal/val3/annotations/'+name)
            os.remove('F:/dataset/undertotal/val3/images/'+name[:-4]+'.jpg')
        print(name[:-4]+'.txt')
    print(name1)



def changeyolov3toRRnet():
    for label in os.listdir('F:/dataset/UNDERALL/mergelabel'):
        F1 = open('F:/dataset/UNDERALL/mergelabel/' + label, 'r')
        F2 = open('F:/dataset/UNDERALL/mergechangetxt/' + label, 'w')
        for line in F1.readlines():
            F2.write('{},{},{},{},{},{},{},{}\n'.format(line.split(' ')[1], line.split(' ')[2], int(line.split(' ')[3]) - int(line.split(' ')[1]), int(line.strip('\n').split(' ')[4]) - int(line.split(' ')[2]), 1, line.split(' ')[0], 0, 0))
            # print('{},{},{},{},{},{},{},{}\n'.format(line.split(' ')[1], line.split(' ')[2], line.split(' ')[3], line.strip('\n').split(' ')[4], 1, line.split(' ')[0], 0, 0))



if __name__ == '__main__':
    # convert xml to json

    convertor = Convertor('F:/dataset/UNDERALL/2018origin/', 'F:/dataset/UNDERALL/2018origin/')
    convertor.start()

    # convert xml to txt

    # xml2txt()

    # split dataset 0.2/0.8

    # splitdata()

    # changeyolov3toRRnet()
