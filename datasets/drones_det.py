import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import re
from datasets.transforms import *
import xml.etree.ElementTree as ET
import cv2


class DronesDET(Dataset):
    def __init__(self, root_dir, transforms=None, split='train'):
        '''
        :param root_dir: root of annotations and image dirs
        :param transform: Optional transform to be applied
                on a sample.
        '''
        # get the csv
        self.images_dir = os.path.join(root_dir, 'images')
        self.annotations_dir = os.path.join(root_dir, 'annotations')
        mdf = os.listdir(self.images_dir)
        restr = r'\w+?(?=(.jpg))'
        for index, mm in enumerate(mdf):
            mdf[index] = re.match(restr, mm).group()
        self.mdf = mdf
        self.transforms = transforms

    def __len__(self):
        return len(self.mdf)

    def __getitem__(self, item):
        name = self.mdf[item]
        img_name = os.path.join(self.images_dir, '{}.jpg'.format(name))
        txt_name = os.path.join(self.annotations_dir, '{}.xml'.format(name))
        # txt_name = os.path.join('./test.txt')
        # read image
        image = Image.open(img_name).convert("RGB")

        tree = ET.parse(txt_name)
        root = tree.getroot()
        annotation = []
        for object in root.findall('object'):
            clsname = object.find('name').text
            Xmin = float(object.find('bndbox').find('xmin').text)
            Ymin = float(object.find('bndbox').find('ymin').text)
            Xmax = float(object.find('bndbox').find('xmax').text)
            Ymax = float(object.find('bndbox').find('ymax').text)
            W = Xmax - Xmin
            H = Ymax - Ymin
            if clsname == 'echinus':
                cls = float(0)
            if clsname == 'starfish':
                cls = float(1)
            if clsname == 'scallop':
                cls = float(2)
            if clsname == 'holothurian':
                cls = float(3)
            if clsname == 'waterweeds':
                cls = float(4)
            score = float(1)
            single = [Xmin, Ymin, W, H, score, cls]
            annotation.append(single)
        annotation = np.array(annotation)

        # read road segmentation

        sample = (image, annotation)

        if self.transforms:
            sample = self.transforms(sample)
        return sample + (name,)

    @staticmethod
    def collate_fn(batch):
        max_n = 0
        for i, batch_data in enumerate(batch):
            max_n = max(max_n, batch_data[1].size(0))
        imgs, annos, names = [], torch.zeros(len(batch), max_n, 8), []
        for i, batch_data in enumerate(batch):
            imgs.append(batch_data[0].unsqueeze(0))
            annos[i, :batch_data[1].size(0), :] = batch_data[1][:, :8]
            names.append(batch_data[2])
        imgs = torch.cat(imgs)
        return imgs, annos, names

    @staticmethod
    def collate_fn_ctnet(batch):
        max_n = 0
        for i, batch_data in enumerate(batch):
            max_n = max(max_n, batch_data[1].size(0))
        imgs, hms, names = [], [], []
        batchsize = len(batch)
        annos, whs, offsets, inds, reg_masks = \
            torch.zeros(batchsize, max_n, 6), \
            torch.zeros(batchsize, max_n, 2), \
            torch.zeros(batchsize, max_n, 2), \
            torch.zeros(batchsize, max_n, 1), \
            torch.zeros(batchsize, max_n, 1)

        for i, batch_data in enumerate(batch):
            imgs.append(batch_data[0].unsqueeze(0))
            annos[i, :batch_data[1].size(0), :] = batch_data[1][:, :6]
            hms.append(batch_data[2].unsqueeze(0))
            whs[i, :batch_data[3].size(0), :] = batch_data[3]
            inds[i, :batch_data[4].size(0), :] = batch_data[4]
            offsets[i, :batch_data[5].size(0), :] = batch_data[5]
            reg_masks[i, :batch_data[6].size(0), :] = batch_data[6]
            names.append(batch_data[7])
        imgs = torch.cat(imgs)
        hms = torch.cat(hms)
        return imgs, annos, hms, whs, inds, offsets, reg_masks, names
