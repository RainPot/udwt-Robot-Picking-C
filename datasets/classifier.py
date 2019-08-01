from torch.utils.data import Dataset
from datasets.transforms import *
import pandas as pd
import os
from PIL import Image
import numpy as np


class classifier(Dataset):
    def __init__(self, root_dir, mode='train'):
        self.mode = mode
        self.images_path = os.path.join(root_dir, self.mode, 'images')
        self.annotations_path = os.path.join(root_dir, self.mode, 'annotations')
        self.dataname = os.listdir(self.images_path)

        if self.mode == 'train_2019':
            self.transforms = Compose([
                cropsingle(256, self.mode),
                ToTensor(),
                # HorizontalFlip(),
                Normalize()
            ])
        else:
            self.transforms = Compose([
                cropsingle(256, self.mode),
                ToTensor(),
                Normalize()
            ])

    def __len__(self):
        return len(self.dataname)

    def __getitem__(self, item):
        name = self.dataname[item][:-4]
        image_name = os.path.join(self.images_path, '{}.jpg'.format(name))
        annotations_name = os.path.join(self.annotations_path, '{}.txt'.format(name))
        image = Image.open(image_name).convert('RGB')

        annotation = pd.read_csv(annotations_name, header=None)
        annotation = np.array(annotation)[:, :8]

        sample = (image, annotation)
        sample = self.transforms(sample)

        return sample + (name,)


