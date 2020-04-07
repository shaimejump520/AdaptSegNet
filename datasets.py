import numpy as np
import torch
import torchvision
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from PIL import Image
import scipy
import sys
import argparse

from PIL import Image, ImageEnhance, ImageOps

ignore_label = 255
id2label = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
            3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
            7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
            14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
            18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
            28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
            220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
            0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
        'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle']
num_cls = len(classes)
LABEL2TRAIN = id2label

import os
from scipy import io

class GTA5(torch.utils.data.Dataset):

    def __init__(self, root, num_cls=19, split='train', remap_labels=True, 
            transform=None, target_transform=None, scale=None, crop_transform=None):
        self.root = root
        self.split = split
        self.remap_labels = remap_labels
        self.ids = self.collect_ids()
        self.transform = transform
        self.target_transform = target_transform
        self.scale = scale
        self.crop_transform = crop_transform
        m = io.loadmat(os.path.join(self.root, 'mapping.mat'))
        full_classes = [x[0] for x in m['classes'][0]]
        self.classes = []
        for old_id, new_id in LABEL2TRAIN.items():
            if not new_id == 255 and old_id > 0:
                self.classes.append(full_classes[old_id])
        self.num_cls = 19

    
    def collect_ids(self):
        splits = io.loadmat(os.path.join(self.root, 'split.mat'))
        delete_list = np.array(list(range(20801, 20861)) + [15188, 17705])
        if self.split == 'all':
            ids = list(range(1, 24967))
        else:
            ids = splits['{}Ids'.format(self.split)].squeeze()
        ids = np.setdiff1d(ids, delete_list)
        return ids

    def img_path(self, id):
        filename = '{:05d}.png'.format(id)
        return os.path.join(self.root, 'images', filename)

    def label_path(self, id):
        filename = '{:05d}.png'.format(id)
        return os.path.join(self.root, 'labels', filename)

    def __getitem__(self, index):
        id = self.ids[index]
        img_path = self.img_path(id)
        label_path = self.label_path(id)
        img = Image.open(img_path).convert('RGB')
        
        tar = Image.open(label_path)
        if self.remap_labels:
            tar = np.asarray(tar)
            tar = remap_labels_to_train_ids(tar)
            tar = Image.fromarray(tar, 'L')
            
        if img.size != tar.size:
            print('img', img.size)
            print('tar', tar.size)
            print(self.label_path(id))

        if self.scale != None:
            img = img.resize(self.scale, Image.BICUBIC)
            tar = tar.resize(self.scale, Image.NEAREST)
        
        if self.crop_transform is not None:
            img, tar = self.crop_transform([img, tar])
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            tar = self.target_transform(tar)
        
        img = np.asarray(img, np.float32)
#         img = img[:, :, ::-1]  # change to BGR
        img = img.copy()
        tar = np.asarray(tar, np.float32)
        
        return img, tar

    def __len__(self):
        return len(self.ids)


def remap_labels_to_train_ids(arr):
    out = ignore_label * np.ones(arr.shape, dtype=np.uint8)
    for id, label in id2label.items():
        out[arr == id] = int(label)
    return out

class Cityscapes(torch.utils.data.Dataset):

    def __init__(self, root, split='train', remap_labels=True, transform=None,
                 target_transform=None, scale=None, crop_transform=None):
        self.root = root
        sys.path.append(root)
        self.split = split
        self.remap_labels = remap_labels
        self.ids = self.collect_ids()
        self.transform = transform
        self.target_transform = target_transform
        self.scale = scale
        self.crop_transform = crop_transform
        self.num_cls = 19
        
        self.id2label = id2label
        self.classes = classes

    def collect_ids(self):
        im_dir = os.path.join(self.root, 'leftImg8bit', self.split)
        ids = []
        for dirpath, dirnames, filenames in os.walk(im_dir):
            for filename in filenames:
                if filename.endswith('.png'):
                    ids.append('_'.join(filename.split('_')[:3]))
        return ids

    def img_path(self, id):
        fmt = 'leftImg8bit/{}/{}/{}_leftImg8bit.png'
        subdir = id.split('_')[0]
        path = fmt.format(self.split, subdir, id)
        return os.path.join(self.root, path)

    def label_path(self, id):
        if self.split == 'train_extra':
            fmt = 'gtCoarse/{}/{}/{}_gtCoarse_labelIds.png'
        else:  
            fmt = 'gtFine/{}/{}/{}_gtFine_labelIds.png'
        subdir = id.split('_')[0]
        path = fmt.format(self.split, subdir, id)
        return os.path.join(self.root, path)

    def __getitem__(self, index):
        
        id = self.ids[index]
        img = Image.open(self.img_path(id)).convert('RGB')
#         aug_img = policy(img, rand_p1, rand_p2)
       
        tar = Image.open(self.label_path(id)).convert('L')
        if self.remap_labels:
            tar = np.asarray(tar)
            tar = remap_labels_to_train_ids(tar)
            tar = Image.fromarray(np.uint8(tar), 'L')
            
        if img.size != tar.size:
            print('img', img.size)
            print('tar', tar.size)
            print(self.label_path(id))

        if self.scale != None:
            img = img.resize(self.scale, Image.BICUBIC)
            tar = tar.resize(self.scale, Image.NEAREST)
        
        if self.crop_transform is not None:
            img, tar = self.crop_transform([img, tar])
            
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            tar = self.target_transform(tar)
            
        img = np.asarray(img, np.float32)
#         img = img[:, :, ::-1]  # change to BGR
        img = img.copy()
        tar = np.asarray(tar, np.float32)
        
        return img, tar

    def __len__(self):
        return len(self.ids)

def print_palette(label_img):
    converted = Image.new('L', label_img.size)
    converted.putpalette(palette)
    converted.paste(label_img, (0, 0))
    return converted