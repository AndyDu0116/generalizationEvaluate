import torch.utils.data as data
import cv2
import torch
import numpy as np
from numpy import random
import os
import xml.etree.ElementTree as ET

class MyDatasetRFRM(data.Dataset):
    def __init__(self, data_dir, phase, classes_num, input_h=384, input_w=384):
        super(MyDatasetRFRM, self).__init__()
        self.num_classes = classes_num
        self.data_dir = data_dir
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.img_id_label = self.load_img_id_label()

    def load_img_id_label(self):
        image_set_index_file = os.path.join(self.data_dir, '%s.txt'%self.phase)
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file, 'r') as f:
            lines = f.readlines()
        image_lists = [line.strip() for line in lines]
        return image_lists

    def load_image_label(self, index):
        img_id, label = self.img_id_label[index].split(' ')
        imgFile = os.path.join(self.data_dir, 'image/%s'%img_id)
        assert os.path.exists(imgFile), 'image {} not existed'.format(imgFile)
        img = cv2.imread(imgFile)
        return img, int(label)

    def get_seg_map(self, index, input_w, input_h):
        img_id = self.img_id_label[index].split(' ')[0]
        anno_file = os.path.join(self.data_dir, 'annotations/%sxml'%img_id[:-4])
        assert os.path.exists(anno_file), 'annotation {} not existed'.format(anno_file)
        seg_map = np.zeros((input_h, input_w))

        anno_file = open(anno_file)
        tree = ET.parse(anno_file)
        root = tree.getroot()
        size = root.find('size')
        src_w = float(size.find('width').text)
        src_h = float(size.find('height').text)
        w_scale = input_w / src_w
        h_scale = input_h / src_h

        for obj in root.iter('object'):
            hbox = obj.find('bndbox')
            x_max = int(float(hbox.find('xmax').text) * w_scale)
            x_min = int(float(hbox.find('xmin').text) * w_scale)
            y_max = int(float(hbox.find('ymax').text) * h_scale)
            y_min = int(float(hbox.find('ymin').text) * h_scale)
            seg_map[y_min:y_max, x_min:x_max] = 1
        
        return seg_map

    def __len__(self):
        return len(self.img_id_label)

    def __getitem__(self, index):
        image, label = self.load_image_label(index)
        seg_map = self.get_seg_map(index, self.input_w, self.input_h)
        image = cv2.resize(image,(self.input_w, self.input_h))
        #out_image = np.transpose(image / 255. - 0.5, (2, 0, 1))

        out_image = image.astype(np.float32) / 255.
        out_image = out_image - np.array([0.485,0.456,0.406])
        out_image = out_image / np.array([0.229,0.224,0.225])
        out_image = np.transpose(out_image,(2,0,1))

        out_image = torch.from_numpy(out_image).float()
        label = torch.tensor(label).long()
        seg_map = torch.from_numpy(seg_map).bool()

        return out_image, label, seg_map