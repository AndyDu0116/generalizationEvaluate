import torch.utils.data as data
import cv2
import torch
import numpy as np
from numpy import random
import os

class MyDataset_Rotation(data.Dataset):
    def __init__(self, data_dir, phase, classes_num, input_h=384, input_w=384):
        super(MyDataset_Rotation, self).__init__()
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

    def __len__(self):
        return len(self.img_id_label)

    def my_normlization(self, image):
        out_image = image.astype(np.float32) / 255.
        out_image = out_image - np.array([0.485,0.456,0.406])
        out_image = out_image / np.array([0.229,0.224,0.225])
        out_image = np.transpose(out_image,(2,0,1))

        return out_image

    def __getitem__(self, index):
        image, label = self.load_image_label(index)
        image = cv2.resize(image,(self.input_w, self.input_h))

        image = self.my_normlization(image)

        # rotation 90, 180, 270
        height, width = image.shape[:2]
        center = (width/2, height/2)

        rotate_matrix_90 = cv2.getRotationMatrix2D(center=center, angle=90, scale=1)
        rotate_matrix_180 = cv2.getRotationMatrix2D(center=center, angle=180, scale=1)
        rotate_matrix_270 = cv2.getRotationMatrix2D(center=center, angle=270, scale=1)

        # rotate the image using cv2.warpAffine
        rotated_image_90 = cv2.warpAffine(src=image, M=rotate_matrix_90, dsize=(width, height))
        rotated_image_180 = cv2.warpAffine(src=image, M=rotate_matrix_180, dsize=(width, height))
        rotated_image_270 = cv2.warpAffine(src=image, M=rotate_matrix_270, dsize=(width, height))

        src_image = torch.from_numpy(image).float()
        rotated_image_90 = torch.from_numpy(rotated_image_90).float()
        rotated_image_180 = torch.from_numpy(rotated_image_180).float()
        rotated_image_270 = torch.from_numpy(rotated_image_270).float()
        label = torch.tensor(label).long()

        return src_image, rotated_image_90, rotated_image_180, rotated_image_270, label



class MyDataset_Gray(data.Dataset):
    def __init__(self, data_dir, phase, classes_num, input_h=384, input_w=384):
        super(MyDataset_Gray, self).__init__()
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

    def __len__(self):
        return len(self.img_id_label)

    def my_normlization(self, image):
        out_image = image.astype(np.float32) / 255.
        out_image = out_image - np.array([0.5,0.5,0.5])
        #out_image = out_image - np.array([0.485,0.456,0.406])
        #out_image = out_image / np.array([0.229,0.224,0.225])
        out_image = np.transpose(out_image,(2,0,1))

        return out_image

    def __getitem__(self, index):
        image, label = self.load_image_label(index)
        image = cv2.resize(image,(self.input_w, self.input_h))
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray_3 = np.stack((image_gray, image_gray, image_gray), axis=-1)
        image = self.my_normlization(image)
        image_gray_3 = self.my_normlization(image_gray_3)
        

        src_image = torch.from_numpy(image).float()
        gray_image = torch.from_numpy(image_gray_3).float()
        label = torch.tensor(label).long()

        return src_image, gray_image, label

