import torch
import torch.utils.data as data
import cv2
import numpy as np
from numpy import random
import os

class MyDataset(data.Dataset):
    def __init__(self, data_dir, phase, classes_num, data_aug=True, input_h=384, input_w=384):
        super(MyDataset, self).__init__()
        self.num_classes = classes_num
        self.data_dir = data_dir
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.data_aug = data_aug
        self.img_id_label = self.load_img_id_label()
        self.image_distort =  PhotometricDistort()

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

    def __getitem__(self, index):
        image, label = self.load_image_label(index)
        if self.phase == 'train' and self.data_aug:
            image = random_flip(image)
            image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)
            image = self.image_distort(image)
            image = np.asarray(np.clip(image, a_min=0., a_max=255.), np.float32)

        image = cv2.resize(image,(self.input_w, self.input_h))
        #out_image = np.transpose(image / 255. - 0.5, (2, 0, 1))

        out_image = image.astype(np.float32) / 255.
        out_image = out_image - np.array([0.485,0.456,0.406])
        out_image = out_image / np.array([0.229,0.224,0.225])
        out_image = np.transpose(out_image,(2,0,1))

        out_image = torch.from_numpy(out_image).float()
        label = torch.tensor(label).long()

        return out_image , label


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            img *= alpha
        return img

class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, img):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            img += delta
        return img

class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps
    def __call__(self, img):
        img = img[:, :, self.swaps]
        return img

class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
    def __call__(self, img):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            img = shuffle(img)
        return img

class PhotometricDistort(object):
    def __init__(self):
        self.pd = RandomContrast()
        self.rb = RandomBrightness()
        self.rln = RandomLightingNoise()

    def __call__(self, img):
        img = self.rb(img)
        if random.randint(2):
            distort = self.pd
        else:
            distort = self.pd
        img = distort(img)
        img = self.rln(img)
        return img

def random_flip(image):
    # left <---> right flip
    if np.random.random()<0.5:
        image = image[:,::-1,:]
    #up <---> down flip
    if np.random.random()<0.5:
        image = image[::-1,:,:]
        
    return image

def load_test_model(model, resume, strict=True):
    checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
    print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
    state_dict_ = checkpoint['model_state_dict']
    state_dict = {}
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()
    if not strict:
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, ' \
                            'loaded shape{}.'.format(k, model_state_dict[k].shape, state_dict[k].shape))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k))
        for k in model_state_dict:
            if not (k in state_dict):
                print('No param {}.'.format(k))
                state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    train_acc = checkpoint['train_acc']
    test_acc = checkpoint['test_acc']

    return model, train_acc, test_acc