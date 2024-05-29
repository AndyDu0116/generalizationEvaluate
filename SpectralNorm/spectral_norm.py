import argparse
import torch
import os
import numpy as np
import time
import sys
from tqdm import tqdm
sys.path.append('..')
sys.path.append('.')
from utils import MyDataset, load_test_model
from resnet import resnet18, resnet34, resnet50, resnet101

def SVD_Conv_Tensor_NP(filter, inp_size):
    # compute the singular values using FFT
    # first compute the transforms for each pair of input and output channels
    transform_coeff = np.fft.fft2(filter, inp_size, axes=[0, 1])

    # now, for each transform coefficient, compute the singular values of the
    # matrix obtained by selecting that coefficient for
    # input-channel/output-channel pairs
    return np.linalg.svd(transform_coeff, compute_uv=False)

def getHW(name):
    if 'layer1' in name or 'layer2' in name:
        return 56, 56
    elif 'layer3' in name:
        return 28, 28
    elif 'layer4' in name:
        return 14, 14
    else:
        return 224, 224
    
def compute_spec_norm(model, margin):
    '''
    (Π || W_i ||_2 ^2 ) * (||W_j||_F ^2 / || W_j ||_2 ^2)  / margin^2
    
    '''
    assert margin != 0, "error margin == 0 !"
    mul_part1 = 1
    sum_part2 = 0

    for name in model.state_dict():
        if 'conv' in name:
            height, width = getHW(name)
            kernel = model.state_dict()[name].cpu().numpy()
            kernel = np.transpose(kernel, (2,3,1,0))

            norm_2 = SVD_Conv_Tensor_NP(kernel.copy(),[height, width])
            norm_2 = np.flip(np.sort(norm_2.flatten()),0)[0]
            norm_f = np.linalg.norm(kernel)

            tmp1 = norm_2**2
            tmp2 = norm_f**2

            #print('n2={}, nf={}, n22={}, nf2{}'.format(norm_2, norm_f, tmp1, tmp2))
            mul_part1 *= tmp1
            sum_part2 += (tmp2 / tmp1)
        
        elif 'fc.weight' in name:
            arr = model.state_dict()[name].cpu().numpy()
            norm_f = np.linalg.norm(arr)
            norm_2 = np.linalg.norm(arr,ord=2)

            tmp1 = norm_2**2
            tmp2 = norm_f**2
            #print('n2={}, nf={}, n22={}, nf2{}'.format(norm_2, norm_f, tmp1, tmp2))
            mul_part1 *= tmp1
            sum_part2 += (tmp2 / tmp1)
        
    
    return mul_part1 * sum_part2 / (margin * margin)

def val_margin(model, test_data_loader, batch_size, device):

    model.eval()
    correct_1 = 0
    test_data_loader_len = len(test_data_loader)
    margin = torch.Tensor([]).to(device)
    total_samples = test_data_loader_len * batch_size

    for idx, (image, label) in tqdm(enumerate(test_data_loader)):
        image = image.to(device)
        label = label.to(device)
        with torch.no_grad():
            pred_class = model(image)

        pred = pred_class.argmax(dim=1)
        correct = pred.eq(label).float()
        correct_1 += correct.sum()

        # compute the margin
        output_m = pred_class.clone()
        for i in range(label.size(0)):
            output_m[i, label[i]] = output_m[i,:].min()
        margin = torch.cat((margin, pred_class[:, label].diag() - output_m[:, output_m.max(1)[1]].diag()), 0)
    val_margin = np.percentile( margin.cpu().numpy(), 5)
    return val_margin

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-arch', default='ResNet18', type=str)
    parser.add_argument('--model-file', default='../dataExample/modelSet/ResNet18_example1.pth', type=str)
    parser.add_argument('--data-root', default='../dataExample/imageData', type=str)
    args = parser.parse_args()

    device = torch.device('cuda')

    trainset = MyDataset(data_dir=args.data_root,
                        phase='train',
                        classes_num=20,
                        data_aug=False,
                        input_h=224,
                        input_w=224)
    
    train_data_loader = torch.utils.data.DataLoader(trainset,
                                                    batch_size=32,
                                                    shuffle=False,
                                                    num_workers=0,
                                                    pin_memory=False,
                                                    drop_last=True)
    
    net = {
        'ResNet18': resnet18,
        'ResNet34': resnet34,
        'ResNet50': resnet50,
        'ResNet101': resnet101,
    }

    model = net[args.model_arch](pretrained=False, num_classes=20).to(device)
    model, train_acc, test_acc = load_test_model(model, args.model_file)
    print('computing margin....')
    margin = val_margin(model, train_data_loader, 32, device)
    print('val done, margin={}, time{}'.format(margin, time.strftime('%Y-%m-%d_%H-%M-%S')))
    print('computing norm...')
    model_spec_norm = compute_spec_norm(model, margin)
    print('model_spec_norm={}'.format(model_spec_norm))