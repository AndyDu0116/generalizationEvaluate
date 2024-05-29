import torch
import numpy as np
import os
import sys
from EI_dataloader import MyDataset_Gray
from tqdm import tqdm
import argparse
sys.path.append('..')
from resnet import resnet18, resnet34, resnet50, resnet101
from utils import load_test_model

def get_ei_gray(model_name, model_file, dataloader):
    device = torch.device('cuda')
    net = {
        'ResNet18': resnet18,
        'ResNet34': resnet34,
        'ResNet50': resnet50,
        'ResNet101': resnet101,
    }
    model = net[model_name](pretrained=False, num_classes=20).to(device)
    model, train_acc, test_acc = load_test_model(model, model_file)
    
    model.eval()
    ei_gray_ls = []
    for idx, (src_image, gray_image, label) in tqdm(enumerate(dataloader)):
        src_image = src_image.to(device)
        gray_image = gray_image.to(device)

        with torch.no_grad():
            pred_label = model(src_image)
            pred_label_gray = model(gray_image)
        pred_label = torch.softmax(pred_label,dim=1)
        pred_label_gray = torch.softmax(pred_label_gray,dim=1)
        
        pred_cls = pred_label.argmax(dim=1)
        pred_cls_gray = pred_label_gray.argmax(dim=1)
        
        tmp = (pred_label * pred_label_gray).sum(dim=1)
        tmp = torch.sqrt(tmp)
        tmp[pred_cls != pred_cls_gray] = 0
        tmp = tmp.cpu().numpy()
        tmp[tmp!=tmp] = 0

        ei_gray = np.mean(tmp, axis=0)
        
        #ei_gray = ei_gray.cpu().numpy()
        
        #print('90={}, 180={}, 270={},avg={}'.format(ei_90, ei_180, ei_270, (ei_90+ei_180+ei_270)/3))
        ei_gray_ls.append([ei_gray])
        
    return train_acc, test_acc, ei_gray_ls



if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-arch', default='ResNet18', type=str)
    parser.add_argument('--model-file', default='../dataExample/modelSet/ResNet18_example1.pth', type=str)
    parser.add_argument('--data-root', default='../dataExample/imageData', type=str)
    args = parser.parse_args()
    trainset = MyDataset_Gray(data_dir=args.data_root,
                        phase='train',
                        classes_num=20,
                        input_h=224,
                        input_w=224)

    train_data_loader = torch.utils.data.DataLoader(trainset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=0, #if error, try 0
                                                    pin_memory=False,
                                                    drop_last=True,
                                                    )
    
    testset = MyDataset_Gray(data_dir=args.data_root,
                        phase='test',
                        classes_num=20,
                        input_h=224,
                        input_w=224)

    test_data_loader = torch.utils.data.DataLoader(testset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=1,
                                                    pin_memory=False,
                                                    drop_last=True,
                                                    )
    # use train data to predict performance on testset (generalization)
    data_loader = train_data_loader
    
    train_acc, test_acc, ei_ls = get_ei_gray(args.model_arch, args.model_file, data_loader)
    ei_score_gray = np.mean(ei_ls,axis=0)[0]
    print(f'model : {args.model_file}')
    print(f'train_acc = {train_acc}, test_acc = {test_acc}')
    print(f'EI score = {ei_score_gray}')
