import torch
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from nuclear_norm_dataloader import NuclearNormDataset
import sys
sys.path.append('..')
from resnet import resnet18, resnet34, resnet50, resnet101
from utils import load_test_model



def get_softmax_matrix(model_name, model_file, dataloader):

    device = torch.device('cuda')
    input_h = 224
    input_w = 224

    net = {
        'ResNet18': resnet18,
        'ResNet34': resnet34,
        'ResNet50': resnet50,
        'ResNet101': resnet101,
    }
    model = net[model_name](pretrained=False, num_classes=20).to(device)
    model, train_acc, test_acc = load_test_model(model, model_file)
    
    model.eval()
    softmax_matrix_ls = []
    
    for idx, (src_image, label) in tqdm(enumerate(dataloader)):
        src_image = src_image.to(device)
        with torch.no_grad():
            pred_label = model(src_image)
        
        pred_label = torch.softmax(pred_label, dim=1)
        softmax_matrix_ls.append(pred_label)
    return train_acc, test_acc, torch.vstack(softmax_matrix_ls)

if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-arch', default='ResNet18', type=str)
    parser.add_argument('--model-file', default='../dataExample/modelSet/ResNet18_example1.pth', type=str)
    parser.add_argument('--data-root', default='../dataExample/imageData', type=str)
    args = parser.parse_args()

    trainset = NuclearNormDataset(data_dir=args.data_root,
                        phase='train',
                        classes_num=20,
                        data_aug=False,
                        input_h=224,
                        input_w=224)

    train_data_loader = torch.utils.data.DataLoader(trainset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=0,
                                                    pin_memory=False,
                                                    drop_last=False,
                                                    )
    
    testset = NuclearNormDataset(data_dir=args.data_root,
                        phase='test',
                        classes_num=20,
                        input_h=224,
                        input_w=224)

    test_data_loader = torch.utils.data.DataLoader(testset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=0,
                                                    pin_memory=False,
                                                    drop_last=False,
                                                    )

    data_loader = train_data_loader

    train_acc, test_acc, softmax_matrix = get_softmax_matrix(args.model_arch, args.model_file, data_loader)
    cur_nuclear_norm = torch.norm(softmax_matrix, p='nuc').item()
        
    print(f'model : {args.model_file}')
    print(f'train_acc = {train_acc}, test_acc = {test_acc}')
    print(f'Nuclear norm score = {cur_nuclear_norm}')

