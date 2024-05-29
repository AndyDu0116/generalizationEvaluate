import torch
import numpy as np
from EI_dataloader import MyDataset_Rotation
from tqdm import tqdm
import argparse
import sys
sys.path.append('..')
from resnet import resnet18, resnet34, resnet50, resnet101
from utils import load_test_model


def get_ei_rotate_batch(model_name, model_file, dataloader):

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
    ei_rotate_ls = []
    for idx, (src_image, rotated_image_90, rotated_image_180, rotated_image_270, label) in tqdm(enumerate(dataloader)):
        src_image = src_image.to(device)
        rotated_image_90 = rotated_image_90.to(device)
        rotated_image_180 = rotated_image_180.to(device)
        rotated_image_270 = rotated_image_270.to(device)

        with torch.no_grad():
            pred_label = model(src_image)
            pred_label_90 = model(rotated_image_90)
            pred_label_180 = model(rotated_image_180)
            pred_label_270 = model(rotated_image_270)
        pred_label = torch.softmax(pred_label,dim=1)
        pred_label_90 = torch.softmax(pred_label_90,dim=1)
        pred_label_180 = torch.softmax(pred_label_180,dim=1)
        pred_label_270 = torch.softmax(pred_label_270,dim=1)

        pred_cls = pred_label.argmax(dim=1)
        pred_cls_90 = pred_label_90.argmax(dim=1)
        pred_cls_180 = pred_label_180.argmax(dim=1)
        pred_cls_270 = pred_label_270.argmax(dim=1)

        def get_ei_value_batch(pred_label, pred_label_rotate, pred_cls, pre_cls_rotate):
            tmp = (pred_label * pred_label_rotate).sum(dim=1)
            tmp = torch.sqrt(tmp)
            tmp[pred_cls != pre_cls_rotate] = 0
            tmp = tmp.cpu().numpy()
            tmp[tmp!=tmp] = 0
            ei_value = np.mean(tmp, axis=0)
            return ei_value
        
        ei_90 = get_ei_value_batch(pred_label, pred_label_90, pred_cls, pred_cls_90)
        ei_180 = get_ei_value_batch(pred_label, pred_label_180, pred_cls, pred_cls_180)
        ei_270 = get_ei_value_batch(pred_label, pred_label_270, pred_cls, pred_cls_270)

        #print('90={}, 180={}, 270={},avg={}'.format(ei_90, ei_180, ei_270, (ei_90+ei_180+ei_270)/3))
        ei_rotate_ls.append([ei_90, ei_180, ei_270, (ei_90+ei_180+ei_270)/3])
        
    return train_acc, test_acc, ei_rotate_ls


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-arch', default='ResNet18', type=str)
    parser.add_argument('--model-file', default='../dataExample/modelSet/ResNet18_example1.pth', type=str)
    parser.add_argument('--data-root', default='../dataExample/imageData', type=str)
    args = parser.parse_args()

    trainset = MyDataset_Rotation(data_dir=args.data_root,
                        phase='train',
                        classes_num=20,
                        input_h=224,
                        input_w=224)

    train_data_loader = torch.utils.data.DataLoader(trainset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=0,
                                                    pin_memory=False,
                                                    drop_last=True,
                                                    )
    
    testset = MyDataset_Rotation(data_dir=args.data_root,
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

    data_loader = train_data_loader
    
    train_acc, test_acc, ei_ls = get_ei_rotate_batch(args.model_arch, args.model_file, data_loader)
    ei_score_rotation = np.mean(ei_ls,axis=0)[3]
    print(f'model : {args.model_file}')
    print(f'train_acc = {train_acc}, test_acc = {test_acc}')
    print(f'EI score = {ei_score_rotation}')
    
