import argparse
import numpy as np
import torch
from torchvision.transforms import Resize
from dataloaderRFRM import MyDatasetRFRM
from tqdm import tqdm
import sys
from torchcam.methods import SmoothGradCAMpp, GradCAM, GradCAMpp, LayerCAM
sys.path.append('..')
from resnet import resnet18, resnet34, resnet50, resnet101
from utils import load_test_model


device = torch.device('cuda')
RFRMR_THRESHOLD = 0.1
CAM_MRTHOD = {
    'gradcam' : GradCAM, 
    'gradcampp' : GradCAMpp,
    'smoothgradcampp' : SmoothGradCAMpp,
    'layercam' : LayerCAM
}

def get_RFRMr(model, data_loader, cam_method, threshold=0.1):
    assert cam_method in CAM_MRTHOD, f'{cam_method} not support.'

    Tensor_Resize = Resize([224, 224])
    model.eval()

    rfrm_ls = []
    cam = CAM_MRTHOD[cam_method](model, 'layer4')
    for idx, (image, cls_label, seg_map) in tqdm(enumerate(data_loader)):
        image = image.to(device)
        seg_map = seg_map.to(device)
        out = model(image)
        pred_label = out.argmax(dim=1)
        heat_map = cam(class_idx=pred_label.item(), scores=out)
        if cam_method == 'layercam':
            heat_map = cam.fuse_cams(heat_map)
        else:
            heat_map = heat_map[0]
        heat_map = Tensor_Resize(heat_map)

        heat_map[heat_map >= threshold] = 1
        heat_map[heat_map < threshold] = 0
        heat_map = heat_map.bool()

        task_relevant = (heat_map & seg_map).sum().float().item()
        total = (heat_map | seg_map).sum().float().item()

        rfrm = task_relevant / total if total > 0 else 1
        rfrm_ls.append(rfrm)

    return np.mean(rfrm_ls)


if __name__ =='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-arch', default='ResNet18', type=str)
    parser.add_argument('--model-file', default='../dataExample/modelSet/ResNet18_example1.pth', type=str)
    parser.add_argument('--data-root', default='../dataExample/imageData', type=str)
    parser.add_argument('--cam', default='gradcam', choices=['gradcam', 'gradcampp','smoothgradcampp', 'layercam'])
    args = parser.parse_args()

    trainset = MyDatasetRFRM(data_dir=args.data_root,
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
    
    net = {
        'ResNet18': resnet18,
        'ResNet34': resnet34,
        'ResNet50': resnet50,
        'ResNet101': resnet101,
    }

    model = net[args.model_arch](pretrained=False, num_classes=20).to(device)
    model, train_acc, test_acc = load_test_model(model, args.model_file)

    rfrm = get_RFRMr(model, train_data_loader, args.cam, threshold=RFRMR_THRESHOLD)

    

    print(f'model : {args.model_file}')
    print(f'train_acc = {train_acc}, test_acc = {test_acc}')
    print(f'rfrm score = {rfrm}')