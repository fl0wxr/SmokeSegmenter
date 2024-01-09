# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------

import sys

sys.path.append('../../data_tools/src/utils')

import visuals
from copy import deepcopy
import glob
import argparse
import cv2
import os
import numpy as np
import _init_paths
import models
import torch
import torch.nn.functional as F
from PIL import Image
from time import time

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

color_map = [(0, 0,0),
             (1, 1, 1)]

def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')
    
    parser.add_argument('--a', help='pidnet-s, pidnet-m or pidnet-l', default='pidnet-l', type=str)
    parser.add_argument('--p', help='dir for pretrained model', default='../pretrained_models/cityscapes/PIDNet_L_Cityscapes_test.pt', type=str)
    parser.add_argument('--r', help='root or dir for input images', default='../samples/', type=str)

    args = parser.parse_args()

    return args

def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image

def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict = False)
    
    return model

if __name__ == '__main__':
    args = parse_args()
    images_list = glob.glob(args.r+'/*')
    sv_path = './predicted_results'
    
    model = models.pidnet.get_pred_model(args.a, 2) ## Number of classes is 2
    model = load_pretrained(model, args.p).cuda()
    model.eval()
    fps_ = []
    with torch.no_grad():
        for img_path in images_list:


            img_name = img_path.split('/')[-1]
            img = cv2.imread(os.path.join(args.r, img_name),
                               cv2.IMREAD_COLOR)

            img_ = deepcopy(img)

            t0 = time()

            sv_img = np.zeros_like(img).astype(np.uint8)
            img = input_transform(img)
            img = img.transpose((2, 0, 1)).copy()
            img = torch.from_numpy(img).unsqueeze(0).cuda()
            pred = model(img)
            pred = F.interpolate(pred, size=img.size()[-2:], 
                                 mode='bilinear', align_corners=True)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            
            for i, color in enumerate(color_map):
                for j in range(3):
                    sv_img[:,:,j][pred==i] = color_map[i][j]

            img = img[0, ...]
            sv_img = sv_img[..., 0]

            fps_.append(1/(time() - t0))


            # image_plot = visuals.SegmVisuals(classes = ['background', 'smoke'])
            # image_plot.build_plt(img = img_, mask = sv_img, fig_title = 'Smoke Segmentation')
            # image_plot.store_fig(fp = os.path.join(sv_path, 'comb_' + img_name.split('.')[0] + '.png'))

            # sv_img = Image.fromarray(sv_img)
            # sv_img.save(os.path.join(sv_path, img_name.split('.')[0] + '.png'))
            
    print('FPS: %.1f'%(sum(fps_)/len(fps_)))