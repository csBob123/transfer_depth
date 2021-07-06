import argparse
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from LeReS import RelDepthModel, load_ckpt
from path import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Configs for LeReS')
    parser.add_argument('--backbone', default='resnext101', help='Checkpoint path to load')
    parser.add_argument('--load_ckpt', default='weights/res101.pth', help='Checkpoint path to load')

    parser.add_argument('--data-dir', default='./depth_data/bonn/', help='Checkpoint path to load')

    args = parser.parse_args()
    return args


def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
		                                transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img


def inference_one(scene_dir, depth_model):
    
    folder = Path(scene_dir)
    out_dir = folder/'pseudo_depth'

    if os.path.exists(out_dir):
        files = glob.glob(out_dir/'*')
        for f in files:
            os.remove(f)
    else:
        out_dir.makedirs_p()
    
    image_folder = folder/'images'
    rgb_files = sorted(image_folder.files('*.png'))
    
    for idx, f in enumerate(rgb_files):

        rgb = cv2.imread(f)
        rgb_c = rgb[:, :, ::-1].copy()
  
        A_resize = cv2.resize(rgb_c, (448, 448))
        img_torch = scale_torch(A_resize)[None, :, :, :]

        pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()

        name = out_dir/'{:06d}.png'.format(idx)

        cv2.imwrite(name, (pred_depth/pred_depth.max() * 60000).astype(np.uint16))


if __name__ == '__main__':

    args = parse_args()

    # create depth model
    depth_model = RelDepthModel(backbone=args.backbone)
    depth_model.eval()

    # load checkpoint
    load_ckpt(args, depth_model, None, None)
    depth_model.cuda()

    for name in sorted(os.listdir(args.data_dir)):
        if os.path.isdir(args.data_dir+name):
            print(name)
            inference_one(args.data_dir+name, depth_model)
