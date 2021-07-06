import argparse
import time
from path import Path

import numpy as np
import torch

import torch.utils.data
from tqdm.std import tqdm
import models

import custom_transforms
from data.validation_folders import ValidationSet
from loss_functions import compute_errors
from logger import AverageMeter

parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--split', default='val', type=str, help='name of split files')
parser.add_argument('--dataset', type=str, choices=['kitti', 'nyu', 'waymo'], default='kitti', help='the dataset to train')
parser.add_argument('--resnet-layers',  type=int, default=18, choices=[18, 50], help='number of ResNet layers for depth estimation.')
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained dispnet model')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    global device
    args = parser.parse_args()

    # Data loading code
    if args.dataset=='waymo':
        training_size = [384, 640]
    elif args.dataset=='kitti':
        training_size = [192, 640]

    normalize = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                            std=[0.225, 0.225, 0.225])

    test_transform = custom_transforms.Compose([custom_transforms.RescaleTo(training_size), 
                                                 custom_transforms.ArrayToTensor(), 
                                                 normalize])
    
    print("=> fetching scenes in '{}'".format(args.data))
    test_set = ValidationSet(
        args.data,
        split=args.split,
        transform=test_transform,
        dataset=args.dataset,
        load_interval=10
    )
    
    print('{} samples found in {} test scenes'.format(len(test_set), len(test_set.scenes)))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=False)

    # create model
    disp_net = models.DispResNet(args.resnet_layers, False).to(device)
    weights = torch.load(args.pretrained_disp)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    errors, error_names = test(args, test_loader, disp_net)
    
    # print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "scale"))
    print("\n  " + ("{:>8} | " * 7).format(*error_names))
    print(("&{: 8.3f}  " * 7).format(*errors) + "\\\\")    
    # print(errors)
    
    


@torch.no_grad()
def test(args, test_loader, disp_net):
    global device
    error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3', 'scale']
    errors = AverageMeter(i=len(error_names))

    for i, (tgt_img, depth) in tqdm(enumerate(test_loader)):
        tgt_img = tgt_img.to(device)
        depth = depth.to(device)

        # check gt
        if depth.nelement() == 0:
            continue

        # compute output
        output_disp = disp_net(tgt_img)
        output_depth = 1/output_disp[:, 0]

        if depth.nelement() != output_depth.nelement():
            b, h, w = depth.size()
            output_depth = torch.nn.functional.interpolate(output_depth.unsqueeze(1), [h, w]).squeeze(1)

        errors.update(compute_errors(depth, output_depth, args.dataset))

    return errors.avg, error_names


if __name__ == '__main__':
    main()
