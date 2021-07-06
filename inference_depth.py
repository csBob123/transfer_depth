import argparse

import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import torch
from imageio import imread, imsave
from path import Path
from tqdm import tqdm

from models import DispResNet
from utils import tensor2array

parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained", required=True, type=str, help="pretrained DispResNet path")
parser.add_argument('--resnet-layers', required=True, type=int, default=50, choices=[18, 50], help='depth network architecture.')
parser.add_argument("--scale", required=True, type=float, help="scale factor to metric depth")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")

parser.add_argument("--img_dir", type=str, help="Dataset directory")
parser.add_argument("--dep_dir", type=str, help="Output directory")
parser.add_argument("--vis_dir", type=str, help="vis directory")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def depth_visualizer(data):
    """
    Args:
        data (HxW): depth data
    Returns:
        vis_data (HxWx3): depth visualization (RGB)
    """
    vmax = np.percentile(data, 95)
    normalizer = mpl.colors.Normalize(vmin=data.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='rainbow')
    vis_data = (mapper.to_rgba(data)[:, :, :3] * 255).astype(np.uint8)
    return vis_data

@torch.no_grad()
def main():
    args = parser.parse_args()
 
    disp_net = DispResNet(args.resnet_layers, False).to(device)
    weights = torch.load(args.pretrained)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    img_dir = Path(args.img_dir)
    dep_dir = Path(args.dep_dir)
    vis_dir = Path(args.vis_dir)
    
    dep_dir = Path(args.dep_dir)
    vis_dir.makedirs_p()

    img_files = img_dir.files('*.png')
    img_files.sort()

    print('{} files to test'.format(len(img_files)))

    for file in tqdm(img_files):

        img = imread(file)

        h, w, _ = img.shape
        if (h != args.img_height or w != args.img_width):
            tensor_img = cv2.resize(img, (args.img_width, args.img_height)).astype(np.float32)
        tensor_img = np.transpose(tensor_img, (2, 0, 1))
        tensor_img = torch.from_numpy(tensor_img).unsqueeze(0)
        tensor_img = ((tensor_img/255 - 0.45)/0.225).to(device)

        disp = disp_net(tensor_img)[0,0].cpu().numpy()

        depth = args.scale * 1.0 / disp
        
        depth = cv2.resize(depth, (w, h)).astype(np.float32)
        
        file_path, file_ext = file.relpath(args.img_dir).splitext()
        file_name = file_path.basename()

        np.save(dep_dir/'{}.npy'.format(file_name), depth)
        
        # vis_d = depth_visualizer(depth)
        # vis = np.concatenate([img, vis_d], axis=0)
        # imsave(vis_dir/'{}.png'.format(file_name), vis)


if __name__ == '__main__':
    main()
