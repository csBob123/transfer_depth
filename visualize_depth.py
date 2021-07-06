from imageio import imread, imsave
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
import cv2
import matplotlib as mpl
import matplotlib.cm as cm

parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--img_dir")
parser.add_argument("--dep_dir")
parser.add_argument("--vis_dir")


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


def main():

    args = parser.parse_args()
    
    img_dir = Path(args.img_dir)
    dep_dir = Path(args.dep_dir)
    
    vis_dir = Path(args.vis_dir)
    vis_dir.makedirs_p()

    img_files = img_dir.files('*.png')
    npy_files = dep_dir.files('*.npy')
    
    img_files.sort()
    npy_files.sort()

    print('{} files to test'.format(len(img_files)))

    idx = 0
    for img_f, npy_f in tqdm(zip(img_files, npy_files)):

        img = imread(img_f)
        
        dep = np.load(npy_f)

        vis_d = depth_visualizer(dep)
        
        vis = np.concatenate([img, vis_d], axis=0)
        
        imsave(vis_dir/'{:06d}.png'.format(idx), vis)
        
        idx += 1
      

if __name__ == '__main__':
    main()
