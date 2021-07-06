from numpy.core.fromnumeric import sort
import torch
import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random


def load_as_float(path):
    return imread(path).astype(np.float32)


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        root/scene_1/depth/0000000.npz
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, skip_frames=1, dataset='kitti'):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.dataset = dataset
        self.k = skip_frames

        if self.dataset == 'bonn':
            self.k = 10
        self.crawl_folders(sequence_length)


    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length * self.k, demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)
        for scene in self.scenes:
            imgs = sorted(scene.files('*.jpg') + (scene/'images').files('*.png'))
            pseudo_depths = sorted((scene/'pseudo_depth').files('*.png'))
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            poses = np.loadtxt(scene/'poses.txt').astype(np.float32).reshape(-1,3,4)

            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length * self.k, len(imgs)-demi_length * self.k):
                sample = {'intrinsics': intrinsics, 'tgt_img': imgs[i], 'ref_imgs': [],
                        'tgt_pseudo_depth': pseudo_depths[i], 'ref_pseudo_depths': [],
                        'tgt_pose': poses[i], 'ref_poses':[]}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                    sample['ref_pseudo_depths'].append(pseudo_depths[i+j])
                    sample['ref_poses'].append(poses[i+j])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt_img'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        
        tgt_pseudo_depth = load_as_float(sample['tgt_pseudo_depth'])
        ref_pseudo_depths = [load_as_float(ref) for ref in sample['ref_pseudo_depths']]

        imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
        tgt_img = imgs[0]
        ref_imgs = imgs[1:]
        
        # 
        pseudo_depths, _ = self.transform([tgt_pseudo_depth] + ref_pseudo_depths, np.copy(sample['intrinsics']))
        
        # # clip far depth
        # for d in pseudo_depths:
        #     d[d>np.percentile(d,50)] = 0
        
        tgt_pseudo_depth = pseudo_depths[0]
        ref_pseudo_depths = pseudo_depths[1:]

        poses = []
        poses_inv = []

        tgt_p = sample['tgt_pose']
        tgt_p = np.vstack([tgt_p, np.array([0,0,0,1])]).astype(np.float32)
                
        for ref_p in sample['ref_poses']:
            ref_p = np.vstack([ref_p, np.array([0,0,0,1])]).astype(np.float32)
            pose = np.linalg.inv(ref_p) @ tgt_p
            pose_inv = np.linalg.inv(pose)
            poses.append(torch.from_numpy(pose[:3]).float())
            poses_inv.append(torch.from_numpy(pose_inv[:3]).float())
        
        return tgt_img, ref_imgs, tgt_pseudo_depth, ref_pseudo_depths, intrinsics, poses, poses_inv

    def __len__(self):
        return len(self.samples)
