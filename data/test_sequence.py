import itertools
import math
import multiprocessing
import os

import cv2
from imageio import imread, imwrite
import matplotlib.pyplot as plt
import numpy as np
from path import Path
from scipy import sparse

def save_sparse_depth(depth, filename):
    sparse_depth = sparse.csr_matrix(depth)
    sparse.save_npz(filename, sparse_depth)

def load_sparse_depth(filename):
    sparse_depth = sparse.load_npz(filename)
    depth = sparse_depth.todense()
    return depth

def extend_pose(p):
    # input 3x4
    # out 4x4
    pose = np.eye(4)
    pose[:3]=p
    return pose

def draw_match_2_side(img1, kp1, img2, kp2, N):
    """Draw matches on 2 sides
    Args:
        img1 (HxW(xC) array): image 1
        kp1 (Nx2 array): keypoint for image 1
        img2 (HxW(xC) array): image 2
        kp2 (Nx2 array): keypoint for image 2
        N (int): number of matches to draw
    Returns:
        out_img (Hx2W(xC) array): output image with drawn matches
    """
    kp_list = np.linspace(0, min(kp1.shape[0], kp2.shape[0])-1, N,
                                            dtype=np.int
                                            )

    # Convert keypoints to cv2.Keypoint object
    cv_kp1 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), _size=5) for pt in kp1[kp_list]]
    cv_kp2 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), _size=5) for pt in kp2[kp_list]]

    out_img = np.array([])
    good_matches = [cv2.DMatch(_imgIdx=0, _queryIdx=idx, _trainIdx=idx,_distance=0) for idx in range(N)]
    out_img = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, matches1to2=good_matches, outImg=out_img)

    return out_img 

def convert_pose(pose):
    # input is cam2vech pose 4x4
    # output is cam2velo pose 4x4
    T_front_cam_to_ref = np.array([[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0],[1.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 1.0]])
    
    T_cam_to_vehicle = pose
    T_vehicle_to_cam = np.linalg.inv(T_cam_to_vehicle)
    Tr_velo_to_cam = T_front_cam_to_ref @ T_vehicle_to_cam
    
    return np.linalg.inv(Tr_velo_to_cam)
    

# input_dir = Path('./depth_data/waymo/training/0000')
# img1 = imread(input_dir/'000000.jpg')
# img2 = imread(input_dir/'000005.jpg')
# d1 = load_sparse_depth(input_dir/'depth/000000.npz')

input_dir = Path('/media/bjw/Disk/depth_data/kitti/training/2011_09_26_drive_0001_sync_02')
img1 = imread(input_dir/'0000000000.jpg')
img2 = imread(input_dir/'0000000001.jpg')
d1 = load_sparse_depth(input_dir/'depth/0000000000.npz')

cam = np.loadtxt(input_dir/'cam.txt')

poses = np.loadtxt(input_dir/'poses.txt')
p1 = poses[0].reshape(3,4)
p2 = poses[1].reshape(3,4)

extend_p1 = extend_pose(p1)
extend_p2 = extend_pose(p2)

# extend_p1 = convert_pose(extend_p1)
# extend_p2 = convert_pose(extend_p2)

kp1=[]
kp2=[]

h,w = d1.shape

for y in range(h):
    for x in range(w):
        z = d1[y,x]
        if z > 0:
                        
            pt = np.array([x,y,1])
            
            cam_pt = np.linalg.inv(cam) @ np.transpose(pt) # 3x1
            cam_pt = cam_pt * z
            
            word_pt = extend_p1 @ np.concatenate([cam_pt, np.array([1])], axis=0) # 4x1
            
            cam2_pt = np.linalg.inv(extend_p2) @ word_pt # 4x1
            
            cam2_pt = cam2_pt[:3] / cam2_pt[2]
            
            pt2 = cam @ cam2_pt

            x2, y2 = pt2[:2]
            if x2 > 0 and x2 < w and y2 > 0 and y2 < h:
                kp1.append(np.array([x,y]))     
                kp2.append(np.array([x2,y2]))

kp1 = np.stack(kp1,axis=0)
kp2 = np.stack(kp2,axis=0)            

print(kp1.shape,kp2.shape)

show = draw_match_2_side(img1, kp1, img2, kp2, N=200)

imwrite('./data/test_image.jpg', show)

