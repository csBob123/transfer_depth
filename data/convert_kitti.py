import os
import numpy as np
from path import Path
from scipy import sparse
import glob

def save_sparse_depth(depth, filename):
    sparse_depth = sparse.csr_matrix(depth)
    sparse.save_npz(filename, sparse_depth)

def load_sparse_depth(filename):
    sparse_depth = sparse.load_npz(filename)
    depth = sparse_depth.todense()
    return depth

def convert_one(folder):
    
    folder = Path(folder)
    out_dir = folder/'depth'

    if os.path.exists(out_dir):
        files = glob.glob(out_dir/'*')
        for f in files:
            os.remove(f)
    else:
        out_dir.makedirs_p()
    
    npy_files = sorted(folder.files('*.npy'))

    for idx, f in enumerate(npy_files):
        d = np.load(npy_files[idx])

        name = out_dir/'{:010d}.npz'.format(idx)
        save_sparse_depth(d, name)

data_dir = '/media/bjw/Disk/depth_data/kitti/training/'

for name in sorted(os.listdir(data_dir)):
    if os.path.isdir(data_dir+name):
        print(name)
        convert_one(data_dir+name)


