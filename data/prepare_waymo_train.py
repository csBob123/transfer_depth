import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from path import Path
from scipy import sparse
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import (frame_utils, range_image_utils,
                                      transform_utils)

tf.enable_eager_execution()

def save_sparse_depth(depth, filename):
    sparse_depth = sparse.csr_matrix(depth)
    sparse.save_npz(filename, sparse_depth)

def load_sparse_depth(filename):
    sparse_depth = sparse.load_npz(filename)
    depth = sparse_depth.todense()
    return depth

def convert_pose(pose):
    # input is cam2vech pose 4x4
    # output is cam2velo pose 4x4
    T_front_cam_to_ref = np.array([[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0],[1.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 1.0]])
    
    T_cam_to_vehicle = pose
    T_vehicle_to_cam = np.linalg.inv(T_cam_to_vehicle)
    Tr_velo_to_cam = T_front_cam_to_ref @ T_vehicle_to_cam
    
    return np.linalg.inv(Tr_velo_to_cam)

def decode_waymo_data(frame):
    # input: a tfrecord data
    # output: color, depth, intrinsic, pose

    images = sorted(frame.images, key=lambda i: i.name)
    frame.lasers.sort(key=lambda laser: laser.name)

    (range_images, camera_projections,
     range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose)

    points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        ri_index=1)

    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    points_all_ri2 = np.concatenate(points_ri2, axis=0)

    # camera projection corresponding to each point.
    cp_points_all = np.concatenate(cp_points, axis=0)
    cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)

    cp_points_all_concat = np.concatenate([cp_points_all, points_all], axis=-1)
    cp_points_all_concat_tensor = tf.constant(cp_points_all_concat)

    # The distance between lidar points and vehicle frame origin.
    points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
    cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

    mask = tf.equal(cp_points_all_tensor[..., 0], images[0].name)

    cp_points_all_tensor = tf.cast(tf.gather_nd(
        cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
    points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))

    projected_points_all_from_raw_data = tf.concat(
        [cp_points_all_tensor[..., 1:3], points_all_tensor], axis=-1).numpy()

    color = tf.image.decode_jpeg(images[0].image).numpy()
    
    h,w,c = color.shape

    depth = np.zeros([h,w], dtype=np.float32)
    for point in projected_points_all_from_raw_data:
        x, y = int(point[0]), int(point[1])
        if (x < w) & (y < h):
            depth[y,x] = point[2]

    # pose
    pose = convert_pose(np.array(frame.pose.transform).reshape(4,4))

    # intrinsic
    camera = frame.context.camera_calibrations[0]
    assert(camera.name==1)
    intrinsic = np.eye(3,3)
    intrinsic[0, 0] = camera.intrinsic[0]
    intrinsic[1, 1] = camera.intrinsic[1]
    intrinsic[0, 2] = camera.intrinsic[2]
    intrinsic[1, 2] = camera.intrinsic[3]
    
    return color, depth, intrinsic, pose

def convert_one_scene(out_dir, scene_idx, seg_filename):
    out_dir = Path(out_dir)/'{:04d}'.format(scene_idx)
    out_dir.mkdir_p()

    img_dir = out_dir
    img_dir.mkdir_p()

    depth_dir = out_dir/'depth'
    depth_dir.mkdir_p()

    dataset = tf.data.TFRecordDataset(seg_filename, compression_type='')

    cam = 0
    poses = []
    
    for frame_idx, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        
        color, depth, intrinsic, pose = decode_waymo_data(frame)
        
        rgb_name = img_dir/'{:06d}.jpg'.format(frame_idx)
        cv2.imwrite(rgb_name, color[:,:,::-1])

        depth_name = depth_dir/'{:06d}.npz'.format(frame_idx)
        save_sparse_depth(depth, depth_name)
        
        cam = intrinsic
        
        pose = convert_pose(np.array(frame.pose.transform).reshape(4,4))
        poses.append(pose[:3].reshape(1,12))

    poses = np.concatenate(poses, axis=0)
    np.savetxt(out_dir/'poses.txt', poses)

    np.savetxt(out_dir/'cam.txt', cam)


input_dir = '/octant_efs/datasets/mm3d_data/data/waymo/waymo_format/training/'

seg_files = Path(input_dir).files('*.tfrecord')
seg_files.sort()


out_dir = '/octant_efs/datasets/depth_data/waymo/training/'

print(len(seg_files))

# # save data
# for idx in range(len(seg_files)):
#     print('processing {}th scene'.format(idx))
#     convert_one_scene(out_dir, idx, seg_files[idx])

# generate split 10% for val and 90% for train
train_split=[]
val_split=[]
for idx in range(len(seg_files)):
    if idx < 0.9*len(seg_files):
        train_split.append(idx)
    else:
        val_split.append(idx)
train_split=np.stack(train_split)
val_split=np.stack(val_split)
np.savetxt(Path(out_dir)/'train.txt', train_split, fmt='%04d')
np.savetxt(Path(out_dir)/'val.txt', val_split, fmt='%04d')

