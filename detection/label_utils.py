import cv2
import numpy as np


def corners_nd(dims, origin=0.5):
    """Generate relative box corners based on length per dim and origin point.

    Args:
        dims (np.ndarray, shape=[N, ndim]): Array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        np.ndarray, shape=[N, 2 ** ndim, ndim]: Returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1.
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim),
        axis=1).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2**ndim, ndim])
    return corners


def rotation_3d_in_axis(points, angles, axis=0):
    """Rotate points in specific axis.

    Args:
        points (np.ndarray, shape=[N, point_size, 3]]):
        angles (np.ndarray, shape=[N]]):
        axis (int): Axis to rotate at.

    Returns:
        np.ndarray: Rotated points.
    """
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack([[rot_cos, zeros, -rot_sin], [zeros, ones, zeros],
                              [rot_sin, zeros, rot_cos]])
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack([[rot_cos, -rot_sin, zeros],
                              [rot_sin, rot_cos, zeros], [zeros, zeros, ones]])
    elif axis == 0:
        rot_mat_T = np.stack([[zeros, rot_cos, -rot_sin],
                              [zeros, rot_sin, rot_cos], [ones, zeros, zeros]])
    else:
        raise ValueError('axis should in range')

    return np.einsum('aij,jka->aik', points, rot_mat_T)


def center_to_corner_box3d(centers,
                           dims,
                           angles=None,
                           origin=(0.5, 1.0, 0.5),
                           axis=1):
    """Convert kitti locations, dimensions and angles to corners.

    Args:
        centers (np.ndarray): Locations in kitti label file with shape (N, 3).
        dims (np.ndarray): Dimensions in kitti label file with shape (N, 3).
        angles (np.ndarray): Rotation_y in kitti label file with shape (N).
        origin (list or array or float): Origin point relate to smallest point.
            use (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0) in lidar.
        axis (int): Rotation axis. 1 for camera and 2 for lidar.

    Returns:
        np.ndarray: Corners with the shape of (N, 8, 3).
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners


def plot_corners3d_on_img(corners_3d,
                          raw_img,
                          lidar2img_rt,
                          color=(0, 255, 0),
                          thickness=3):
    """Project the 3D bbox on 2D image.

    Args:
        bboxes3d (numpy.array, shape=[M, 7]):
            3d bbox (x, y, z, dx, dy, dz, yaw) to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        color (tuple[int]): the color to draw bboxes. Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    img = raw_img.copy()
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3),
         np.ones((num_bbox * 8, 1))], axis=-1)
    pts_2d = pts_4d @ lidar2img_rt.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    for i in range(num_bbox):
        corners = imgfov_pts_2d[i].astype(np.int)
        out_of_img = 0
        for start, end in line_indices:
            try:
                cv2.line(img, (corners[start, 0], corners[start, 1]),
                         (corners[end, 0], corners[end, 1]), color, thickness,
                         cv2.LINE_AA)
            except:
                out_of_img = 1
                pass
        if out_of_img:
            print('out of image')

    return img


def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()

    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array([[float(info) for info in x[4:8]]
                                    for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array([[float(info) for info in x[8:11]]
                                          for x in content
                                          ]).reshape(-1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array([[float(info) for info in x[11:14]]
                                        for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array([float(x[14])
                                          for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations


def plot_corners2d_on_img(corners_2d, raw_img, color=(0, 0, 255), thickness=3):
    img = raw_img.copy()
    num_bbox = corners_2d.shape[0]

    line_indices = ((0, 1), (0, 2), (1, 3), (2, 3))
    for i in range(num_bbox):
        corners = corners_2d[i].astype(np.int)
        out_of_img = 0
        for start, end in line_indices:
            try:
                cv2.line(img, (corners[start, 0], corners[start, 1]),
                         (corners[end, 0], corners[end, 1]), color, thickness,
                         cv2.LINE_AA)
            except:
                out_of_img = 1
                pass
        if out_of_img:
            print('out of image (2d)')

    return img


def draw_bboxes3d_on_img(raw_img, annos, cam_intrinsic, draw_class='Sign'):

    corners3d = center_to_corner_box3d(annos['location'], annos['dimensions'], annos['rotation_y'])

    bboxes2d = annos['bbox']

    # delete other class and out-of-range bbx
    to_delete = []
    for idx in range(annos['name'].shape[0]):
        if annos['name'][idx] == draw_class:
            bbox = bboxes2d[idx]
            if (np.sum(bbox) == 0):
                to_delete.append(idx)
        else:
            to_delete.append(idx)

    corners3d = np.delete(corners3d, obj=to_delete, axis=0)

    K = np.eye(4)
    K[:3, :3] = cam_intrinsic

    img3d = plot_corners3d_on_img(corners3d, raw_img, K)
    return img3d


def draw_bboxes2d_on_img(raw_img, annos, draw_class='Sign'):

    bboxes2d = annos['bbox']

    # delete other class and out-of-range bbx
    to_delete = []
    for idx in range(annos['name'].shape[0]):
        if annos['name'][idx] == draw_class:
            bbox = bboxes2d[idx]
            if (np.sum(bbox) == 0):
                to_delete.append(idx)
        else:
            to_delete.append(idx)

    bboxes2d = np.delete(bboxes2d, obj=to_delete, axis=0)

    corners2d = []
    for idx in range(bboxes2d.shape[0]):
        x0, y0, x1, y1 = bboxes2d[idx]
        corners = np.array([x0, y0, x1, y0, x0, y1, x1, y1]).reshape(4, 2)
        corners2d.append(corners)
    corners2d = np.stack(corners2d)

    img2d = plot_corners2d_on_img(corners2d, raw_img)
    return img2d


if __name__ == "__main__":

    scene_idx = 600
    frame_idx = 160

    img_path = 'depth_data/waymo/training/{:04d}/{:06d}.jpg'.format(scene_idx, frame_idx)
    image = cv2.imread(img_path)

    labe_path = 'depth_data/waymo/training/{:04d}/label/{:06d}.txt'.format(scene_idx, frame_idx)
    annos = get_label_anno(labe_path)

    cam_intrinsic = np.loadtxt('depth_data/waymo/training/{:04d}/cam.txt'.format(scene_idx))

    img3d = draw_bboxes3d_on_img(image, annos, cam_intrinsic, draw_class='Sign')
    cv2.imwrite('./detection/bbox3d_img.png', img3d.astype(np.uint8))

    img2d = draw_bboxes2d_on_img(image, annos, draw_class='Sign')
    cv2.imwrite('./detection/bbox2d_img.png', img2d.astype(np.uint8))
