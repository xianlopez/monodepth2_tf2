import os
import skimage.transform
import numpy as np
import shutil

from kitti_utils import generate_depth_map


def get_depth(velo_path, calib_dir, height, width):
    assert os.path.isdir(calib_dir)
    assert os.path.isfile(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    assert os.path.isfile(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    depth_full_size = generate_depth_map(calib_dir, velo_path)
    depth_resized = skimage.transform.resize(
        depth_full_size, (height, width), order=0, preserve_range=True, mode='constant')

    depth_resized = depth_resized.astype(np.float32)

    return depth_resized  # (height, width)


def preprocess_depth(kitti_path, output_dir, height, width):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for day in os.listdir(kitti_path):
        calib_dir = os.path.join(kitti_path, day)
        os.makedirs(os.path.join(output_dir, day))
        for drive in os.listdir(os.path.join(kitti_path, day)):
            # Discard the calibration files:
            if drive[-4:] == ".txt":
                continue
            velodyne_dir = os.path.join(kitti_path, day, drive, 'velodyne_points', 'data')
            out_depth_dir = os.path.join(output_dir, day, drive, 'depth')
            os.makedirs(out_depth_dir)
            assert os.path.isdir(velodyne_dir)
            for velo_name in os.listdir(velodyne_dir):
                velo_path = os.path.join(velodyne_dir, velo_name)
                depth = get_depth(velo_path, calib_dir, height, width)
                rawname = os.path.splitext(velo_name)[0]
                output_path = os.path.join(out_depth_dir, rawname + '.npy')
                np.save(output_path, depth)


if __name__ == '__main__':
    kitti_path = '/home/xian/kitti_data'
    output_dir = '/home/xian/monodepth2_tf2/preprocessed_depths'
    height = 192
    width = 640
    preprocess_depth(kitti_path, output_dir, height, width)
