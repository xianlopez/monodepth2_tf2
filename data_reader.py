from multiprocessing import Pool, Queue
import numpy as np
import os
import cv2
import random
import skimage.transform

from kitti_utils import generate_depth_map


depth_height = 375
depth_width = 1242


image_means = np.array([123.0, 117.0, 104.0])
image_means /= 255.0
image_means = np.reshape(image_means, [1, 1, 3])

T_left_right = np.eye(4, dtype=np.float32)
T_left_right[0, 3] = 0.1
T_right_left = np.eye(4, dtype=np.float32)
T_right_left[0, 3] = -0.1


def get_image_path(kitti_path, sequence, frame_idx, side):
    side_folder = 'image_02' if side == 'l' else 'image_03'
    img_path = os.path.join(kitti_path, sequence, side_folder, 'data', '%010d.jpg' % frame_idx)
    assert os.path.isfile(img_path)
    return img_path


def get_calib_dir(kitti_path, sequence):
    return os.path.join(kitti_path, os.path.dirname(sequence))


def get_velo_path(kitti_path, sequence, frame_idx):
    velodyne_dir = os.path.join(kitti_path, sequence, 'velodyne_points', 'data')
    velo_path = os.path.join(velodyne_dir, '%010d.bin' % frame_idx)
    assert os.path.isfile(velo_path)
    return velo_path


def get_items_info(kitti_path, files_list):
    items_paths = []
    with open(files_list, 'r') as fid:
        lines = fid.readlines()
    for line in lines:
        line_split = line.split(' ')
        assert len(line_split) == 3
        sequence = line_split[0]
        frame_idx = int(line_split[1])
        assert frame_idx > 0
        side = line_split[2].rstrip()  # rstrip needed to remove new line character
        assert side == 'l' or side == 'r'
        other_side = 'l' if side == 'r' else 'r'
        items_paths.append([get_image_path(kitti_path, sequence, frame_idx - 1, side),  # previous image
                            get_image_path(kitti_path, sequence, frame_idx, side),  # target image
                            get_image_path(kitti_path, sequence, frame_idx + 1, side),  # next image
                            get_image_path(kitti_path, sequence, frame_idx, other_side),  # opposite image
                            get_velo_path(kitti_path, sequence, frame_idx),  # target velodyne
                            get_calib_dir(kitti_path, sequence),  # calib dir
                            side])  # side, needed to get the depth
    print('Total number of items: ' + str(len(items_paths)))
    return items_paths


def read_batch(batch_info, opts):
    batch_imgs_np = np.zeros((opts.batch_size, opts.img_height, opts.img_width, 12), np.float32)
    batch_T_opposite_target = np.zeros((opts.batch_size, 4, 4), np.float32)
    batch_depth_np = np.zeros((opts.batch_size, depth_height, depth_width), np.float32)
    for i in range(len(batch_info)):
        item_info = batch_info[i]
        all_images, T_opposite_target, depth = read_item(item_info, opts)
        batch_imgs_np[i, :, :, :] = all_images
        batch_T_opposite_target[i, :, :] = T_opposite_target
        batch_depth_np[i, :, :] = depth
    output_queue.put((batch_imgs_np, batch_T_opposite_target, batch_depth_np))


def read_item(item_info, opts):
    # Read images:
    img1 = cv2.imread(item_info[0])
    img2 = cv2.imread(item_info[1])
    img3 = cv2.imread(item_info[2])
    img4 = cv2.imread(item_info[3])
    # Resize:
    img1 = cv2.resize(img1, (opts.img_width, opts.img_height))
    img2 = cv2.resize(img2, (opts.img_width, opts.img_height))
    img3 = cv2.resize(img3, (opts.img_width, opts.img_height))
    img4 = cv2.resize(img4, (opts.img_width, opts.img_height))
    # Make pixel values between 0 and 1:
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    img3 = img3.astype(np.float32) / 255.0
    img4 = img4.astype(np.float32) / 255.0
    # Subtract mean:
    img1 = img1 - image_means
    img2 = img2 - image_means
    img3 = img3 - image_means
    img4 = img4 - image_means
    # Concatenate:
    all_images = np.concatenate([img1, img2, img3, img4], axis=2)  # (height, width, 12)

    # Depth:
    side = item_info[6]
    depth = get_depth(item_info[4], item_info[5], side)

    # Transformation between stereo pair (depends if the target image is left or right)
    if side == 'r':
        T_opposite_target = T_left_right
    else:
        T_opposite_target = T_right_left

    return all_images, T_opposite_target, depth


def get_depth(velo_path, calib_dir, side):
    # In some cases the velodyne file is missing. We'll just return a matrix of zeros
    # which will be interpreted as no depth information for every pixel.
    if not os.path.isfile(velo_path):
        return np.zeros((depth_height, depth_width), dtype=np.float32)
    assert os.path.isdir(calib_dir)
    assert os.path.isfile(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    assert os.path.isfile(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    assert side in ('l', 'r')
    cam = 2 if side == 'l' else 3
    depth = generate_depth_map(calib_dir, velo_path, cam)
    depth_resized = skimage.transform.resize(
        depth, (depth_height, depth_width), order=0, preserve_range=True, mode='constant')
    depth_resized = depth_resized.astype(np.float32)
    return depth_resized


def init_worker(queue):
    global output_queue
    output_queue = queue


class ReaderOpts:
    def __init__(self, kitti_path, files_list, batch_size, img_height, img_width, nworkers):
        self.kitti_path = kitti_path
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.nworkers = nworkers
        self.files_list = files_list


class AsyncReader:
    def __init__(self, opts):
        self.opts = opts
        self.data_info = get_items_info(opts.kitti_path, opts.files_list)
        # self.data_info = self.data_info[:200]
        self.nbatches = len(self.data_info) // opts.batch_size

        self.output_queue = Queue()
        self.pool = Pool(processes=self.opts.nworkers, initializer=init_worker, initargs=(self.output_queue,))
        self.next_batch_idx = 0
        random.shuffle(self.data_info)
        for i in range(min(self.opts.nworkers, self.nbatches)):
            self.add_fetch_task()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        print('Closing AsyncReader')
        self.pool.close()
        # OpenCV seems to play bad with multiprocessing, so I need to add this here. Maybe I could change
        # the reading of the images to use skimage instead of cv2.
        self.pool.terminate()
        self.pool.join()
        print('Closed')

    def add_fetch_task(self):
        batch_info = []
        for i in range(self.opts.batch_size):
            batch_info.append(self.data_info[self.next_batch_idx * self.opts.batch_size + i])
        self.pool.apply_async(read_batch, args=(batch_info, self.opts))
        if self.next_batch_idx == self.nbatches - 1:
            self.next_batch_idx = 0
            random.shuffle(self.data_info)
        else:
            self.next_batch_idx += 1

    def get_batch(self):
        imgs, batch_T_opposite_target, depth = self.output_queue.get()
        self.add_fetch_task()
        return imgs, batch_T_opposite_target, depth

