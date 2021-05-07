import os
import random
import cv2
import numpy as np
from multiprocessing import Pool, RawArray, Process, Pipe
from kitti_utils import generate_depth_map
import skimage.transform


var_dict = {}

image_means = np.array([123.0, 117.0, 104.0])
image_means /= 255.0
image_means = np.reshape(image_means, [1, 1, 3])


def get_images_paths(kitti_path):
    items_paths = []
    for day in os.listdir(kitti_path):
        n_sequences = 0
        n_trios = 0
        for drive in os.listdir(os.path.join(kitti_path, day)):
            # Discard the calibration files:
            if drive[-4:] == ".txt":
                continue
            n_sequences += 1
            # TODO: consider using image_03 as well (right camera)
            images_dir = os.path.join(kitti_path, day, drive, 'image_02', 'data')
            velodyne_dir = os.path.join(kitti_path, day, drive, 'velodyne_points', 'data')
            assert os.path.isdir(images_dir)
            frames = os.listdir(images_dir)
            frames.sort()
            n_trios += len(frames) - 1
            for i in range(len(frames) - 2):
                target_velo_name = os.path.splitext(frames[i + 1])[0] + '.bin'
                items_paths.append([os.path.join(images_dir, frames[i]),  # previous image
                                    os.path.join(images_dir, frames[i + 1]),  # target image
                                    os.path.join(images_dir, frames[i + 2]),  # next image
                                    os.path.join(kitti_path, day),  # calib dir
                                    os.path.join(velodyne_dir, target_velo_name)])  # target velodyne
        print('    Day ' + day + ': ' + str(n_trios) + ' trios in ' + str(n_sequences) + ' sequences.')
    print('Total number of KITTI trios: ' + str(len(items_paths)))
    return items_paths


def init_worker(batch_imgs_Arr, batch_imgs_shape, batch_depth_Arr, batch_depth_shape):
    var_dict['batch_imgs_Arr'] = batch_imgs_Arr
    var_dict['batch_imgs_shape'] = batch_imgs_shape
    var_dict['batch_depth_Arr'] = batch_depth_Arr
    var_dict['batch_depth_shape'] = batch_depth_shape


def get_depth(velo_path, calib_dir, height, width):
    assert os.path.isfile(velo_path)
    assert os.path.isdir(calib_dir)
    assert os.path.isfile(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    assert os.path.isfile(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    depth_full_size = generate_depth_map(calib_dir, velo_path)
    depth_resized = skimage.transform.resize(
        depth_full_size, (height, width), order=0, preserve_range=True, mode='constant')
    return depth_resized  # (height, width)


def read_item(inputs):
    opts = inputs[0]
    item_paths = inputs[1]
    position_in_batch = inputs[2]

    # Read images:
    img1 = cv2.imread(item_paths[0])
    img2 = cv2.imread(item_paths[1])
    img3 = cv2.imread(item_paths[2])
    # Resize:
    img1 = cv2.resize(img1, (opts.img_width, opts.img_height))
    img2 = cv2.resize(img2, (opts.img_width, opts.img_height))
    img3 = cv2.resize(img3, (opts.img_width, opts.img_height))
    # Make pixel values between 0 and 1:
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    img3 = img3.astype(np.float32) / 255.0
    # Subtract mean:
    img1 = img1 - image_means
    img2 = img2 - image_means
    img3 = img3 - image_means
    # Concatenate:
    all_images = np.concatenate([img1, img2, img3], axis=2)  # (height, width, 9)
    # Wrap shared data as numpy arrays:
    batch_imgs_np = np.frombuffer(var_dict['batch_imgs_Arr'], dtype=np.float32).reshape(var_dict['batch_imgs_shape'])
    # Assign values:
    batch_imgs_np[position_in_batch, :, :, :] = all_images

    # Depth ground truth (corresponding to the middle image):
    calib_dir = item_paths[3]
    velodyne_path = item_paths[4]
    depth = get_depth(velodyne_path, calib_dir, opts.img_height, opts.img_width)  # (height, width)
    batch_depth_np = np.frombuffer(var_dict['batch_depth_Arr'], dtype=np.float32).reshape(var_dict['batch_depth_shape'])
    batch_depth_np[position_in_batch, :, :] = depth


class ReaderOpts:
    def __init__(self, kitti_path, batch_size, img_height, img_width, nworkers):
        self.kitti_path = kitti_path
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.nworkers = nworkers


class ParallelReader:
    def __init__(self, opts):
        self.opts = opts
        self.items_paths = get_images_paths(opts.kitti_path)
        # self.items_paths = self.items_paths[:100]
        random.shuffle(self.items_paths)
        self.batch_size = opts.batch_size
        self.nbatches = len(self.items_paths) // self.batch_size
        self.batch_index = 0

        # Initialize batch buffers:
        self.batch_imgs_shape = (opts.batch_size, opts.img_height, opts.img_width, 9)
        self.batch_imgs_Arr = RawArray('f', self.batch_imgs_shape[0] * self.batch_imgs_shape[1] *
                                  self.batch_imgs_shape[2] * self.batch_imgs_shape[3])
        self.batch_depth_shape = (opts.batch_size, opts.img_height, opts.img_width)
        self.batch_depth_Arr = RawArray('f', self.batch_depth_shape[0] * self.batch_depth_shape[1] *
                                  self.batch_depth_shape[2])
        # Initialize pool:
        self.pool = Pool(processes=opts.nworkers, initializer=init_worker, initargs=
            (self.batch_imgs_Arr, self.batch_imgs_shape, self.batch_depth_Arr, self.batch_depth_shape))

    def fetch_batch(self):
        input_data = []
        for position_in_batch in range(self.opts.batch_size):
            data_idx = self.batch_index * self.opts.batch_size + position_in_batch
            input_data.append((self.opts, self.items_paths[data_idx], position_in_batch))

        self.pool.map(read_item, input_data)

        batch_imgs_np = np.frombuffer(self.batch_imgs_Arr, dtype=np.float32).reshape(self.batch_imgs_shape)
        batch_depth_np = np.frombuffer(self.batch_depth_Arr, dtype=np.float32).reshape(self.batch_depth_shape)

        self.batch_index += 1
        if self.batch_index == self.nbatches:
            self.batch_index = 0
            random.shuffle(self.items_paths)

        # batch_imgs_np  # (batch_size, height, width, 9)
        # batch_depth_np  # (batch_size, height, width)
        return batch_imgs_np, batch_depth_np


def async_reader_loop(opts, conn):
    print('async_reader_loop is alive!')
    reader = ParallelReader(opts)
    conn.send(reader.nbatches)
    batch_imgs, batch_depth = reader.fetch_batch()
    while conn.recv() == 'GET':
        conn.send((batch_imgs, batch_depth))
        batch_imgs, batch_depth = reader.fetch_batch()
    print('async_reader_loop says goodbye!')


class AsyncParallelReader:
    def __init__(self, opts):
        print('Starting AsyncParallelReader')
        self.conn1, conn2 = Pipe()
        self.reader_process = Process(target=async_reader_loop, args=(opts, conn2))
        self.reader_process.start()
        self.nbatches = self.conn1.recv()

    def get_batch(self):
        self.conn1.send('GET')
        batch_imgs, batch_depth = self.conn1.recv()
        return batch_imgs, batch_depth

    def __exit__(self, type, value, traceback):
        print('Ending AsyncParallelReader')
        self.conn1.send('END')
        self.reader_process.join()

    def __enter__(self):
        return self
