import os
import random
import cv2
import numpy as np
from multiprocessing import Pool, RawArray, Process, Pipe


var_dict = {}


def get_images_paths(kitti_path):
    path_trios = []
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
            assert os.path.isdir(images_dir)
            frames = os.listdir(images_dir)
            frames.sort()
            n_trios += len(frames) - 1
            for i in range(len(frames) - 2):
                path_trios.append([os.path.join(images_dir, frames[i]),
                                   os.path.join(images_dir, frames[i + 1]),
                                   os.path.join(images_dir, frames[i + 2])])
        print('    Day ' + day + ': ' + str(n_trios) + ' trios in ' + str(n_sequences) + ' sequences.')
    print('Total number of KITTI trios: ' + str(len(path_trios)))
    return path_trios


def init_worker(batch_imgs_Arr, batch_imgs_shape):
    var_dict['batch_imgs_Arr'] = batch_imgs_Arr
    var_dict['batch_imgs_shape'] = batch_imgs_shape


def read_images(inputs):
    opts = inputs[0]
    imgs_paths = inputs[1]
    position_in_batch = inputs[2]

    # Read images:
    img1 = cv2.imread(imgs_paths[0])
    img2 = cv2.imread(imgs_paths[1])
    img3 = cv2.imread(imgs_paths[2])
    # Resize:
    img1 = cv2.resize(img1, (opts.img_width, opts.img_height))
    img2 = cv2.resize(img2, (opts.img_width, opts.img_height))
    img3 = cv2.resize(img3, (opts.img_width, opts.img_height))
    # Make pixel values between 0 and 1:
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0
    img3 = img3.astype(np.float32) / 255.0
    # Concatenate:
    all_images = np.concatenate([img1, img2, img3], axis=2)  # (height, width, 9)
    # Wrap shared data as numpy arrays:
    batch_imgs_np = np.frombuffer(var_dict['batch_imgs_Arr'], dtype=np.float32).reshape(var_dict['batch_imgs_shape'])
    # Assign values:
    batch_imgs_np[position_in_batch, :, :, :] = all_images


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
        self.path_trios = get_images_paths(opts.kitti_path)
        # self.path_trios = self.path_trios[:100]
        random.shuffle(self.path_trios)
        self.batch_size = opts.batch_size
        self.nbatches = len(self.path_trios) // self.batch_size
        self.batch_index = 0

        # Initialize batch buffers:
        self.batch_imgs_shape = (opts.batch_size, opts.img_height, opts.img_width, 9)
        self.batch_imgs_Arr = RawArray('f', self.batch_imgs_shape[0] * self.batch_imgs_shape[1] *
                                  self.batch_imgs_shape[2] * self.batch_imgs_shape[3])
        # Initialize pool:
        self.pool = Pool(processes=opts.nworkers, initializer=init_worker, initargs=
            (self.batch_imgs_Arr, self.batch_imgs_shape))

    def fetch_batch(self):
        input_data = []
        for position_in_batch in range(self.opts.batch_size):
            data_idx = self.batch_index * self.opts.batch_size + position_in_batch
            input_data.append((self.opts, self.path_trios[data_idx], position_in_batch))

        self.pool.map(read_images, input_data)

        batch_imgs_np = np.frombuffer(self.batch_imgs_Arr, dtype=np.float32).reshape(self.batch_imgs_shape)

        self.batch_index += 1
        if self.batch_index == self.nbatches:
            self.batch_index = 0
            random.shuffle(self.path_trios)

        return batch_imgs_np  # (batch_size, height, width, 9)


def async_reader_loop(opts, conn):
    print('async_reader_loop is alive!')
    reader = ParallelReader(opts)
    conn.send(reader.nbatches)
    batch_imgs = reader.fetch_batch()
    while conn.recv() == 'GET':
        conn.send(batch_imgs)
        batch_imgs = reader.fetch_batch()
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
        batch_imgs = self.conn1.recv()
        return batch_imgs

    def __exit__(self, type, value, traceback):
        print('Ending AsyncParallelReader')
        self.conn1.send('END')
        self.reader_process.join()

    def __enter__(self):
        return self
