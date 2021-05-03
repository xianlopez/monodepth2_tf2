import os
import random
import cv2
import tensorflow as tf
import numpy as np
from multiprocessing import Process, Pipe

height = 192
width = 640


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


class ReaderOpts:
    def __init__(self, kitti_path, batch_size):
        self.kitti_path = kitti_path
        self.batch_size = batch_size


class DataReader(tf.keras.utils.Sequence):
    def __init__(self, opts):
        self.path_trios = get_images_paths(opts.kitti_path)
        # self.path_trios = self.path_trios[:100]
        random.shuffle(self.path_trios)
        self.batch_size = opts.batch_size
        self.nbatches = len(self.path_trios) // self.batch_size
        self.batch_index = 0

    def fetch_batch(self):
        x = np.zeros((self.batch_size, height, width, 9), np.float32)
        for idx_in_batch in range(self.batch_size):
            trio_idx = self.batch_index * self.batch_size + idx_in_batch
            this_trio = self.path_trios[trio_idx]
            # Read images:
            img1 = cv2.imread(this_trio[0])
            img2 = cv2.imread(this_trio[1])
            img3 = cv2.imread(this_trio[2])
            # Resize:
            img1 = cv2.resize(img1, (width, height))
            img2 = cv2.resize(img2, (width, height))
            img3 = cv2.resize(img3, (width, height))
            # Make pixel values between 0 and 1:
            img1 = img1.astype(np.float32) / 255.0
            img2 = img2.astype(np.float32) / 255.0
            img3 = img3.astype(np.float32) / 255.0
            # Concatenate:
            x[idx_in_batch, ...] = np.concatenate([img1, img2, img3], axis=2)  # (height, width, 9)

        self.batch_index += 1
        if self.batch_index == self.nbatches:
            self.batch_index = 0
            random.shuffle(self.path_trios)
            # print('Rewinding data!')

        return x  # (batch_size, height, width, 9)

def async_reader_loop(opts, conn):
    print('async_reader_loop is alive!')
    reader = DataReader(opts)
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
