import os
import random
import cv2
import tensorflow as tf
import numpy as np

height = 192
width = 640


def get_images_paths(kitti_path):
    path_trios = []
    for scenario in os.listdir(kitti_path):
        print('Reading ' + scenario + '...')
        for day in os.listdir(os.path.join(kitti_path, scenario)):
            n_sequences = 0
            n_trios = 0
            for drive in os.listdir(os.path.join(kitti_path, scenario, day)):
                n_sequences += 1
                # TODO: consider using image_03 as well (right camera)
                images_dir = os.path.join(kitti_path, scenario, day, drive, 'image_02', 'data')
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


class DataReader(tf.keras.utils.Sequence):
    def __init__(self, kitt_path, batch_size):
        self.path_trios = get_images_paths(kitt_path)
        # self.path_trios = self.path_trios[:100]
        random.shuffle(self.path_trios)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.path_trios) // self.batch_size

    def __getitem__(self, batch_idx):
        x = np.zeros((self.batch_size, height, width, 9), np.float32)
        for idx_in_batch in range(self.batch_size):
            trio_idx = batch_idx * self.batch_size + idx_in_batch
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
        return x, x  # (batch_size, height, width, 9)

    def on_epoch_end(self):
        random.shuffle(self.path_trios)
