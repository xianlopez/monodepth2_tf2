from multiprocessing import Pool, Queue
import numpy as np
import os
import cv2
import random


depth_height = 375
depth_width = 1242


image_means = np.array([123.0, 117.0, 104.0])
image_means /= 255.0
image_means = np.reshape(image_means, [1, 1, 3])


def get_images_paths(kitti_path, depths_path):
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
            drive_depths_path = os.path.join(depths_path, day, drive, 'depth')
            assert os.path.isdir(drive_depths_path)
            assert os.path.isdir(images_dir)
            frames = os.listdir(images_dir)
            frames.sort()
            n_trios += len(frames) - 1
            for i in range(len(frames) - 2):
                target_depth_name = os.path.splitext(frames[i + 1])[0] + '.npy'
                items_paths.append([os.path.join(images_dir, frames[i]),  # previous image
                                    os.path.join(images_dir, frames[i + 1]),  # target image
                                    os.path.join(images_dir, frames[i + 2]),  # next image
                                    os.path.join(drive_depths_path, target_depth_name)])  # target velodyne
        print('    Day ' + day + ': ' + str(n_trios) + ' trios in ' + str(n_sequences) + ' sequences.')
    print('Total number of KITTI trios: ' + str(len(items_paths)))
    return items_paths


def read_batch(batch_info, opts):
    batch_imgs_np = np.zeros((opts.batch_size, opts.img_height, opts.img_width, 9), np.float32)
    batch_depth_np = np.zeros((opts.batch_size, depth_height, depth_width), np.float32)
    for i in range(len(batch_info)):
        item_paths = batch_info[i]
        all_images, depth = read_item(item_paths, opts)
        batch_imgs_np[i, :, :, :] = all_images
        batch_depth_np[i, :, :] = depth
    output_queue.put((batch_imgs_np, batch_depth_np))


def read_item(item_paths, opts):
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

    # Depth:
    depth = read_depth(item_paths[3], opts)

    return all_images, depth


def read_depth(depth_path, opts):
    # In some cases the velodyne file is missing, so we don't have depth. Let's return a matirx
    # of zeros in that case, which will be interpreted as no depth information.
    if os.path.isfile(depth_path):
        depth = np.load(depth_path)
    else:
        depth = np.zeros((depth_height, depth_width), dtype=np.float32)
    return depth


def init_worker(queue):
    global output_queue
    output_queue = queue


class ReaderOpts:
    def __init__(self, kitti_path, depths_path, batch_size, img_height, img_width, nworkers):
        self.kitti_path = kitti_path
        self.depths_path = depths_path
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.nworkers = nworkers


class AsyncReader:
    def __init__(self, opts):
        self.opts = opts
        self.data_info = get_images_paths(opts.kitti_path, opts.depths_path)
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
        imgs, depth = self.output_queue.get()
        self.add_fetch_task()
        return imgs, depth

