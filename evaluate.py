import argparse
import tensorflow as tf
import numpy as np
import os
import cv2

from model import training_model
from data_reader import height, width
from loss import compute_loss


K = np.array([[0.58 * width, 0, 0.5 * width],
              [0, 1.92 * height, 0.5 * height],
              [0, 0, 1]], dtype=np.float32)


# TODO: Maybe this values should be the same used when clipping the model output.
min_depth = 1e-3
max_depth = 80


def compute_metrics(net_output, batch_gt):
    # batch_gt: (batch_size, gt_height, gt_width, 2) [depth, mask]
    # net_output: (batch_size, height, width, 3 * 2 * 4 + 1)
    depth_pred = net_output[:, :, :, -1]   # (batch_size, height, width)
    # TODO: Change when I change the model
    # # net_output: (batch_size, height, width, 1)
    # depth_pred = tf.squeeze(net_output, axis=-1)   # (batch_size, height, width)
    _, gt_height, gt_width, _ = batch_gt.shape

    depth_gt = batch_gt[:, :, :, 0]
    mask = batch_gt[:, :, :, 1]
    n_valid_gt = tf.reduce_sum(mask)  # Note this is float32

    # TODO: Consider flattening and doing gather_nd to have only the GT and predictions in the mask.

    # We evaluate on the scale the ground truth is.
    # Expand dimension to allow usage of tf.image.resize, then squeeze.
    depth_pred = tf.image.resize(tf.expand_dims(depth_pred, axis=-1), (gt_height, gt_width))
    depth_pred = tf.squeeze(depth_pred, axis=-1)  # (batch_size, gt_height, gt_width)

    depth_diff_masked = (depth_pred - depth_gt) * mask
    are = tf.reduce_sum(tf.abs(depth_diff_masked / tf.maximum(depth_gt, min_depth))) / n_valid_gt  # Absolute relative error
    rmse = tf.reduce_sum(tf.sqrt(tf.square(depth_diff_masked)) / n_valid_gt)
    # RMSE log as in monodepth2 code:
    rmse_log = tf.reduce_sum(tf.sqrt(tf.square(tf.math.log(depth_gt) - tf.math.log(depth_pred))) / n_valid_gt)
    delta = tf.maximum(depth_pred / tf.maximum(depth_gt, min_depth), depth_gt / tf.maximum(depth_pred, min_depth))
    a1 = tf.reduce_sum(tf.cast(delta < 1.25, tf.float32)) / n_valid_gt
    a2 = tf.reduce_sum(tf.cast(delta < 1.25 ** 2, tf.float32)) / n_valid_gt
    a3 = tf.reduce_sum(tf.cast(delta < 1.25 ** 3, tf.float32)) / n_valid_gt

    return tf.stack([are, rmse, rmse_log, a1, a2, a3], axis=0)  # (6)


def build_val_step_fun(model, loss_fun):
    @tf.function
    def val_step_fun(batch_imgs, batch_gt):
        # TODO: This shouldn't be done. I should be able to use a model that expects just one image.
        batch_imgs_extended = tf.tile(batch_imgs, [1, 1, 1, 3])
        net_output = model(batch_imgs_extended, training=True)  # TODO: training=False
        loss_value = loss_fun(batch_imgs_extended, net_output)
        loss_value += sum(model.losses)
        metrics = compute_metrics(net_output, batch_gt)
        return loss_value, metrics, net_output
    return val_step_fun


def read_evaluation_data(kitti_path):
    # These things are taken from monodepth2.
    # Evaluation images:
    eval_files_path = os.path.join('evaluation', 'test_files.txt')
    images = []
    with open(eval_files_path, 'r') as fid:
        lines = fid.read().splitlines()
    for line in lines:
        line_split = line.split(' ')
        assert len(line_split) == 3
        assert line_split[2] == 'l'
        seq_path = line_split[0]
        frame_idx = line_split[1]
        img_path = os.path.join(kitti_path, seq_path, 'image_02', 'data', frame_idx + '.jpg')
        img = cv2.imread(img_path)
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32) / 255.0
        images.append(img)
    # Ground truth:
    depth_gt_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'evaluation', 'gt_depths.npz')
    assert os.path.isfile(depth_gt_path)
    depth_gt = np.load(depth_gt_path, allow_pickle=True)["data"]
    # depth_gt is an array with as many elements as images, and each one is a matrix with the ground truth depth.
    assert len(images) == len(depth_gt)
    # Ground truth mask:
    all_gt = []
    for i in range(len(depth_gt)):
        this_depth = depth_gt[i]
        mask = np.logical_and(this_depth > min_depth, this_depth < max_depth)
        gt_height, gt_width = this_depth.shape
        # Again, I took these values from monodepth2 code. I guess it's masking the borders of the images and the
        # lower part (maybe some part of the car is visible, or there is no velodyne data...?)
        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask).astype(np.float32)
        depth_and_mask = np.stack([this_depth, mask], axis=-1)  # (gt_height, gt_width, 2)
        all_gt.append(depth_and_mask)
    return images, all_gt


def evaluate(kitti_path, ckpt_idx):
    model = training_model(K)

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_to_load = 'ckpts/ckpt_' + str(ckpt_idx)
    read_result = checkpoint.read(checkpoint_to_load).expect_partial()
    read_result.assert_existing_objects_matched()

    val_step_fun = build_val_step_fun(model, compute_loss)

    images, all_gt = read_evaluation_data(kitti_path)

    accum_metrics = np.zeros((6), np.float32)
    for i in range(len(images)):
        image = images[i]
        gt = all_gt[i]
        # Add the batch dimension:
        # TODO: We could use a batch size higher than 1 for evaluation.
        batch_imgs = np.expand_dims(image, axis=0)  # (1, height, width, 3)
        batch_gt = np.expand_dims(gt, axis=0)  # (1, gt_height, gt_width, 2)
        loss_value, metrics, net_output = val_step_fun(
            tf.constant(batch_imgs, tf.float32), tf.constant(batch_gt, tf.float32))
        # print('metrics: ' + str(metrics.numpy()))
        accum_metrics += metrics.numpy()

    avg_metrics = accum_metrics / len(images)
    print('Absolute relative error: %.4f' % avg_metrics[0])
    print('RMSE: %.4f' % avg_metrics[1])
    print('RMSE log: %.4f' % avg_metrics[2])
    print('a1: %.4f' % avg_metrics[3])
    print('a2: %.4f' % avg_metrics[4])
    print('a3: %.4f' % avg_metrics[5])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument('--kitti_path', type=str, default='/home/xian/KITTI', help='path to KITTI dataset')
    parser.add_argument('--ckpt_idx', type=int, help='index of the checkpoint to load the initial weights')
    args = parser.parse_args()

    evaluate(args.kitti_path, args.ckpt_idx)
