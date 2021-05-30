import tensorflow as tf
import numpy as np
from datetime import datetime
from sys import stdout
import shutil
import os

from data_reader import AsyncReader, ReaderOpts
from transformations import make_transformation_matrix, concat_images
from loss import LossLayer
from models import build_depth_net, build_pose_net
from drawing import display_training
from metrics import compute_metrics

img_height = 192
img_width = 640

kitti_path = '/home/xian/kitti_data'
val_list = os.path.join(os.path.dirname(__file__), 'splits', 'val_files.txt')
batch_size = 8
nworkers = 6
reader_opts = ReaderOpts(kitti_path, val_list, batch_size, img_height, img_width, nworkers)

K = np.array([[0.58 * img_width, 0, 0.5 * img_width],
              [0, 1.92 * img_height, 0.5 * img_height],
              [0, 0, 1]], dtype=np.float32)

depth_weights_path = '/home/xian/monodepth2_tf2/ckpts/depth_net_19/weights'
pose_weights_path = '/home/xian/monodepth2_tf2/ckpts/pose_net_19/weights'

depth_net = build_depth_net(img_height, img_width, None)
pose_net = build_pose_net(img_height, img_width)

depth_net.load_weights(depth_weights_path)
pose_net.load_weights(pose_weights_path)

loss_layer = LossLayer(K, img_height, img_width, batch_size)


@tf.function
def eval_step(batch_imgs, depth_gt):
    # batch_imgs: (batch_size, height, width, 9) Three images concatenated on the channels dimension
    # depth_gt: (batch_size, height, width)

    img_before = batch_imgs[:, :, :, :3]
    img_target = batch_imgs[:, :, :, 3:6]
    img_after = batch_imgs[:, :, :, 6:]

    disps = depth_net(img_target)  # disparities at different scales, in increasing resolution

    # Loss:
    T_before_target = pose_net(concat_images(img_before, img_target))  # (bs, 6)
    T_target_after = pose_net(concat_images(img_target, img_after))  # (bs, 6)
    matrixT_before_target = make_transformation_matrix(T_before_target, False)  # (bs, 4, 4)
    matrixT_after_target = make_transformation_matrix(T_target_after, True)  # (bs, 4, 4)
    loss_value, image_from_before, image_from_after = \
        loss_layer(disps, matrixT_before_target, matrixT_after_target, img_before, img_target, img_after)

    metrics = compute_metrics(disps[-1], depth_gt)

    return loss_value, metrics, disps, image_from_before, image_from_after


with AsyncReader(reader_opts) as eval_reader:
    eval_start = datetime.now()
    accum_loss = 0.0
    accum_metrics = np.zeros((7), np.float32)
    for batch_idx in range(eval_reader.nbatches):
        batch_imgs, batch_depth_gt = eval_reader.get_batch()
        loss_value, metrics, disps, image_from_before, image_from_after = eval_step(batch_imgs, batch_depth_gt)
        loss_value = loss_value.numpy()
        accum_loss += loss_value
        assert len(metrics) == len(accum_metrics)
        for i in range(len(accum_metrics)):
            accum_metrics[i] += metrics[i]
        stdout.write("\rbatch %d/%d, loss: %.2e    " % (batch_idx + 1, eval_reader.nbatches, loss_value))
        stdout.flush()
        if (batch_idx + 1) % 10 == 0:
            display_training(batch_imgs, disps, image_from_before, image_from_after)
        stdout.write('\n')

    print('Evaluation computed in ' + str(datetime.now() - eval_start))
    print('Average loss: %.4f' % (accum_loss / float(eval_reader.nbatches)))
    print('Average metrics:')
    print('abs_rel...: %.4f' % (accum_metrics[0] / float(eval_reader.nbatches)))
    print('sq_rel....: %.4f' % (accum_metrics[1] / float(eval_reader.nbatches)))
    print('rmse......: %.4f' % (accum_metrics[2] / float(eval_reader.nbatches)))
    print('rmse_log..: %.4f' % (accum_metrics[3] / float(eval_reader.nbatches)))
    print('a1........: %.4f' % (accum_metrics[4] / float(eval_reader.nbatches)))
    print('a2........: %.4f' % (accum_metrics[5] / float(eval_reader.nbatches)))
    print('a3........: %.4f' % (accum_metrics[6] / float(eval_reader.nbatches)))
