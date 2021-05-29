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
from drawing import display_training_basic
from metrics import compute_metrics

# TODO: Data augmentation
# TODO: Validation
# TODO: Learning rate schedule

img_height = 192
img_width = 640

kitti_path = '/home/xian/kitti_data'
batch_size = 8
nworkers = 6
reader_opts = ReaderOpts(kitti_path, batch_size, img_height, img_width, nworkers)

nepochs = 20

K = np.array([[0.58 * img_width, 0, 0.5 * img_width],
              [0, 1.92 * img_height, 0.5 * img_height],
              [0, 0, 1]], dtype=np.float32)

pretrained_weights_path = '/home/xian/ckpts/resnet18_fully_trained/ckpt'

depth_net = build_depth_net(img_height, img_width, pretrained_weights_path)
pose_net = build_pose_net(img_height, img_width)

trainable_weights = depth_net.trainable_weights
trainable_weights.extend(pose_net.trainable_weights)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

loss_layer = LossLayer(K, img_height, img_width, batch_size)

train_log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
if os.path.exists(train_log_dir):
    shutil.rmtree(train_log_dir)
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
train_summary_writer.set_as_default()

save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ckpts')

@tf.function
def compute_and_log_metrics(disp_pred, depth_gt, step_count):
    # disp_pred: (batch_size, height, width, 1)
    # depth_gt: (batch_size, depth_height, depth_width)
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_metrics(disp_pred, depth_gt)
    with train_summary_writer.as_default():
        tf.summary.scalar('abs_rel', abs_rel, step=step_count)
        tf.summary.scalar('sq_rel', sq_rel, step=step_count)
        tf.summary.scalar('rmse', rmse, step=step_count)
        tf.summary.scalar('rmse_log', rmse_log, step=step_count)
        tf.summary.scalar('a1', a1, step=step_count)
        tf.summary.scalar('a2', a2, step=step_count)
        tf.summary.scalar('a3', a3, step=step_count)


@tf.function
def train_step(batch_imgs, step_count):
    # batch_imgs: (batch_size, height, width, 9) Three images concatenated on the channels dimension
    img_before = batch_imgs[:, :, :, :3]
    img_target = batch_imgs[:, :, :, 3:6]
    img_after = batch_imgs[:, :, :, 6:]

    with tf.GradientTape() as tape:
        disps = depth_net(img_target)  # disparities at different scales, in increasing resolution
        # TODO: I can spare one of this concatenations, but for now this is clearer:
        T_before_target = pose_net(concat_images(img_before, img_target))  # (bs, 6)
        T_target_after = pose_net(concat_images(img_target, img_after))  # (bs, 6)
        matrixT_before_target = make_transformation_matrix(T_before_target, False)  # (bs, 4, 4)
        matrixT_after_target = make_transformation_matrix(T_target_after, True)  # (bs, 4, 4)
        loss_value, image_from_before, image_from_after =\
            loss_layer(disps, matrixT_before_target, matrixT_after_target, img_before, img_target, img_after)

    grads = tape.gradient(loss_value, trainable_weights)
    optimizer.apply_gradients(zip(grads, trainable_weights))

    with train_summary_writer.as_default():
        tf.summary.scalar('loss', loss_value, step=step_count)

    return loss_value, disps, image_from_before, image_from_after


with AsyncReader(reader_opts) as train_reader:
    step_count = 0
    for epoch in range(nepochs):
        print("\nStart epoch ", epoch + 1)
        epoch_start = datetime.now()
        for batch_idx in range(train_reader.nbatches):
            batch_imgs, batch_depth_gt = train_reader.get_batch()
            step_count_tf = tf.convert_to_tensor(step_count, dtype=tf.int64)
            batch_imgs_tf = tf.convert_to_tensor(batch_imgs, dtype=tf.float32)
            loss_value, disps, image_from_before, image_from_after = train_step(batch_imgs_tf, step_count_tf)
            depth_gt_tf = tf.convert_to_tensor(batch_depth_gt, dtype=tf.float32)
            compute_and_log_metrics(disps[-1], depth_gt_tf, step_count_tf)
            train_summary_writer.flush()
            stdout.write("\rbatch %d/%d, loss: %.2e    " % (batch_idx + 1, train_reader.nbatches, loss_value.numpy()))
            stdout.flush()
            if (batch_idx + 1) % 10 == 0:
                display_training_basic(batch_imgs, disps, batch_depth_gt)
            step_count += 1
        stdout.write('\n')
        print('Epoch computed in ' + str(datetime.now() - epoch_start))

        # Save models:
        print('Saving models')
        pose_net.save_weights(os.path.join(save_dir, 'pose_net_' + str(epoch), 'weights'))
        depth_net.save_weights(os.path.join(save_dir, 'depth_net_' + str(epoch), 'weights'))
