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

# TODO: Data augmentation

img_height = 192
img_width = 640

kitti_path = '/home/xian/kitti_data'
train_list = os.path.join(os.path.dirname(__file__), 'splits', 'train_files.txt')
val_list = os.path.join(os.path.dirname(__file__), 'splits', 'val_files.txt')
batch_size = 8
nworkers = 6
train_reader_opts = ReaderOpts(kitti_path, train_list, batch_size, img_height, img_width, nworkers)
val_reader_opts = ReaderOpts(kitti_path, val_list, batch_size, img_height, img_width, nworkers)

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

log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
val_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'val'))

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
def train_step(batch_imgs, batch_T_opposite_target, step_count):
    # batch_imgs: (batch_size, height, width, 12) Four images concatenated on the channels dimension
    # batch_T_opposite_target: (batch_size, 4, 4) rotation-vector plus translation
    img_before = batch_imgs[:, :, :, :3]
    img_target = batch_imgs[:, :, :, 3:6]
    img_after = batch_imgs[:, :, :, 6:9]
    img_opposite = batch_imgs[:, :, :, 9:]

    with tf.GradientTape() as tape:
        disps = depth_net(img_target)  # disparities at different scales, in increasing resolution
        # TODO: I can spare one of this concatenations, but for now this is clearer:
        T_before_target = pose_net(concat_images(img_before, img_target))  # (bs, 6)
        T_target_after = pose_net(concat_images(img_target, img_after))  # (bs, 6)
        matrixT_before_target = make_transformation_matrix(T_before_target, False)  # (bs, 4, 4)
        matrixT_after_target = make_transformation_matrix(T_target_after, True)  # (bs, 4, 4)
        loss_value, image_from_before, image_from_after, image_from_opposite =\
            loss_layer(disps, matrixT_before_target, matrixT_after_target, batch_T_opposite_target,
                       img_before, img_target, img_after, img_opposite)

    grads = tape.gradient(loss_value, trainable_weights)
    optimizer.apply_gradients(zip(grads, trainable_weights))

    with train_summary_writer.as_default():
        tf.summary.scalar('loss', loss_value, step=step_count)

    return loss_value, disps, image_from_before, image_from_after, image_from_opposite


@tf.function
def eval_step(batch_imgs, batch_T_opposite_target, depth_gt):
    # batch_imgs: (batch_size, height, width, 12) Four images concatenated on the channels dimension
    # batch_T_opposite_target: (batch_size, 4, 4) rotation-vector plus translation
    # depth_gt: (batch_size, height, width)

    img_before = batch_imgs[:, :, :, :3]
    img_target = batch_imgs[:, :, :, 3:6]
    img_after = batch_imgs[:, :, :, 6:9]
    img_opposite = batch_imgs[:, :, :, 9:]

    disps = depth_net(img_target)  # disparities at different scales, in increasing resolution

    # Loss:
    T_before_target = pose_net(concat_images(img_before, img_target))  # (bs, 6)
    T_target_after = pose_net(concat_images(img_target, img_after))  # (bs, 6)
    matrixT_before_target = make_transformation_matrix(T_before_target, False)  # (bs, 4, 4)
    matrixT_after_target = make_transformation_matrix(T_target_after, True)  # (bs, 4, 4)
    loss_value, image_from_before, image_from_after, image_from_opposite = \
        loss_layer(disps, matrixT_before_target, matrixT_after_target, batch_T_opposite_target,
                   img_before, img_target, img_after, img_opposite)

    metrics = compute_metrics(disps[-1], depth_gt)

    return loss_value, metrics, disps, image_from_before, image_from_after, image_from_opposite


def evaluation_loop(val_reader, step_count):
    print('Evaluating...')
    eval_start = datetime.now()
    accum_loss = 0.0
    accum_metrics = np.zeros((7), np.float32)
    for batch_idx in range(val_reader.nbatches):
        batch_imgs, batch_T_opposite_target, batch_depth_gt = val_reader.get_batch()
        loss_value, metrics, disps, image_from_before, image_from_after, image_from_opposite =\
            eval_step(batch_imgs, batch_T_opposite_target, batch_depth_gt)
        loss_value = loss_value.numpy()
        accum_loss += loss_value
        assert len(metrics) == len(accum_metrics)
        for i in range(len(accum_metrics)):
            accum_metrics[i] += metrics[i]
    metrics = accum_metrics / float(val_reader.nbatches)
    average_loss = accum_loss / float(val_reader.nbatches)
    with val_summary_writer.as_default():
        tf.summary.scalar('loss', average_loss, step=step_count)
        tf.summary.scalar('abs_rel', metrics[0], step=step_count)
        tf.summary.scalar('sq_rel', metrics[1], step=step_count)
        tf.summary.scalar('rmse', metrics[2], step=step_count)
        tf.summary.scalar('rmse_log', metrics[3], step=step_count)
        tf.summary.scalar('a1', metrics[4], step=step_count)
        tf.summary.scalar('a2', metrics[5], step=step_count)
        tf.summary.scalar('a3', metrics[6], step=step_count)
    print('Evaluation computed in ' + str(datetime.now() - eval_start))


with AsyncReader(train_reader_opts) as train_reader, AsyncReader(val_reader_opts) as val_reader:
    step_count = 0
    for epoch in range(nepochs):
        print("\nStart epoch ", epoch + 1)
        epoch_start = datetime.now()
        if epoch == 15:
            optimizer.learning_rate = 1e-5
            print('Changing learning rate to: %.2e' % optimizer.learning_rate)
        for batch_idx in range(train_reader.nbatches):
            batch_imgs, batch_T_opposite_target, batch_depth_gt = train_reader.get_batch()
            step_count_tf = tf.convert_to_tensor(step_count, dtype=tf.int64)
            batch_imgs_tf = tf.convert_to_tensor(batch_imgs, dtype=tf.float32)
            loss_value, disps, image_from_before, image_from_after, image_from_opposite =\
                train_step(batch_imgs_tf, batch_T_opposite_target, step_count_tf)
            depth_gt_tf = tf.convert_to_tensor(batch_depth_gt, dtype=tf.float32)
            compute_and_log_metrics(disps[-1], depth_gt_tf, step_count_tf)
            train_summary_writer.flush()
            stdout.write("\rbatch %d/%d, loss: %.2e    " % (batch_idx + 1, train_reader.nbatches, loss_value.numpy()))
            stdout.flush()
            if (batch_idx + 1) % 10 == 0:
                display_training(batch_imgs, disps[-1], image_from_before, image_from_after, image_from_opposite)
            step_count += 1
        stdout.write('\n')
        print('Epoch computed in ' + str(datetime.now() - epoch_start))

        evaluation_loop(val_reader, step_count)

        # Save models:
        print('Saving models')
        pose_net.save_weights(os.path.join(save_dir, 'pose_net_' + str(epoch), 'weights'))
        depth_net.save_weights(os.path.join(save_dir, 'depth_net_' + str(epoch), 'weights'))
