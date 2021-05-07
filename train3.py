import tensorflow as tf
import numpy as np
from datetime import datetime
from sys import stdout
import shutil
import os

from data_reader import AsyncParallelReader, ReaderOpts
from transformations3 import make_transformation_matrix, concat_images
from loss3 import LossLayer
from models3 import build_depth_net, build_pose_net
from drawing3 import display_training

# TODO: Data augmentation
# TODO: Saving
# TODO: Validation
# TODO: Learning rate schedule
# TODO: Metrics

img_height = 192
img_width = 640

kitti_path = '/home/xian/kitti_data'
batch_size = 8
reader_opts = ReaderOpts(kitti_path, batch_size, img_height, img_width, 8)

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


@tf.function
def train_step(batch_imgs, batch_depth, step_count):
    # batch_imgs: (batch_size, height, width, 9) Three images concatenated on the channels dimension
    # batch_depth: (batch_size, height, width) Ground truth depth for target image (used only to compute metrics)
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


with AsyncParallelReader(reader_opts) as train_reader:
    step_count = 0
    for epoch in range(nepochs):
        print("\nStart epoch ", epoch + 1)
        epoch_start = datetime.now()
        for batch_idx in range(train_reader.nbatches):
            batch_imgs, batch_depth = train_reader.get_batch()
            loss_value, disps, image_from_before, image_from_after =\
                train_step(batch_imgs, batch_depth, tf.constant(step_count, tf.int64))
            train_summary_writer.flush()
            stdout.write("\rbatch %d/%d, loss: %.2e    " % (batch_idx + 1, train_reader.nbatches, loss_value.numpy()))
            stdout.flush()
            if (batch_idx + 1) % 10 == 0:
                display_training(batch_imgs, disps, image_from_before, image_from_after)
            step_count += 1
        stdout.write('\n')
        print('Epoch computed in ' + str(datetime.now() - epoch_start))
