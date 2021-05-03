import tensorflow as tf
import numpy as np
from datetime import datetime
from sys import stdout

from data_reader import DataReader, AsyncParallelReader, ReaderOpts
from transformations3 import make_transformation_matrix, concat_images
from loss3 import LossLayer
from models3 import build_depth_net, build_pose_net

img_height = 192
img_width = 640

kitti_path = '/home/xian/kitti_data_monodepth2'
batch_size = 2
reader_opts = ReaderOpts(kitti_path, batch_size)
# train_dataset = DataReader(kitti_path, batch_size)

nepochs = 20

K = np.array([[0.58 * img_width, 0, 0.5 * img_width],
              [0, 1.92 * img_height, 0.5 * img_height],
              [0, 0, 1]], dtype=np.float32)


depth_net = build_depth_net(img_height, img_width)
pose_net = build_pose_net(img_height, img_width)

trainable_weights = depth_net.trainable_weights
trainable_weights.extend(pose_net.trainable_weights)

optimizer = tf.keras.optimizers.Adam()  # TODO

loss_layer = LossLayer(K, img_height, img_width, batch_size)


# @tf.function
def train_step(batch_imgs):
    # batch_imgs: (batch_size, img_size, img_size, 9) Three images concatenated on the channels dimension
    img_before = batch_imgs[:, :, :, :3]
    img_target = batch_imgs[:, :, :, 3:6]
    img_after = batch_imgs[:, :, :, 6:]

    with tf.GradientTape() as tape:
        disps = depth_net(img_target)  # disparities at different scales, in increasing resolution
        # TODO: I can spare one of this concatenations, but for now this is clearer:
        T_before_target = pose_net(concat_images(img_before, img_target))  # (batch_size, 6)
        T_target_after = pose_net(concat_images(img_target, img_after))  # (batch_size, 6)
        matrixT_before_target = make_transformation_matrix(T_before_target, False)  # (bs, 4, 4)
        matrixT_after_target = make_transformation_matrix(T_target_after, True)  # (bs, 4, 4)
        loss_value = loss_layer(disps, matrixT_before_target, matrixT_after_target, img_before, img_target, img_after)

    grads = tape.gradient(loss_value, trainable_weights)
    optimizer.apply_gradients(zip(grads, trainable_weights))

    return loss_value


with AsyncParallelReader(reader_opts) as train_reader:
    for epoch in range(nepochs):
        print("\nStart epoch ", epoch + 1)
        epoch_start = datetime.now()
        for batch_idx in range(train_reader.nbatches):
            batch_imgs = train_reader.get_batch()
            loss_value = train_step(batch_imgs)
            stdout.write("\rbatch %d/%d, loss: %.2e    " % (batch_idx + 1, train_reader.nbatches, loss_value.numpy()))
            stdout.flush()
        stdout.write('\n')
        print('Epoch computed in ' + str(datetime.now() - epoch_start))
