import tensorflow as tf
import numpy as np
import os
import shutil
from datetime import datetime
from sys import stdout

from model import training_model
from loss import compute_loss
from data_reader import DataReader, AsyncParallelReader, ReaderOpts
from data_reader import height, width
from display import display_depth

K = np.array([[0.58 * width, 0, 0.5 * width],
              [0, 1.92 * height, 0.5 * height],
              [0, 0, 1]], dtype=np.float32)

model = training_model(K)
optimizer = tf.optimizers.Adam(learning_rate=1e-4)
# model.compile(loss=compute_loss, optimizer=optimizer)

kitti_path = '/home/xian/kitti_data_monodepth2'
batch_size = 6
reader_opts = ReaderOpts(kitti_path, batch_size)
# train_dataset = DataReader(kitti_path, batch_size)

nepochs = 20

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', update_freq='batch')

# TODO: Data augmentation
# TODO: Visualization
# TODO: Saving
# TODO: Resnet18


def build_train_step_fun(model, loss, optimizer, train_summary_writer):
    @tf.function
    def train_step_fun(batch_imgs, step):
        with tf.GradientTape() as tape:
            warped_images, disparities = model(batch_imgs, training=True)
            loss_value = loss(batch_imgs, warped_images)
            loss_value += sum(model.losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss_value, step=step)
        return loss_value, disparities
    return train_step_fun

train_log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
if os.path.exists(train_log_dir):
    shutil.rmtree(train_log_dir)
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
train_summary_writer.set_as_default()

train_step_fun = build_train_step_fun(model, compute_loss, optimizer, train_summary_writer)

period_display = 10

with AsyncParallelReader(reader_opts) as train_reader:
    train_step = -1
    for epoch in range(nepochs):
        print("\nStart epoch ", epoch + 1)
        # Training:
        epoch_start = datetime.now()
        for batch_idx in range(train_reader.nbatches):
            train_step += 1
            batch_imgs = train_reader.get_batch()
            loss_value, disparities = train_step_fun(batch_imgs, tf.cast(tf.convert_to_tensor(train_step), tf.int64))
            stdout.write("\rbatch %d/%d, loss: %.2e    " % (batch_idx + 1, train_reader.nbatches, loss_value.numpy()))
            stdout.flush()
            train_summary_writer.flush()

            if (batch_idx + 1) % period_display == 0:
                display_depth(disparities, batch_imgs)
        stdout.write('\n')
        print('Epoch computed in ' + str(datetime.now() - epoch_start))


# model.fit(train_dataset, epochs=20, callbacks=[tensorboard_callback])
