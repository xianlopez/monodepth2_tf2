import tensorflow as tf
import numpy as np
import os
import shutil

from model import training_model
from loss import compute_loss
from data_reader import DataReader
from data_reader import height, width

K = np.array([[0.58 * width, 0, 0.5 * width],
              [0, 1.92 * height, 0.5 * height],
              [0, 0, 1]], dtype=np.float32)

model = training_model(K)
optimizer = tf.optimizers.Adam(learning_rate=1e-4)
# model.compile(loss=compute_loss, optimizer=optimizer)

train_log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
if os.path.exists(train_log_dir):
    shutil.rmtree(train_log_dir)
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
train_summary_writer.set_as_default()

kitt_path = '/home/xian/KITTI'
batch_size = 12
nepochs = 20
train_dataset = DataReader(kitt_path, batch_size)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', update_freq='batch')

# TODO: Data augmentation
# TODO: Visualization
# TODO: Saving
# TODO: Resnet18

@tf.function
def train_step(batch_imgs, step):
    with tf.GradientTape() as tape:
        net_output = model(batch_imgs, training=True)
        loss_value = compute_loss(batch_imgs, net_output)
        loss_value += sum(model.losses)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', loss_value, step=step)
    return loss_value

step = 0
for epoch in range(nepochs):
    for batch_idx in range(train_dataset.__len__()):
        batch_imgs, _ = train_dataset.__getitem__(batch_idx)
        loss_value = train_step(tf.constant(batch_imgs, tf.float32), tf.constant(step, tf.int64))
        train_summary_writer.flush()
        print('epoch %d / %d, batch %d / %d, loss: %.4f' % (epoch + 1, nepochs, batch_idx + 1,
                                                            train_dataset.__len__(), loss_value))
        step += 1
    train_dataset.on_epoch_end()
