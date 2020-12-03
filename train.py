import tensorflow as tf
import numpy as np

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
def train_step(batch_imgs):
    with tf.GradientTape() as tape:
        net_output = model(batch_imgs, training=True)
        loss_value = compute_loss(batch_imgs, net_output)
        loss_value += sum(model.losses)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value

for epoch in range(nepochs):
    for batch_idx in range(train_dataset.__len__()):
        batch_imgs, _ = train_dataset.__getitem__(batch_idx)
        loss_value = train_step(tf.constant(batch_imgs, tf.float32))
        print('epoch %d, batch %d, loss: %.4f' % (epoch, batch_idx, loss_value))
    train_dataset.on_epoch_end()
