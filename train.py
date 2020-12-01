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
model.compile(loss=compute_loss, optimizer=optimizer)

kitt_path = '/home/xian/KITTI'
batch_size = 12
train_dataset = DataReader(kitt_path, batch_size)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs', update_freq='batch')

# TODO: Data augmentation
# TODO: Visualization
# TODO: Saving
# TODO: Resnet18

model.fit(train_dataset, epochs=20, callbacks=[tensorboard_callback])
