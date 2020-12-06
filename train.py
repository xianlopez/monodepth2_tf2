import tensorflow as tf
import numpy as np
import os
import shutil
from sys import stdout
import ast
import argparse
from datetime import datetime

from model import training_model
from loss import compute_loss
from data_reader import DataReader, height, width
import tools
from drawing import display_training

# TODO: Data augmentation
# TODO: Resnet18


K = np.array([[0.58 * width, 0, 0.5 * width],
              [0, 1.92 * height, 0.5 * height],
              [0, 0, 1]], dtype=np.float32)


def lr_schedule(current_epoch, epoch_lr_pairs, initial_lr):
    prev_starting_epoch = np.inf
    for starting_epoch, lr in epoch_lr_pairs[::-1]:
        assert starting_epoch < prev_starting_epoch
        if current_epoch >= starting_epoch:
            return lr
    return initial_lr


def build_train_step_fun(model, loss_fun, optimizer, train_summary_writer):
    @tf.function
    def train_step_fun(batch_imgs, step):
        with tf.GradientTape() as tape:
            net_output = model(batch_imgs, training=True)
            loss_value = loss_fun(batch_imgs, net_output)
            loss_value += sum(model.losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss_value, step=step)
        return loss_value, net_output
    return train_step_fun


def train(kitti_path, ckpt_idx, batch_size, nepochs, initial_lr, epoch_lr_pairs, period_display):
    model = training_model(K)
    optimizer = tf.optimizers.Adam(learning_rate=initial_lr)
    # model.compile(loss=compute_loss, optimizer=optimizer)

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    if ckpt_idx:
        checkpoint_to_load = 'ckpts/ckpt_' + str(ckpt_idx)
        read_result = checkpoint.read(checkpoint_to_load)
        read_result.assert_existing_objects_matched()

    train_dataset = DataReader(kitti_path, batch_size)

    train_log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')
    if os.path.exists(train_log_dir):
        shutil.rmtree(train_log_dir)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    train_summary_writer.set_as_default()

    train_step = build_train_step_fun(model, compute_loss, optimizer, train_summary_writer)

    step = 0
    for epoch in range(1, nepochs + 1):
        print("\nStart epoch ", epoch)
        optimizer.learning_rate = lr_schedule(epoch, epoch_lr_pairs, initial_lr)
        print('Learning rate: %.2e' % optimizer.learning_rate)
        epoch_start = datetime.now()
        for batch_idx in range(train_dataset.__len__()):
            batch_imgs, _ = train_dataset.__getitem__(batch_idx)
            loss_value, net_output = train_step(tf.constant(batch_imgs, tf.float32), tf.constant(step, tf.int64))
            train_summary_writer.flush()
            stdout.write("\rbatch %d/%d, loss: %.2e    " % (batch_idx + 1, train_dataset.__len__(), loss_value.numpy()))
            stdout.flush()
            if (batch_idx + 1) % period_display == 0:
                display_training(batch_imgs, net_output)
            step += 1
        stdout.write('\n')
        train_dataset.on_epoch_end()
        print('Epoch computed in ' + str(datetime.now() - epoch_start))

        # Save model:
        print('Saving model')
        checkpoint.write('ckpts/ckpt_' + str(epoch))
        # Erase last epoch's checkpoint:
        if epoch > 1:
            tools.delete_checkpoint_with_index(epoch - 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument('--kitti_path', type=str, default='/home/xian/KITTI', help='path to KITTI dataset')
    parser.add_argument('--ckpt_idx', type=int, help='index of the checkpoint to load the initial weights')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--nepochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--lr_changes', default='[(15, 1e-5)]',
                        help='changes in learning rate, as a list of tuples where the first element is the epoch from '
                             'which the second one (learning rate) applies')
    parser.add_argument('--period_display', type=int, default=10,
                        help='number of batches between two consecutive displays')
    args = parser.parse_args()

    epoch_lr_pairs = ast.literal_eval(args.lr_changes)

    train(args.kitti_path, args.ckpt_idx, args.batch_size, args.nepochs, args.learning_rate,
          epoch_lr_pairs, args.period_display)
