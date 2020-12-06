import tensorflow as tf
import transformations
import numpy as np


class Disp2Depth(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Disp2Depth, self).__init__(**kwargs)

    def call(self, disp):
        # disp: (batch_size, height, width, 1)
        depth = disp_to_depth(disp)
        return depth


def compute_smoothness_loss(disparity, image):
    # disparity: (batch_size, height, width, 1)
    # image: (batch_size, height, width, 3)
    # The image is used for edge-aware smoothness.
    mean_disp = tf.reduce_mean(disparity, axis=[1, 2], keepdims=True)  # (batch_size, 1, 1, 1)
    normalized_disp = disparity / (mean_disp + 1e-7)  # (batch_size, height, width, 1)

    grad_disp_i = tf.abs(normalized_disp[:, :-1, :, :] - normalized_disp[:, 1:, :, :])
    grad_disp_j = tf.abs(normalized_disp[:, :, :-1, :] - normalized_disp[:, :, 1:, :])

    grad_img_i = tf.reduce_mean(tf.abs(image[:, :-1, :, :] - image[:, 1:, :, :]), axis=-1, keepdims=True)
    grad_img_j = tf.reduce_mean(tf.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), axis=-1, keepdims=True)

    # Weight the disparity gradients with the image gradients:
    grad_disp_i *= tf.exp(-grad_img_i)  # (batch_size, height - 1, width, 1)
    grad_disp_j *= tf.exp(-grad_img_j)  # (batch_size, height, width - 1, 1)

    # Squeeze and take mean along spatial dimensions:
    loss_i = tf.reduce_mean(tf.squeeze(grad_disp_i, axis=3), axis=[1, 2])  # (batch_size)
    loss_j = tf.reduce_mean(tf.squeeze(grad_disp_j, axis=3), axis=[1, 2])  # (batch_size)

    return loss_i + loss_j  # (batch_size)


def disp_to_depth(disp, min_depth=0.1, max_depth=100.0):
    min_disp = 1.0 / max_depth
    max_disp = 1.0 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1.0 / scaled_disp
    return depth


class WarpInputs(tf.keras.layers.Layer):
    def __init__(self, K, **kwargs):
        # K: intrinsic parameters matrix, 3x3.
        assert K.shape == (3, 3)
        super(WarpInputs, self).__init__(**kwargs)
        self.K = K
        self.Kinv = np.linalg.inv(K)

    def call(self, inputs):
        # inputs: [images, pose_net_output, disp0, disp1, disp2, disp3]
        # images: (batch_size, height, width, 9)
        # pose_net_output: (batch_size, 2 * 6)
        # disp*: (batch_size, height, width, 1)

        images, pose_net_output, disp0, disp1, disp2, disp3 = inputs

        previous_imgs = images[:, :, :, :3]
        current_imgs = images[:, :, :, 3:6]
        next_imgs = images[:, :, :, 6:]  # (batch_size, height, width, 3)

        # Tcp refers to the pose of the previous camera in the current frame (and the opposite for Tpc).
        # Tcn refers to the pose of the next camera in the current frame (and the opposite for Tnc).
        axisangleTcp = pose_net_output[:, :3]
        translationTcp = pose_net_output[:, 3:6]
        axisangleTnc = pose_net_output[:, 6:9]
        translationTnc = pose_net_output[:, 9:]

        all_disparities = [disp0, disp1, disp2, disp3]
        warped_images = []
        for disp in all_disparities:
            depth = disp_to_depth(disp)
            warped_images.append(transformations.warp_images(
                previous_imgs, depth, self.K, self.Kinv, axisangleTcp, translationTcp, False))
            warped_images.append(transformations.warp_images(
                next_imgs, depth, self.K, self.Kinv, axisangleTnc, translationTnc, True))

        # Smoothness loss on disparities:
        for scale_idx in range(len(all_disparities)):
            disp = all_disparities[scale_idx]
            # self.add_loss(1e-3 * compute_smoothness_loss(disp, current_imgs) / (2.0 ** scale_idx))
            # Despite of all the info I find only, it seems that Keras wants a scalar as output for the loss.
            self.add_loss(1e-3 * tf.reduce_mean(compute_smoothness_loss(disp, current_imgs)) / (2.0 ** scale_idx))

        return tf.concat(warped_images, axis=-1)  # (batch_size, height, width, 3 * 2 * 4)

