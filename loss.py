import tensorflow as tf
from transformations import BackprojectLayer, WarpLayer
from models import disp2depth


def SSIM(x, y):
    # x, y: (batch_size, height, width, 3)
    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='reflect')  # (batch_size, height + 2, width + 2, 3)
    y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='reflect')  # (batch_size, height + 2, width + 2, 3)
    mu_x = tf.nn.avg_pool(x, 3, 1, padding='VALID')  # (batch_size, height, width, 3)
    mu_y = tf.nn.avg_pool(y, 3, 1, padding='VALID')  # (batch_size, height, width, 3)
    sigma_x = tf.nn.avg_pool(x * x, 3, 1, padding='VALID') - mu_x * mu_x  # (batch_size, height, width, 3)
    sigma_y = tf.nn.avg_pool(y * y, 3, 1, padding='VALID') - mu_y * mu_y  # (batch_size, height, width, 3)
    sigma_xy = tf.nn.avg_pool(x * y, 3, 1, padding='VALID') - mu_x * mu_y  # (batch_size, height, width, 3)
    C1 = 0.01 * 0.01
    C2 = 0.03 * 0.03
    SSIM_n = (2.0 * mu_x * mu_y + C1) * (2.0 * sigma_xy + C2)  # (batch_size, height, width, 3)
    SSIM_d = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2)  # (batch_size, height, width, 3)
    SSIM_loss = tf.clip_by_value((1.0 - SSIM_n / SSIM_d) / 2.0, 0.0, 1.0)  # (batch_size, height, width, 3)
    SSIM_loss = tf.reduce_mean(SSIM_loss, axis=-1)  # (batch_size, height, width)
    return SSIM_loss


def compute_reprojection_loss(x, y):
    # x, y: (batch_size, height, width, 3)
    # Note: pixel values of images must be between 0 and 1.
    abs_diff = tf.math.abs(x - y)  # (batch_size, height, width, 3)
    l1_loss = tf.reduce_mean(abs_diff, axis=-1)  # (batch_size, height, width)
    ssim_loss = SSIM(x, y)  # (batch_size, height, width)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss  # (batch_size, height, width)
    return reprojection_loss


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

    return tf.reduce_mean(loss_i + loss_j)


# Note: I don't know if this fits the requirements of a layer loss for Keras. But
# this is just how I need to be and it works with my training loop (I hope).
class LossLayer:
    def __init__(self, K, height, width, batch_size):
        # K: intrinsic parameters matrix, (3, 3)
        self.K = K
        self.height = height
        self.width = width
        self.backproject = BackprojectLayer(K, height, width, batch_size)
        self.warp = WarpLayer(K, height, width, batch_size)
        self.smoothness_factor = 1e-3

    def __call__(self, disps, T_before_target, T_after_target, T_opposite_target,
                 img_before, img_target, img_after, img_opposite):
        # disps: list with tensors of shape (batch_size, ?, ?, 1)
        # T_before_target: (batch_size, 4, 4)
        # T_after_target: (batch_size, 4, 4)
        # T_opposite_target: (batch_size, 4, 4)
        # img_before: (batch_size, height, width, 3)
        # img_target: (batch_size, height, width, 3)
        # img_after: (batch_size, height, width, 3)
        # img_opposite: (batch_size, height, width, 3)
        num_scales = len(disps)
        loss = tf.zeros((), tf.float32)

        image_from_before = None
        image_from_after = None
        image_from_opposite = None

        for scale_idx in range(num_scales):
            scale_factor = 2.0 ** scale_idx
            scaled_height = int(self.height / (2.0 ** scale_idx))
            scaled_width = int(self.width / (2.0 ** scale_idx))
            # The disparity and depth correspond to the "target" image.
            # The disparities are in increasing resolution, whilst this loop goes in the
            # other direction.
            disp_original_size = disps[num_scales - scale_idx - 1]  # (bs, scaled_height, scaled_width, 1)
            assert disp_original_size.shape[1] == scaled_height
            assert disp_original_size.shape[2] == scaled_width
            disp = tf.image.resize(disp_original_size, (self.height, self.width))  # (bs, h, w, 1)
            depth = disp2depth(disp)  # (bs, h, w, 1)
            points3d_hom_target = self.backproject(depth)  # (bs, h, w, 4)
            target_image_from_before = self.warp(img_before, points3d_hom_target, T_before_target)
            target_image_from_after = self.warp(img_after, points3d_hom_target, T_after_target)
            target_image_from_opposite = self.warp(img_opposite, points3d_hom_target, T_opposite_target)

            if scale_idx == 0:
                image_from_before = target_image_from_before
                image_from_after = target_image_from_after
                image_from_opposite = target_image_from_opposite

            # All the losses below have shape (bs, h, w)
            reprojection_loss_before = compute_reprojection_loss(img_target, target_image_from_before)
            reprojection_loss_after = compute_reprojection_loss(img_target, target_image_from_after)
            reprojection_loss_opposite = compute_reprojection_loss(img_target, target_image_from_opposite)

            identity_loss_before = compute_reprojection_loss(img_target, img_before)
            identity_loss_after = compute_reprojection_loss(img_target, img_after)
            identity_loss_opposite = compute_reprojection_loss(img_target, img_opposite)  # TODO: Is this necessary?

            # (bs, h, w, 4):
            reprojection_losses = tf.stack([reprojection_loss_before, reprojection_loss_after, reprojection_loss_opposite,
                                            identity_loss_before, identity_loss_after, identity_loss_opposite], axis=-1)

            min_reprojection_loss = tf.reduce_min(reprojection_losses, axis=-1)  # (bs, h, w)

            loss += tf.reduce_mean(min_reprojection_loss)

            # The smoothness loss is computed on the scaled size:
            img_target_scaled = tf.image.resize(img_target, (scaled_height, scaled_width))
            assert img_target_scaled.shape[1] == disp_original_size.shape[1]
            assert img_target_scaled.shape[2] == disp_original_size.shape[2]
            smoothness_loss = compute_smoothness_loss(disp_original_size, img_target_scaled)
            loss += smoothness_loss * self.smoothness_factor / scale_factor

        loss /= float(num_scales)

        return loss, image_from_before, image_from_after, image_from_opposite
