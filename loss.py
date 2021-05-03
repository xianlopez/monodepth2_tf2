import tensorflow as tf


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


def compute_loss(images, y_pred):
    # images: contains the input images, concatenated over the channels dimension: (batch_size, height, width, 3 * 3)
    # The images are in temporal order. Let's call them "previous" (p), "current" (c) and "next" (n).
    # y_pred: (batch_size, height, width, 3 * 2 * 4) The warped images, in the following order:
    # (scale0_previous, scale0_next, scale1_previous, scale1_next, scale2_previous, scale2_next,
    # scale3_previous, scale3_next)

    n_disparities = 4
    assert y_pred.shape[3] == 3 * 2 * n_disparities

    _, height, width, _ = images.shape
    previous_imgs = images[:, :, :, :3]
    current_imgs = images[:, :, :, 3:6]
    next_imgs = images[:, :, :, 6:]  # (batch_size, height, width, 3)

    # Identity reprojection losses:
    identity_reprojection_loss_previous = compute_reprojection_loss(current_imgs, previous_imgs)
    identity_reprojection_loss_next = compute_reprojection_loss(current_imgs, next_imgs)

    losses_on_scales = []
    for scale_idx in range(n_disparities):
        # Note: smoothness loss was added in the network definition. TODO
        reprojection_losses = []
        start_idx = scale_idx * 6
        prev_imgs_warped = y_pred[..., start_idx:(start_idx+3)]
        reprojection_losses.append(compute_reprojection_loss(prev_imgs_warped, current_imgs))
        next_imgs_warped = y_pred[..., (start_idx+3):(start_idx+6)]
        reprojection_losses.append(compute_reprojection_loss(next_imgs_warped, current_imgs))

        # Add the identity reprojection losses, so they can be selected in the minimum operation:
        reprojection_losses.append(identity_reprojection_loss_previous)
        reprojection_losses.append(identity_reprojection_loss_next)

        reprojection_losses = tf.stack(reprojection_losses, axis=-1)  # (batch_size, height, width, 2 + 2)

        # Take the minimum over the 4 reprojectin losses computed:
        min_reproj_loss = tf.reduce_min(reprojection_losses, axis=-1)  # (batch_size, height, width)

        # Take mean over spatial dimensions:
        losses_on_scales.append(tf.reduce_mean(min_reproj_loss, axis=[1, 2]))  # (batch_size))

    loss = tf.math.accumulate_n(losses_on_scales) / float(n_disparities)  # (batch_size))

    # Despite of all the info I find only, it seems that Keras wants a scalar as output for the loss.
    return tf.reduce_mean(loss)
