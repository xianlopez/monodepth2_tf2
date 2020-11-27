import tensorflow as tf
import transformations


def disp_to_depth(disp, min_depth=0.1, max_depth=100.0):
    min_disp = 1.0 / max_depth
    max_disp = 1.0 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1.0 / scaled_disp
    return depth


def SSIM(x, y):
    # x, y: (batch_size, height, width, 3)
    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='reflect')  # (batch_size, height + 2, width + 2, 3)
    y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='reflect')  # (batch_size, height + 2, width + 2, 3)
    mu_x = tf.nn.avg_pool(x, 3, 1, padding='valid')  # (batch_size, height, width, 3)
    mu_y = tf.nn.avg_pool(y, 3, 1, padding='valid')  # (batch_size, height, width, 3)
    sigma_x = tf.nn.avg_pool(x * x, 3, 1, padding='valid') - mu_x * mu_x  # (batch_size, height, width, 3)
    sigma_y = tf.nn.avg_pool(y * y, 3, 1, padding='valid') - mu_y * mu_y  # (batch_size, height, width, 3)
    sigma_xy = tf.nn.avg_pool(x * y, 3, 1, padding='valid') - mu_x * mu_y  # (batch_size, height, width, 3)
    C1 = 0.01 * 0.01
    C2 = 0.03 * 0.03
    SSIM_n = (2.0 * mu_x * mu_y + C1) * (2.0 * sigma_xy + C2)  # (batch_size, height, width, 3)
    SSIM_d = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2)  # (batch_size, height, width, 3)
    SSIM_loss = tf.clip_by_value((1.0 - SSIM_n / SSIM_d) / 2.0, 0.0, 1.0)  # (batch_size, height, width, 3)
    SSIM_loss = tf.reduce_mean(SSIM_loss, axis=-1)  # (batch_size, height, width)
    assert SSIM_loss.shape == x.shape[:3]
    return


def compute_reprojection_loss(x, y):
    # x, y: (batch_size, height, width, 3)
    # TODO: pixel values of images must be between 0 and 1.
    abs_diff = tf.math.abs(x - y)  # (batch_size, height, width, 3)
    l1_loss = tf.reduce_mean(abs_diff, axis=-1)  # (batch_size, height, width)
    ssim_loss = SSIM(x, y)  # (batch_size, height, width)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss  # (batch_size, height, width)
    assert reprojection_loss.shape == x.shape[:3]
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

    return tf.reduce_mean(grad_disp_i) + tf.reduce_mean(grad_disp_j)  # ()


def myloss(y_true, y_pred):
    # y_true: contains the input images, concatenated over the channels dimension: (batch_size, height, width, 3 * 3)
    # The images are in temporal order. Let's call them "previous" (p), "current" (c) and "next" (n).
    # y_pred: list with (disp0, disp1, disp2, disp3, pose_net_output)
    # disp0: (batch_size, height, width, 1)
    # disp1: (batch_size, height / 2, width / 2, 1)
    # disp2: (batch_size, height / 4, width / 4, 1)
    # disp3: (batch_size, height / 8, width / 8, 1)
    # pose_net_output: (batch_size, 2 * 6)

    _, height, width, _ = y_true.shape
    previous_imgs = y_true[:, :, :, :3]
    current_imgs = y_true[:, :, :, 3:6]
    next_imgs = y_true[:, :, :, 6:]  # (batch_size, height, width, 3)

    disp0, disp1, disp2, disp3, pose_net_output = y_pred
    # Tcp refers to the pose of the previous camera in the current frame (and the opposite for Tpc).
    # Tcn refers to the pose of the next camera in the current frame (and the opposite for Tnc).
    axisangleTcp = pose_net_output[:, :3]
    translationTcp = pose_net_output[:, 3:6]
    axisangleTnc = pose_net_output[:, 6:9]
    translationTnc = pose_net_output[:, 9:]

    # Upsample disparities to original size:
    # TODO: This could actually be done in the network.
    disp1 = tf.image.resize(disp1, [height, width])  # (batch_size, height, width, 1)
    disp2 = tf.image.resize(disp2, [height, width])  # (batch_size, height, width, 1)
    disp3 = tf.image.resize(disp3, [height, width])  # (batch_size, height, width, 1)
    all_disparities = [disp0, disp1, disp2, disp3]

    # Identity reprojection losses:
    identity_reprojection_loss_previous = compute_reprojection_loss(current_imgs, previous_imgs)
    identity_reprojection_loss_next = compute_reprojection_loss(current_imgs, next_imgs)

    losses_on_scales = []
    for scale_idx in range(len(all_disparities)):
        disparity = all_disparities[scale_idx]
        depth = disp_to_depth(disparity)
        # Warp the previous and current images with the computed transformations, and compute the loss comparing them
        # agains the current one.
        reprojection_losses = []
        prev_imgs_warped = transformations.warp_images(previous_imgs, depth, K, Kinv, axisangleTcp, translationTcp, False)
        reprojection_losses.append(compute_reprojection_loss(prev_imgs_warped, current_imgs))
        next_imgs_warped = transformations.warp_images(next_imgs, depth, K, Kinv, axisangleTnc, translationTnc, True)
        reprojection_losses.append(compute_reprojection_loss(next_imgs_warped, current_imgs))

        # Add the identity reprojection losses, so they can be selected in the minimum operation:
        reprojection_losses.append(identity_reprojection_loss_previous)
        reprojection_losses.append(identity_reprojection_loss_next)

        reprojection_losses = tf.stack(reprojection_losses, axis=-1)  # (batch_size, height, width, 2 + 2)

        loss_this_scale = tf.reduce_mean(tf.reduce_min(reprojection_losses, axis=-1))
        smooth_loss = compute_smoothness_loss(disparity, current_imgs)
        loss_this_scale += 1e-3 * smooth_loss / (2.0 ** scale_idx)
        losses_on_scales.append(loss_this_scale)

    loss = tf.math.accumulate_n(losses_on_scales) / len(all_disparities)

    return loss
