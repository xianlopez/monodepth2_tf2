import tensorflow as tf
import tensorflow_probability as tfp
from models import disp2depth
import numpy as np


depth_height = 375
depth_width = 1242

# To build the first mask for the depth (the constant valid area), I'll make
# it first in numpy for simplicity (tensorflow doesn't allow item assignment):
mask1 = np.zeros((depth_height, depth_width), np.bool)
mask1[153:371, 44:1197] = 1
# Convert to tensorflow and then reshape (I'm pretty sure the reshape would be
# the same in numpy, but just in case):
mask1 = tf.convert_to_tensor(mask1, tf.bool)
mask1 = tf.reshape(mask1, [-1])


def compute_metrics(disp_pred, depth_gt):
    # disp_pred: (batch_size, height, width, 1)
    # depth_gt: (batch_size, depth_height, depth_width)

    # Note: I'm computing the metrics over the entire batch here. Maybe it would be more
    # correct to compute it over each image, and then average. But I suspect the results
    # will be very similar, and this way is easier.

    batch_size = depth_gt.shape[0]

    depth_pred = disp2depth(disp_pred)  # (batch_size, height, width, 1)

    depth_pred = tf.image.resize(depth_pred, (depth_height, depth_width))

    # Flatten:
    depth_pred = tf.reshape(depth_pred, [-1])
    depth_gt = tf.reshape(depth_gt, [-1])

    # Zero elements in ground truth correspond to places where we don't have depth information.
    mask2 = depth_gt > 0.1

    # Batch mask1:  TODO: This could be done just once at the beginning
    mask1_batched = tf.tile(mask1, [batch_size])

    # Combine masks:
    gt_mask = tf.logical_and(mask1_batched, mask2)

    # Keep relevant elements:
    depth_pred = tf.boolean_mask(depth_pred, gt_mask)
    depth_gt = tf.boolean_mask(depth_gt, gt_mask)

    # Scale predicted depth:
    median_gt = tfp.stats.percentile(depth_gt, 50.0)
    median_pred = tfp.stats.percentile(depth_pred, 50.0)
    depth_pred *= median_gt / median_pred

    depth_pred = tf.clip_by_value(depth_pred, 1e-3, 80)

    delta = tf.maximum((depth_gt / depth_pred), (depth_pred / depth_gt))
    a1 = tf.reduce_mean(tf.cast(delta < 1.25, tf.float32))
    a2 = tf.reduce_mean(tf.cast(delta < 1.25 ** 2, tf.float32))
    a3 = tf.reduce_mean(tf.cast(delta < 1.25 ** 3, tf.float32))

    rmse = tf.sqrt(tf.reduce_mean(tf.square(depth_gt - depth_pred)))

    rmse_log = tf.sqrt(tf.reduce_mean(tf.square(tf.math.log(depth_gt) - tf.math.log(depth_pred))))

    abs_rel = tf.reduce_mean(tf.math.abs(depth_gt - depth_pred) / depth_gt)

    sq_rel = tf.reduce_mean(tf.square(depth_gt - depth_pred) / depth_gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
