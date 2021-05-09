import tensorflow as tf
import tensorflow_probability as tfp
from models3 import disp2depth


def compute_metrics(disp_pred, depth_gt):
    # disp_pred: (batch_size, height, width, 1)
    # depth_gt: (batch_size, height, width)

    # Note: I'm computing the metrics over the entire batch here. Maybe it would be more
    # correct to compute it over each image, and then average. But I suspect the results
    # will be very similar, and this way is easier.

    depth_pred = disp2depth(disp_pred)  # (batch_size, height, width, 1)

    # Flatten:
    depth_pred = tf.reshape(depth_pred, (-1))
    depth_gt = tf.reshape(depth_gt, (-1))

    # Zero elements in ground truth correspond to places where we don't have depth information.
    gt_mask = depth_gt > 0.1
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
