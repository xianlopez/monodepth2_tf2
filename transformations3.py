import tensorflow as tf
import numpy as np


def concat_images(imgs1, imgs2):
    # imgs1: (batch_size, h, w, 3)
    # imgs2: (batch_size, h, w, 3)
    all_imgs = tf.concat([imgs1, imgs2], axis=-1)
    return all_imgs  # (batch_size, h, w, 6)


class BackprojectLayer:
    def __init__(self, K, height, width, batch_size):
        # K: intrinsics matrix (3, 3)
        Kinv = np.linalg.inv(K)
        x = tf.cast(tf.range(width), tf.float32)
        y = tf.cast(tf.range(height), tf.float32)
        X, Y = tf.meshgrid(x, y)  # (h, w)
        pixel_coords_hom = tf.stack([X, Y, tf.ones_like(X)], axis=-1)  # (h, w, 3)
        rays = tf.linalg.matvec(Kinv, pixel_coords_hom)  # (h, w, 3)
        self.rays = tf.tile(tf.expand_dims(rays, axis=0), [batch_size, 1, 1, 1])  # (batch_size, h, w, 3)
        self.ones = tf.ones((batch_size, height, width, 1), tf.float32)

    def __call__(self, depth):
        # depth: (batch_size, h, w, 1)
        assert depth.shape == self.ones.shape
        points3d = depth * self.rays  # (batch_size, h, w, 3)
        points3d_hom = tf.concat([points3d, self.ones], axis=-1)
        return points3d_hom  # (batch_size, h, w, 4)


class ProjectLayer:
    def __init__(self, K):
        # K: (3, 3) Intrinsics parameters matrix.
        self.K_ext = tf.concat([K, tf.zeros((3, 1), tf.float32)], axis=1)  # (3, 4)

    def __call__(self, points3d_hom):
        # points3d_hom: (batch_size, h, w, 4)
        pixel_coords_hom = tf.linalg.matvec(self.K_ext, points3d_hom)  # (batch_size, h, w, 3)
        pixel_coords = pixel_coords_hom[:, :, :, :2] / (tf.expand_dims(pixel_coords_hom[:, :, :, 2], axis=-1) + 1e-7)
        return pixel_coords  # (batch_size, h, w, 2)


class WarpLayer:
    def __init__(self, K):
        # K: (3, 3) Intrinsics parameters matrix.
        self.project = ProjectLayer(K)

    def __call__(self, image_source, points3d_target, T_source_target):
        # image_source: (batch_size, h, w, 3)
        # points3d_target: (batch_size, h, w, 4)
        # T_source_target: (batch_size, 4, 4)
        points3d_source = transform3d(T_source_target, points3d_target)
        pix_coords_source = self.project(points3d_source)
        image_target = bilinear_interpolation(image_source, pix_coords_source)
        return image_target


def transform3d(T21, points3d_hom_1):
    # T21: (batch_size, 4, 4)
    # points3d_hom_1: (batch_size, h, w, 4)
    T21_ext = tf.expand_dims(tf.expand_dims(T21, axis=1), axis=1)  # (batch_size, 1, 1, 4, 4)
    points3d_hom_2 = tf.linalg.matvec(T21_ext, points3d_hom_1)
    return points3d_hom_2  # (batch_size, h, w, 4)


def evaluate_tensor_on_xy_grid(input_tensor, x, y):
    # input_tensor: (batch_size, height, width, nchannels)
    # x: (batch_size, height, width)
    # y: (batch_size, height, width)
    _, height, width, nchannels = input_tensor.shape
    batch_size = tf.shape(input_tensor)[0]
    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    batch_idx = tf.tile(batch_idx, (1, height, width))
    indices = tf.stack([batch_idx, y, x], axis=-1)  # (batch_size, height, width, 3)
    values = tf.gather_nd(input_tensor, indices)  # (batch_size, height, width, nchannels)
    return values


# https://github.com/kevinzakka/spatial-transformer-network/blob/375f99046383316b18edfb5c575dc390c4ee3193/stn/transformer.py#L66
def bilinear_interpolation(input_tensor, sampling_points):
    # input_tensor: (batch_size, height, width, nchannels)
    # sampling_points: (batch_size, height, width, 2)
    # sampling_points are the coordinates on which to interpolate input_tensor, in 'xy' format. They are absolute
    # coordinates (between 0 and width - 1 for the X axis, and between 0 and height - 1 for the Y axis.

    batch_size, height, width, nchannels = input_tensor.shape

    x = sampling_points[:, :, :, 0]  # (batch_size, height, width)
    y = sampling_points[:, :, :, 1]  # (batch_size, height, width)
    assert x.dtype == y.dtype == tf.float32

    # Get the 4 nearest input points for each sampling point:
    x0 = tf.cast(tf.floor(x), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), tf.int32)
    y1 = y0 + 1

    # Clip to input tensor boundaries:
    x0 = tf.clip_by_value(x0, 0, width - 1)
    x1 = tf.clip_by_value(x1, 0, width - 1)
    y0 = tf.clip_by_value(y0, 0, height - 1)
    y1 = tf.clip_by_value(y1, 0, height - 1)  # (batch_size, height, width)

    # Get values at input points:
    values_x0y0 = evaluate_tensor_on_xy_grid(input_tensor, x0, y0)
    values_x0y1 = evaluate_tensor_on_xy_grid(input_tensor, x0, y1)
    values_x1y0 = evaluate_tensor_on_xy_grid(input_tensor, x1, y0)
    values_x1y1 = evaluate_tensor_on_xy_grid(input_tensor, x1, y1)  # (batch_size, height, width, nchannels)

    # Cast pixel coordinates to float:
    x1 = tf.cast(x1, tf.float32)
    y1 = tf.cast(y1, tf.float32)

    # Compute interpolation weights:
    x1minusx = x1 - x
    y1minusy = y1 - y
    weight_x0y0 = tf.expand_dims(x1minusx * y1minusy, axis=-1)
    weight_x0y1 = tf.expand_dims(x1minusx * (1.0 - y1minusy), axis=-1)
    weight_x1y0 = tf.expand_dims((1.0 - x1minusx) * y1minusy, axis=-1)
    weight_x1y1 = tf.expand_dims((1.0 - x1minusx) * (1.0 - y1minusy), axis=-1)  # (batch_size, height, width, 1)

    # output shape: (batch_size, height, width, nchannels)
    return tf.math.accumulate_n([weight_x0y0 * values_x0y0, weight_x0y1 * values_x0y1,
                                 weight_x1y0 * values_x1y0, weight_x1y1 * values_x1y1])


# https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
def rotation_from_axisangle(axisangle):
    # axisangle: (batch_size, 3)

    theta = tf.norm(axisangle, axis=1, keepdims=True)  # (batch_size, 1)
    axis = axisangle / (theta + 1e-7)  # (batch_size, 3)

    theta = tf.squeeze(theta, axis=1)  # (batch_size)
    cos_theta = tf.cos(theta)  # (batch_size)
    sin_theta = tf.sin(theta)  # (batch_size)
    C = 1.0 - cos_theta  # (batch_size)

    x = axis[:, 0]  # (batch_size)
    y = axis[:, 1]  # (batch_size)
    z = axis[:, 2]  # (batch_size)

    xs = x * sin_theta
    ys = y * sin_theta
    zs = z * sin_theta
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    # Elements of the rotation matrix (each has dimension (batch_size)):
    R00 = x * xC + cos_theta
    R01 = xyC - zs
    R02 = zxC + ys
    R10 = xyC + zs
    R11 = y * yC + cos_theta
    R12 = yzC - xs
    R20 = zxC - ys
    R21 = yzC + xs
    R22 = z * zC + cos_theta

    # Stack all together:
    col0 = tf.stack([R00, R10, R20], axis=1)  # (batch_size, 3)
    col1 = tf.stack([R01, R11, R21], axis=1)  # (batch_size, 3)
    col2 = tf.stack([R02, R12, R22], axis=1)  # (batch_size, 3)
    R = tf.stack([col0, col1, col2], axis=2)  # (batch_size, 3, 3)

    return R


def make_transformation_matrix(raw_transformation, invert):
    # raw_transformation: (batch_size, 6)
    # The first 3 elements are the rotation, as an axis-angle. The last 3 are the translation.
    axisangle = raw_transformation[:, :3]
    translation = raw_transformation[:, 3:]
    batch_size = tf.shape(translation)[0]
    rotation = rotation_from_axisangle(axisangle)  # (batch_size, 3, 3)
    if invert:
        rotation = tf.transpose(rotation, perm=[0, 2, 1])  # (batch_size, 3, 3)
        translation = -tf.linalg.matvec(rotation, translation)  # (batch_size, 3)
    aux1 = tf.concat([rotation, tf.zeros((batch_size, 1, 3), tf.float32)], axis=1)  # (batch_size, 4, 3)
    aux2 = tf.concat([translation, tf.ones((batch_size, 1), tf.float32)], axis=1)  # (batch_size, 4)
    matrix_transformation = tf.concat([aux1, tf.expand_dims(aux2, axis=2)], axis=2)  # (batch_size, 4, 4)
    return matrix_transformation  # (batch_size, 4, 4)
