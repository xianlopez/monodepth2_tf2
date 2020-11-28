import tensorflow as tf


# TODO: Write unit tests for all these functions.


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


def transformation_from_parameters(axisangle, translation):
    # axisangle: (batch_size, 3)
    # translation: (batch_size, 3)
    batch_size = translation.shape[0]
    rotation = rotation_from_axisangle(axisangle)  # (batch_size, 3, 3)
    aux1 = tf.concat([rotation, tf.zeros((batch_size, 1, 3), tf.float32)], axis=1)  # (batch_size, 4, 3)
    aux2 = tf.concat([translation, tf.ones((batch_size, 1), tf.float32)], axis=1)  # (batch_size, 4)
    T = tf.concat([aux1, tf.expand_dims(aux2, axis=2)], axis=2)  # (batch_size, 4, 4)
    return T  # (batch_size, 4, 4)


def transformation_from_parameters_inv(axisangle, translation):
    # axisangle: (batch_size, 3)
    # translation: (batch_size, 3)
    batch_size = translation.shape[0]
    rotation = tf.transpose(rotation_from_axisangle(axisangle), perm=[0, 2, 1])  # (batch_size, 3, 3)
    translation = -tf.linalg.matvec(rotation, translation)  # (batch_size, 3)
    aux1 = tf.concat([rotation, tf.zeros((batch_size, 1, 3), tf.float32)], axis=1)  # (batch_size, 4, 3)
    aux2 = tf.concat([translation, tf.ones((batch_size, 1), tf.float32)], axis=1)  # (batch_size, 4)
    T = tf.concat([aux1, tf.expand_dims(aux2, axis=2)], axis=2)  # (batch_size, 4, 4)
    return T  # (batch_size, 4, 4)


def backproject(depth, Kinv):
    # depth: (batch_size, h, w, 1)
    # Kinv: (3, 3)
    # TODO: Maybe some things here can be pre-computed.
    batch_size, h, w, _ = depth.shape
    x = tf.cast(tf.range(w), tf.float32)
    y = tf.cast(tf.range(h), tf.float32)
    X, Y = tf.meshgrid(x, y)  # (h, w)
    pixel_coords_hom = tf.stack([X, Y, tf.ones_like(X)], axis=-1)  # (h, w, 3)
    rays = tf.linalg.matvec(Kinv, pixel_coords_hom)  # (h, w, 3)
    rays = tf.tile(tf.expand_dims(rays, axis=0), [batch_size, 1, 1, 1])  # (batch_size, h, w, 3)
    points3d = depth * rays  # (batch_size, h, w, 3)
    # Transform to homogeneous coordinates:
    points3d_hom = tf.concat([points3d, tf.ones((batch_size, h, w, 1), tf.float32)], axis=-1)
    return points3d_hom  # (batch_size, h, w, 4)


def project(points3d_hom, K, Tcw):
    # points3d_hom: (batch_size, h, w, 4)
    # K_ext: (3, 3) Intrinsics parameters matrix.
    # Tcw: (batch_size, 4, 4) Camera pose.
    K_ext = tf.concat([K, tf.zeros((3, 1), tf.float32)], axis=1)  # (3, 4)
    P = tf.matmul(K_ext, Tcw)  # (batch_size, 3, 4)
    assert P.shape == (Tcw.shape[0], 3, 4)
    # TODO: which of these options is faster?
    # cam_coords_hom = tf.linalg.matvec(Tcw, points3d_hom)  # (batch_size, h, w, 4)
    # pixel_coords_hom = tf.linalg.matvec(K_ext, cam_coords_hom)  # (batch_size, h, w, 3)
    P_exp = tf.expand_dims(tf.expand_dims(P, axis=1), axis=1)  # (batch_size, 1, 1, 3, 4)
    pixel_coords_hom = tf.linalg.matvec(P_exp, points3d_hom)  # (batch_size, h, w, 3)
    pixel_coords = pixel_coords_hom[:, :, :, :2] / (tf.expand_dims(pixel_coords_hom[:, :, :, 2], axis=-1) + 1e-7)
    return pixel_coords  # (batch_size, h, w, 2)


def evaluate_tensor_on_xy_grid(input_tensor, x, y):
    # input_tensor: (batch_size, height, width, nchannels)
    # x: (batch_size, height, width)
    # y: (batch_size, height, width)
    batch_size, height, width, nchannels = input_tensor.shape
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
    x0 = tf.cast(x0, tf.float32)
    x1 = tf.cast(x1, tf.float32)
    y0 = tf.cast(y0, tf.float32)
    y1 = tf.cast(y1, tf.float32)

    # Compute interpolation weights:
    x1minusx = x1 - x
    y1minusy = y1 - y
    weight_x0y0 = tf.expand_dims(x1minusx * y1minusy, axis=-1)
    weight_x0y1 = tf.expand_dims(x1minusx * (1.0 - y1minusy), axis=-1)
    weight_x1y0 = tf.expand_dims((1.0 - x1minusx) * y1minusy, axis=-1)
    weight_x1y1 = tf.expand_dims((1.0 - x1minusx) * (1.0 - y1minusy), axis=-1)  # (batch_size, height, width, 1)

    return tf.math.accumulate_n([weight_x0y0 * values_x0y0, weight_x0y1 * values_x0y1,
                                 weight_x1y0 * values_x1y0, weight_x1y1 * values_x1y1])  # (batch_size, height, width, nchannels)


def warp_images(imgs, depth, K, Kinv, axisangle, translation, invert):
    # imgs: (batch_size, height, width, 3)
    # depth: (batch_size, height, width, 1)
    # K: (3, 3)
    # Kinv: (3, 3)
    # axisangle: (batch_size, 3)
    # translation: (batch_size, 3)

    assert len(imgs.shape) == 4
    assert len(depth.shape) == 4
    batch_size, height, width, nchannels = imgs.shape
    assert nchannels == 3
    assert depth.shape == (batch_size, height, width, 1)
    assert axisangle.shape == (batch_size, 3)
    assert translation.shape == (batch_size, 3)

    # We consider the images come from a camera in the reference frame (world, w)
    # Tcw is therefore the transformation that will convert the images to the pose
    # defined by axisangle and translation.
    if invert:
        Tcw = transformation_from_parameters_inv(axisangle, translation)  # (batch_size, 4, 4)
    else:
        Tcw = transformation_from_parameters(axisangle, translation)  # (batch_size, 4, 4)

    # Get the 3D position of each pixel using the depths (in homogeneous coordinates):
    points3d_hom = backproject(depth, Kinv)  # (batch_size, h, w, 4)

    # Project them into the new cameras:
    pixel_coords = project(points3d_hom, K, Tcw)  # (batch_size, h, w, 2)

    return bilinear_interpolation(imgs, pixel_coords)  # (batch_size, height, width, nchannels)
