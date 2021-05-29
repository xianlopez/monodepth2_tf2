import tensorflow as tf
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2

import transformations


def test_rotation_from_axisangle():
    print('test_rotation_from_axisangle')
    # The first element in the batch will have the identity rotation, the second element another
    # more complicated rotation.
    axisangle0 = tf.constant([0.0, 0.0, 0.0], tf.float32)
    axisangle1_numpy = [0.3, -0.22, 1.0]
    axisangle1 = tf.constant(axisangle1_numpy)
    axisangles = tf.stack([axisangle0, axisangle1], axis=0)
    rot_matrices = transformations.rotation_from_axisangle(axisangles)
    assert rot_matrices.shape == (2, 3, 3)
    assert tf.norm(rot_matrices[0, :, :] - tf.eye(3)) < 1e-6
    expected_matrix_1 = tf.constant(R.from_rotvec(axisangle1_numpy).as_matrix(), tf.float32)
    assert tf.norm(rot_matrices[1, :, :] - expected_matrix_1) < 1e-6


def test_make_transformation_matrix():
    print('test_make_transformation_matrix')
    axisangle_numpy = np.array([[0.0, 0.0, 0.0], [0.3, -0.22, 1.0]])
    translation_numpy = np.array([[0.0, 0.0, 0.0], [-1.0, 2.56, 0.7]])

    rotations = R.from_rotvec(axisangle_numpy).as_matrix()
    expectedT_0 = np.eye(4, dtype=np.float32)
    expectedT_0[:3, :3] = rotations[0]
    expectedT_0[:3, 3] = translation_numpy[0, :]
    expectedT_1 = np.eye(4, dtype=np.float32)
    expectedT_1[:3, :3] = rotations[1]
    expectedT_1[:3, 3] = translation_numpy[1, :]

    axisangle = tf.constant(axisangle_numpy, tf.float32)
    translation = tf.constant(translation_numpy, tf.float32)
    T = transformations.make_transformation_matrix(tf.concat([axisangle, translation], axis=1), False)

    assert T.shape == (2, 4, 4)
    assert tf.norm(T[0, :, :] - expectedT_0) < 1e-6
    assert tf.norm(T[1, :, :] - expectedT_1) < 1e-6


def test_make_transformation_matrix_inv():
    print('test_make_transformation_matrix_inv')
    axisangle_numpy = np.array([[0.0, 0.0, 0.0], [0.3, -0.22, 1.0]])
    translation_numpy = np.array([[0.0, 0.0, 0.0], [-1.0, 2.56, 0.7]])

    rotations = R.from_rotvec(axisangle_numpy).as_matrix()
    rotation0 = np.transpose(rotations[0, :, :])
    rotation1 = np.transpose(rotations[1, :, :])

    expectedT_0 = np.eye(4, dtype=np.float32)
    expectedT_0[:3, :3] = rotation0
    expectedT_0[:3, 3] = -rotation0.dot(translation_numpy[0, :])
    expectedT_1 = np.eye(4, dtype=np.float32)
    expectedT_1[:3, :3] = rotation1
    expectedT_1[:3, 3] = -rotation1.dot(translation_numpy[1, :])

    axisangle = tf.constant(axisangle_numpy, tf.float32)
    translation = tf.constant(translation_numpy, tf.float32)
    T = transformations.make_transformation_matrix(tf.concat([axisangle, translation], axis=1), True)

    assert T.shape == (2, 4, 4)
    assert tf.norm(T[0, :, :] - expectedT_0) < 1e-6
    assert tf.norm(T[1, :, :] - expectedT_1) < 1e-6


def test_backproject():
    print('test_backproject')

    width = 20
    height = 30
    fx = 70
    fy = 60
    cx = 10
    cy = 12
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    Kinv = np.linalg.inv(K)

    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)
    pixel_coords_hom = np.stack([xx, yy, np.ones((height, width), np.float32)], axis=-1)  # (height, width, 3)
    rays = np.zeros((height, width, 3), np.float32)
    for i in range(height):
        for j in range(width):
            rays[i, j, :] = Kinv.dot(pixel_coords_hom[i, j, :])

    depth_0 = np.ones((height, width, 1), np.float32)
    points3d_expected_0 = depth_0 * rays  # (height, width, 3)

    depth_1 = np.random.rand(height, width, 1) + 1.0  # depth between 1 and 2.
    points3d_expected_1 = depth_1 * rays  # (height, width, 3)

    depth_tf = tf.constant(np.stack([depth_0, depth_1], axis=0), tf.float32)  # (2, height, width, 1)

    backproject_layer = transformations.BackprojectLayer(K, height, width, 2)
    points3d_hom_tf = backproject_layer(depth_tf)

    assert points3d_hom_tf.shape == (2, height, width, 4)
    points3d_tf = points3d_hom_tf[:, :, :, :3] / tf.expand_dims(points3d_hom_tf[:, :, :, 3], axis=-1)
    assert tf.norm(points3d_tf[0, ...] - points3d_expected_0) < 1e-6
    assert tf.norm(points3d_tf[1, ...] - points3d_expected_1) < 1e-6

    # Now we'll check as well one point, computing the expected value in a different way:
    px = 12
    py = 3
    point3d_p = depth_1[py, px, 0] * np.array([(px - cx) / fx, (py - cy) / fy, 1.0], np.float32)
    assert tf.norm(points3d_tf[1, py, px, :] - point3d_p) < 1e-6


def test_project():
    print('test_project')

    batch_size = 2
    width = 20
    height = 30
    fx = 35
    fy = 60
    cx = 10
    cy = 12
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy

    # Define 3D points with x in (-1, 1), y in (-1, 1) and z in (3, 5)
    x = np.random.rand(batch_size, height, width) * 2.0 - 1.0
    y = np.random.rand(batch_size, height, width) * 2.0 - 1.0
    z = np.random.rand(batch_size, height, width) * 2.0 + 3.0
    points3d_hom = np.stack([x, y, z, np.ones((batch_size, height, width), np.float32)], axis=-1)  # (batch_size, height, width, 4)

    # Camera 0 pose:
    position_0 = [0.0, 0.0, 0.0]
    rotation_0 = R.from_euler('ZYX', [0, 0, 0], degrees=True)
    Twc_0 = np.eye(4, dtype=np.float32)
    Twc_0[:3, :3] = rotation_0.as_matrix()
    Twc_0[:3, 3] = position_0
    Tcw_0 = np.linalg.inv(Twc_0)

    # Camera 1 pose:
    position_1 = [3.0, -0.2, 1.0]
    rotation_1 = R.from_euler('ZYX', [3, -45, -5], degrees=True)
    Twc_1 = np.eye(4, dtype=np.float32)
    Twc_1[:3, :3] = rotation_1.as_matrix()
    Twc_1[:3, 3] = position_1
    Tcw_1 = np.linalg.inv(Twc_1)

    Tcw = np.stack([Tcw_0, Tcw_1], axis=0)  # (2, 4, 4)

    pixel_coords = np.zeros((batch_size, height, width, 2), np.float32)
    for b in range(batch_size):
        for i in range(height):
            for j in range(width):
                point_3d_hom_cam_coords = np.matmul(Tcw[b, :, :], points3d_hom[b, i, j, :])
                point_3d_cam_coords = point_3d_hom_cam_coords[:3] / point_3d_hom_cam_coords[3]
                pixel_coords_hom = np.matmul(K, point_3d_cam_coords)
                pixel_coords[b, i, j] = pixel_coords_hom[:2] / pixel_coords_hom[2]

    points3d_hom_tf = tf.constant(points3d_hom, tf.float32)
    project_layer = transformations.ProjectLayer(K, height, width, batch_size)
    Tcw_tf = tf.constant(Tcw, tf.float32)

    points3d_hom_cam = transformations.transform3d(Tcw_tf, points3d_hom_tf)
    pixel_coords_tf = project_layer(points3d_hom_cam)

    assert pixel_coords_tf.shape == (batch_size, height, width, 2)
    assert tf.norm(pixel_coords_tf - pixel_coords) < 1e-4


def test_evaluate_tensor_on_xy_grid():
    print('test_evaluate_tensor_on_xy_grid')
    batch_size = 2
    height = 3
    width = 4
    nchannels = 3
    input_values = np.random.rand(batch_size, height, width, nchannels)
    x = np.zeros(shape=(batch_size, height, width), dtype=np.int32)
    y = np.zeros(shape=(batch_size, height, width), dtype=np.int32)
    # Element (0, 0, 0) must be input_values[0, 0, 1, :]:
    x[0, 0, 0] = 1
    # Element (0, 1, 0) must be input_values[0, 0, 2, :]:
    x[0, 1, 0] = 2
    y[0, 1, 0] = 0
    # Element (1, 2, 3) must be input_values[1, 1, 3, :]:
    x[1, 2, 3] = 3
    y[1, 2, 3] = 1
    # Element (1, 1, 2) must be input_values[1, 1, 2, :]:
    x[1, 1, 2] = 2
    y[1, 1, 2] = 1
    # Element (1, 0, 3) must be input_values[1, 1, 2, :]:
    x[1, 0, 3] = 2
    y[1, 0, 3] = 1
    # The rest of the elements should be (0, 0) on its corresponding batch.

    input_tensor = tf.constant(input_values, tf.float32)
    x_tf = tf.constant(x, tf.int32)
    y_tf = tf.constant(y, tf.int32)
    output_tensor = transformations.evaluate_tensor_on_xy_grid(input_tensor, x_tf, y_tf)

    assert output_tensor.shape == (batch_size, height, width, nchannels)
    for b in range(batch_size):
        for i in range(height):
            for j in range(width):
                if b == 0 and i == 0 and j == 0:
                    assert tf.norm(output_tensor[b, i, j, :] - input_values[0, 0, 1, :]) < 1e-6
                elif b == 0 and i == 1 and j == 0:
                    assert tf.norm(output_tensor[b, i, j, :] - input_values[0, 0, 2, :]) < 1e-6
                elif b == 1 and i == 2 and j == 3:
                    assert tf.norm(output_tensor[b, i, j, :] - input_values[1, 1, 3, :]) < 1e-6
                elif b == 1 and i == 1 and j == 2:
                    assert tf.norm(output_tensor[b, i, j, :] - input_values[1, 1, 2, :]) < 1e-6
                elif b == 1 and i == 0 and j == 3:
                    assert tf.norm(output_tensor[b, i, j, :] - input_values[1, 1, 2, :]) < 1e-6
                else:
                    assert tf.norm(output_tensor[b, i, j, :] - input_values[b, 0, 0, :]) < 1e-6


def interpolate(A, x, y):
    # A: (height, width)
    # A must be indexed as A(y, x)
    x1 = int(np.floor(x))
    x2 = x1 + 1
    y1 = int(np.floor(y))
    y2 = y1 + 1
    output = (y2 - y) * ((x2 - x) * A[y1, x1] + (x - x1) * A[y1, x2]) + \
             (y - y1) * ((x2 - x) * A[y2, x1] + (x - x1) * A[y2, x2])
    return output


def test_bilinear_interpolation():
    print('test_bilinear_interpolation')
    batch_size = 2
    height = 5
    width = 4
    nchannels = 3
    input_tensor = np.random.rand(batch_size, height, width, nchannels)
    sampling_points = np.zeros((batch_size, height, width, 2), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            sampling_points[:, i, j, 0] = j
            sampling_points[:, i, j, 1] = i
    # Two points outside (to test the border method):
    sampling_points[0, 1, 1, 0] = 7
    sampling_points[0, 3, 3, 1] = -0.5
    # One point just changed to another exact position (repetition):
    sampling_points[1, 2, 3, :] = sampling_points[1, 2, 2, :]
    # A few points not falling on original locations (to test interpolation):
    sampling_points[0, 0, 0, :] = [0.1, 0.2]
    sampling_points[1, 4, 2, :] = [2.7, 3.5]
    sampling_points[1, 3, 1, :] = [2.2, 0]
    sampling_points[0, 0, 1, :] = [2.9, 3.6]

    # Expected values for the ones that need interpolation:
    expected000 = [interpolate(input_tensor[0, :, :, 0], 0.1, 0.2),
                   interpolate(input_tensor[0, :, :, 1], 0.1, 0.2),
                   interpolate(input_tensor[0, :, :, 2], 0.1, 0.2)]
    expected142 = [interpolate(input_tensor[1, :, :, 0], 2.7, 3.5),
                   interpolate(input_tensor[1, :, :, 1], 2.7, 3.5),
                   interpolate(input_tensor[1, :, :, 2], 2.7, 3.5)]
    expected131 = [interpolate(input_tensor[1, :, :, 0], 2.2, 0),
                   interpolate(input_tensor[1, :, :, 1], 2.2, 0),
                   interpolate(input_tensor[1, :, :, 2], 2.2, 0)]
    expected001 = [interpolate(input_tensor[0, :, :, 0], 2.9, 3.6),
                   interpolate(input_tensor[0, :, :, 1], 2.9, 3.6),
                   interpolate(input_tensor[0, :, :, 2], 2.9, 3.6)]

    input_tensor_tf = tf.constant(input_tensor, tf.float32)
    sampling_points_tf = tf.constant(sampling_points, tf.float32)
    output_tensor_tf = transformations.bilinear_interpolation(input_tensor_tf, sampling_points_tf)

    assert output_tensor_tf.shape == (batch_size, height, width, nchannels)
    assert tf.norm(output_tensor_tf[1, 0, 0, :] - input_tensor[1, 0, 0, :]) < 1e-6
    assert tf.norm(output_tensor_tf[0, 1, 1, :] - input_tensor[0, 1, -1, :]) < 1e-6
    assert tf.norm(output_tensor_tf[0, 3, 3, :] - input_tensor[0, 0, 3, :]) < 1e-6
    assert tf.norm(output_tensor_tf[1, 2, 3, :] - input_tensor[1, 2, 2, :]) < 1e-6
    assert tf.norm(output_tensor_tf[0, 0, 0, :] - expected000) < 1e-6
    assert tf.norm(output_tensor_tf[1, 4, 2, :] - expected142) < 1e-6
    assert tf.norm(output_tensor_tf[1, 3, 1, :] - expected131) < 1e-6
    assert tf.norm(output_tensor_tf[0, 0, 1, :] - expected001) < 1e-6


def test_warp_images():
    print('test_warp_images')
    test_img = cv2.imread('test_image_1.jpg')  # Black border
    # test_img = cv2.imread('test_image_2.jpg')  # Non black border
    test_img = test_img.astype(np.float32) / 255.0
    cv2.imshow('original image', test_img)

    height, width, _ = test_img.shape
    batch_size = 7

    depth = np.ones((height, width, 1), np.float32) * 5.0
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = 50  # fx
    K[1, 1] = 50  # fy
    K[0, 2] = width / 2.0  # cx
    K[1, 2] = height / 2.0  # cy

    # Camera 0 (identity):
    translation0 = [0.0, 0.0, 0.0]
    axisangle0 = [0.0, 0.0, 0.0]
    # Camera 1 (moving forward):
    translation1 = [0.0, 0.0, 1.0]
    axisangle1 = [0.0, 0.0, 0.0]
    # Camera 2 (moving backwards):
    translation2 = [0.0, 0.0, -1.0]
    axisangle2 = [0.0, 0.0, 0.0]
    # Camera 3 (lateral motion):
    translation3 = [0.5, 0.0, 0.0]
    axisangle3 = [0.0, 0.0, 0.0]
    # Camera 4 (rotation -30 degrees):
    translation4 = [0.0, 0.0, 0.0]
    axisangle4 = [0.0, 0.0, -30.0 / 180.0 * np.pi]
    # Camera 5 (rotation 90 degrees):
    translation5 = [0.0, 0.0, 0.0]
    axisangle5 = [0.0, 0.0, np.pi / 2.0]
    # Camera 6 (tilted and moved downwards):
    translation6 = [0.0, 2.0, 0.0]
    axisangle6 = [21.0814 / 180.0 * np.pi, 0.0, 0.0]

    test_img_tf = tf.constant(np.tile(np.expand_dims(test_img, axis=0), [batch_size, 1, 1, 1]), dtype=tf.float32)
    depth_tf = tf.constant(np.tile(np.expand_dims(depth, axis=0), [batch_size, 1, 1, 1]))
    axisangle_tf = tf.constant(np.stack([axisangle0, axisangle1, axisangle2, axisangle3,
                                         axisangle4, axisangle5, axisangle6], axis=0), dtype=tf.float32)
    translation_tf = tf.constant(np.stack([translation0, translation1, translation2, translation3,
                                           translation4, translation5, translation6], axis=0), dtype=tf.float32)

    backproject_layer = transformations.BackprojectLayer(K, height, width, batch_size)
    points3d_hom_target = backproject_layer(depth_tf)

    raw_transformations = tf.concat([axisangle_tf, translation_tf], axis=1)
    T = transformations.make_transformation_matrix(raw_transformations, False)

    warp_layer = transformations.WarpLayer(K, height, width, batch_size)
    new_images = warp_layer(test_img_tf, points3d_hom_target, T)

    assert new_images.shape == (batch_size, height, width, 3)

    cv2.imshow('identity', new_images[0, :, :, :].numpy())
    cv2.imshow('moving forward', new_images[1, :, :, :].numpy())
    cv2.imshow('moving backwards', new_images[2, :, :, :].numpy())
    cv2.imshow('lateral motion', new_images[3, :, :, :].numpy())
    cv2.imshow('rotation -30 degrees', new_images[4, :, :, :].numpy())
    cv2.imshow('rotation 90 degrees', new_images[5, :, :, :].numpy())
    cv2.imshow('tilted and moved downwards', new_images[6, :, :, :].numpy())

    cv2.waitKey()


if __name__ == '__main__':
    test_rotation_from_axisangle()
    test_make_transformation_matrix()
    test_make_transformation_matrix_inv()
    test_backproject()
    test_project()
    test_evaluate_tensor_on_xy_grid()
    test_bilinear_interpolation()
    test_warp_images()
