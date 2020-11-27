import tensorflow as tf
from scipy.spatial.transform import Rotation as R
import numpy as np

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


def test_transformation_from_parameters():
    print('test_transformation_from_parameters')
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
    T = transformations.transformation_from_parameters(axisangle, translation)

    assert T.shape == (2, 4, 4)
    assert tf.norm(T[0, :, :] - expectedT_0) < 1e-6
    assert tf.norm(T[1, :, :] - expectedT_1) < 1e-6


def test_transformation_from_parameters_inv():
    print('test_transformation_from_parameters_inv')
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
    T = transformations.transformation_from_parameters_inv(axisangle, translation)

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
    Kinv_tf = tf.constant(Kinv, tf.float32)  # (3, 3)
    points3d_hom_tf = transformations.backproject(depth_tf, Kinv_tf)  # (2, height, width, 4)

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
    K_tf = tf.constant(K, tf.float32)
    Tcw_tf = tf.constant(Tcw, tf.float32)
    pixel_coords_tf = transformations.project(points3d_hom_tf, K_tf, Tcw_tf)

    assert pixel_coords_tf.shape == (batch_size, height, width, 2)
    assert tf.norm(pixel_coords_tf - pixel_coords) < 1e-4







if __name__ == '__main__':
    test_rotation_from_axisangle()
    test_transformation_from_parameters()
    test_transformation_from_parameters_inv()
    test_backproject()
    test_project()
