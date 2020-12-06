import cv2
import numpy as np


# Display the first element of the batch.
def display_training(batch_imgs, net_output):
    # batch_imgs: (batch_size, height, width, 9)
    # net_output: (batch_size, height, width, 3 * 2 * 4 + 1)
    # The warped images and the depth prediction, in the following order:
    # (scale0_previous, scale0_next, scale1_previous, scale1_next, scale2_previous, scale2_next,
    # scale3_previous, scale3_next, depth)
    previous_orig = batch_imgs[0, :, :, :3]
    current_orig = batch_imgs[0, :, :, 3:6]
    next_orig = batch_imgs[0, :, :, 6:]
    previous_warped = net_output[0, :, :, :3].numpy()
    next_warped = net_output[0, :, :, 3:6].numpy()
    depth = net_output[0, :, :, -1].numpy()
    # print('depth range: %.4f %.4f %.4f' % (np.min(depth), np.mean(depth), np.max(depth)))
    cv2.imshow('previous image', previous_orig)
    cv2.imshow('current image', current_orig)
    cv2.imshow('next image', next_orig)
    cv2.imshow('previous warped', previous_warped)
    cv2.imshow('next warped', next_warped)
    cv2.imshow('depth', np.squeeze(depth) / 100.0)
    cv2.waitKey(1)
