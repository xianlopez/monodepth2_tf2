import cv2
import numpy as np
from data_reader import image_means

# We display the first element of the batch.
def display_training(batch_imgs, disps, image_from_before, image_from_after):
    # batch_imgs: (batch_size, height, width, 9)
    # disps: list with disparities, in increasing resolution. Each element has shape
    # (batch_size, height / 2^s, width / 2^s, 1), being s the scale index.
    # image_from_before: (batch_size, height, width, 3)
    # image_from_after: (batch_size, height, width, 3)

    previous_img = batch_imgs[0, :, :, :3] + image_means
    target_img = batch_imgs[0, :, :, 3:6] + image_means
    next_img = batch_imgs[0, :, :, 6:] + image_means
    cv2.imshow("previous", previous_img)
    cv2.imshow("target", target_img)
    cv2.imshow("next", next_img)

    for s in range(len(disps)):
        disparity = disps[s][0, :, :, :].numpy()
        disparity /= np.max(disparity)
        cv2.imshow('disp' + str(s), disparity)

    img_from_before = image_from_before[0, :, :, :].numpy() + image_means
    cv2.imshow('img_from_before', img_from_before)

    img_from_after = image_from_after[0, :, :, :].numpy() + image_means
    cv2.imshow('img_from_after', img_from_after)

    cv2.waitKey(1)
