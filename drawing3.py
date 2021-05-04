import cv2
import numpy as np

# We display the first element of the batch.
def display_training(batch_imgs, disps):
    # batch_imgs: (batch_size, height, width, 9)
    # disps: list with disparities, in increasing resolution. Each element has shape
    # (batch_size, height / 2^s, width / 2^s, 1), being s the scale index.
    # TODO: Add images mean again, when I finally include its subtraction.
    previous_img = batch_imgs[0, :, :, :3]
    target_img = batch_imgs[0, :, :, 3:6]
    next_img = batch_imgs[0, :, :, 6:]
    cv2.imshow("previous", previous_img)
    cv2.imshow("target", target_img)
    cv2.imshow("next", next_img)
    for s in range(len(disps)):
        disparity = disps[s][0, :, :, :].numpy()
        print('disparity.shape: ' + str(disparity.shape))
        disparity /= np.max(disparity)
        cv2.imshow('disp' + str(s), disparity)
    cv2.waitKey(1)
