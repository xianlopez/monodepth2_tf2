import cv2
import numpy as np

def display_depth(disparities, batch_imgs):

    disp0, disp1, disp2, disp3 = disparities

    disp0 = disp0[-1, :, :, 0].numpy()
    disp0 -= np.min(disp0)
    disp0 *= 255.0 / np.max(disp0)

    disp1 = disp1[-1, :, :, 0].numpy()
    disp1 -= np.min(disp1)
    disp1 /= np.max(disp1)

    disp2 = disp2[-1, :, :, 0].numpy()
    disp2 -= np.min(disp2)
    disp2 /= np.max(disp2)

    disp3 = disp3[-1, :, :, 0].numpy()
    disp3 -= np.min(disp3)
    disp3 /= np.max(disp3)

    cv2.imshow("disp0", disp0)
    cv2.imshow("disp1", disp1)
    cv2.imshow("disp2", disp2)
    cv2.imshow("disp3", disp3)

    image1 = batch_imgs[-1, :, :, :3]
    image2 = batch_imgs[-1, :, :, 3:6]
    image3 = batch_imgs[-1, :, :, 6:]

    images_stacked = np.concatenate([image1, image2, image3], axis=0)

    cv2.imshow("images", images_stacked)

    cv2.waitKey(1)





