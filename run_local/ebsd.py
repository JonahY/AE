import cv2
import numpy as np
import matplotlib.pyplot as plt


def ebsd_post_processing(path, low_hsv, high_hsv, plot=False):
    '''
    Extract the percentage area of red, green and blue areas in EBSD images
    :param path: Absolute path of the image
    :param low_hsv: Lower limit of HSV threshold for the corresponding color.
                    e.g. Red = np.array([160,60,60]), Green = np.array([36,25,25]), Blue = np.array([100,43,46])
    :param high_hsv: Upper limit of HSV threshold for the corresponding color.
                    e.g. Red = np.array([180,255,255]), Green = np.array([70,255,255]), Blue = np.array([140,255,255])
    :param plot: Whether to display images
    :return:
    '''
    img = cv2.imread(path)
    resized = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
    resized2 = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)

    if plot:
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
        axes[0].imshow(mask, cmap=plt.cm.gray)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        axes[1].imshow(resized2)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

    return np.where(mask == 255)[0].shape[0], mask.shape[0] * mask.shape[1]
