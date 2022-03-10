import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm


"""
# Crop size of TB
height, width = [105, 1201 + 105], [0, 1599]

# Crop size of IPF
height, width = [0, 414], [0, None]
"""


def crop_image(folder, height, width):
    for file in tqdm(os.listdir(folder)[0]):
        img = cv2.imread(os.path.join(folder, file))
        cv2.imshow("crop", img[height[0]:height[1], width[0]:width[1]])
        cv2.imwrite(os.path.join(folder, file), img[height[0]:height[1], width[0]:width[1]])


def ebsd_post_processing(path, low_hsv, high_hsv, plot=False):
    """
    Extract the percentage area of red, green and blue areas in EBSD images
    :param path: Absolute path of the image
    :param low_hsv: Lower limit of HSV threshold for the corresponding color.
                    e.g. Red = np.array([160,60,60]), Green = np.array([36,25,25]), Blue = np.array([100,43,46])
    :param high_hsv: Upper limit of HSV threshold for the corresponding color.
                    e.g. Red = np.array([180,255,255]), Green = np.array([70,255,255]), Blue = np.array([140,255,255])
    :param plot: Whether to display images
    :return:
    """
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


if __name__ == '__main__':
    for label in ['X', 'Y', 'Z']:
        folder = os.path.join(r'F:\backup\paper\PureNi\Fig_Ni-pure\EBSD\IPF', label)
        savePath = os.path.join(r'F:\backup\paper\PureNi\Fig_Ni-pure\EBSD\IPF', f'{label}_mask')
        if not os.path.exists(savePath):
            os.mkdir(savePath)
        files = sorted(os.listdir(folder), key=lambda i: int(i.split('-')[0]))
        rgb_percent = []
        for file in files:
            tmp = []
            for low_hsv, high_hsv, info in zip(
                    [np.array([160, 60, 60]), np.array([36, 25, 25]), np.array([100, 43, 46])],
                    [np.array([180, 255, 255]), np.array([70, 255, 255]), np.array([140, 255, 255])],
                    ['Red', 'Green', 'Blue']):
                img = cv2.imread(os.path.join(folder, file))
                resized = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
                resized2 = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)

                fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5), num='0')
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
                plt.savefig(os.path.join(savePath, f'{file[:-5]}_{info}.png'), pad_inches=0)
                fig.clear()

                tmp.append(np.where(mask == 255)[0].shape[0] * 100 / (mask.shape[0] * mask.shape[1]))
            rgb_percent.append(tmp)
        rgb_percent = pd.DataFrame(np.array(rgb_percent))
        rgb_percent.columns = ['Red', 'Green', 'Blue']
        rgb_percent.to_csv(os.path.join(savePath, f'{label}_rgb.csv'), index=None)
