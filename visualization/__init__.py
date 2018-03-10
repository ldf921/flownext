import os

import matplotlib.pyplot as plt
import skimage.color
import numpy as np
import math


def convert(img, max_motion=20):
    # ret = np.empty([img.shape[0], img.shape[1], 3], dtype=np.float64)
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         x, y = -img[i, j, 0], -img[i, j, 1]
    #         degree = (math.atan2(y, x) / math.pi + 1) / 2
    #         mag = min(1, ((x ** 2 + y ** 2) ** 0.5) / max_motion)
    #         ret[i, j] = [degree, mag, 1]
    deg = (np.arctan2(-img[:, :, 1], -img[:, :, 0]) / math.pi + 1) / 2
    mag = np.minimum(1, np.linalg.norm(img, axis=-1) / max_motion)
    ret = np.stack([deg, mag, np.ones_like(deg)], axis=2)
    return (skimage.color.hsv2rgb(ret) * 255.0).astype(np.uint8)


fig, axes = plt.subplots(2, 2, figsize=(16, 12))


def plot(img1, img2, gt, pred, path, c=0):
    try:
        os.mkdir(path)
    except OSError:
        pass
    for i in range(0, len(img1)):
        axes[0, 0].imshow(img1[i])
        axes[1, 0].imshow(img2[i])
        axes[0, 1].imshow(convert(gt[i]))
        axes[1, 1].imshow(convert(pred[i] * 20))
        fig.savefig(os.path.join(path, '{}.png'.format(i + c)))
