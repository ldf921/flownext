import os

import matplotlib.pyplot as plt
import skimage.color
import numpy as np
import math


def flow2rgb(flow, norm='constant', value=20):
    '''
    Arguments:
        flow : the optical flow image [H, W, 2]
        norm : normalization method should be "constant" or "max"
        value : the value used for normalization 
    '''
    angle = (np.arctan2(-flow[:, :, 1], -flow[:, :, 0]) / np.pi + 1) / 2
    mag = np.linalg.norm(flow, axis=-1)
    if norm == 'max':
        mag = mag / np.max(mag)
    elif norm == 'constant':
        mag = np.minimum(1, mag / value)
    else:
        raise NotImplementedError
    ret = np.empty(mag.shape + (3,), dtype='float32')
    ret[:, :, 0] = angle
    ret[:, :, 1] = mag
    ret[:, :, 2] = 1
    return skimage.color.hsv2rgb(ret)    


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
