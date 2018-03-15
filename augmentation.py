import skimage.transform
import numpy as np
import math


class GeometryAugmentation:
    def __init__(self, angle_range, zoom_range, translation_range, contrast_range, brightness_sigma, channel_range, target_shape):
        self._angle_range = tuple(map(lambda x : x / 180 * math.pi, angle_range) )
        self._scale_range = zoom_range
        try:
            translation_range = tuple(translation_range)
            if len(translation_range) != 2:
                raise ValueError("expect translation range to have shape [2,], but got {}".format(translation_range))
        except TypeError:
            translation_range = (-translation_range, translation_range)
        self._translation_range = translation_range
        self._target_shape = target_shape
        self._contrast_range = contrast_range
        self._brightness_sigma = brightness_sigma
        self._channel_range = channel_range

    def random_params(self):
        ''' Generate a dict of augmentation parameters
        scale : (sx, sy) zoom factor x' = a x, a < 1, zoom in
        translation : (tx, ty) translation relative to image size
        '''
        return dict(
            rotation = np.random.uniform(*self._angle_range),
            scale = np.random.uniform(*self._scale_range, size=(2,)),
            translation = np.random.uniform(*self._translation_range, size=(2,)),
            contrast = np.random.uniform(*self._contrast_range) + 1,
            brightness = np.random.normal(scale=self._brightness_sigma),
            channel = np.random.uniform(*self._channel_range, size=(3,))
        )

    @staticmethod
    def adjust_value(img, contrast, brightness, channel):
        mean = np.mean(img, keepdims=True, axis=(0,1))
        channel = np.reshape(channel, (1, 1, 3))
        return np.clip((img - mean) * (contrast * channel) + (mean + brightness) * channel, 0, 1)

    def apply(self, params, data):
        '''
            params : dictionary of augmentation parameters
        '''
        img1, img2, flow = data
        rows, cols = img1.shape[:2]
        center = cols / 2 - 0.5, rows / 2 - 0.5
        transform = skimage.transform.AffineTransform(rotation=params['rotation'], scale=params['scale'])
        translation = np.dot(np.identity(2) - transform.params[:2, :2], center) + params['translation'] / params['scale'] * [cols, rows]
        transform.params[:2, 2] = translation
        concat_img = np.concatenate([img1 / 255.0, img2 / 255.0, flow], axis=-1)

        # offset = [ np.random.randint(0, s - ts + 1) for s, ts in zip(concat_img.shape, self._target_shape) ]
        offset = [ (s - ts) // 2 for s, ts in zip(concat_img.shape, self._target_shape) ]
        crop_range = [slice(o, o + s) for o, s in zip(offset, self._target_shape)]
        concat_img = skimage.transform.warp(concat_img.astype(np.float64), transform)[crop_range]

        flow = concat_img[..., 6:8]
        rinv = np.linalg.inv(transform.params[:2, :2]).transpose()
        flow[:] = np.reshape(np.matmul(np.reshape(flow, [-1, 2]),rinv),flow.shape)

        img1, img2 = concat_img[..., :3], concat_img[..., 3:6]
        img1 = self.adjust_value(img1, params['contrast'], params['brightness'], params['channel'])
        img2 = self.adjust_value(img2, params['contrast'], params['brightness'], params['channel'])
        return tuple(map(lambda x : np.transpose(x, (2, 0, 1)), [img1, img2, flow]))