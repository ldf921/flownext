import skimage.transform
import numpy as np
import math


class GeometryAugmentation:
    def __init__(self, angle_range, zoom_range, target_shape):
        self._angle_range = tuple(map(lambda x : x / 180 * math.pi, angle_range) )
        self._scale_range = zoom_range
        self._target_shape = target_shape

    def random_params(self):
        return dict(
            rotation = np.random.uniform(*self._angle_range),
            scale = np.random.uniform(self._scale_range)
        )

    def apply(self, params, data):
        '''
            params : dict(rotation=, scale=)
        '''
        img1, img2, flow = data
        rows, cols = img1.shape[:2]
        center = cols / 2 - 0.5, rows / 2 - 0.5
        transform = skimage.transform.AffineTransform(rotation=params['rotation'], scale=params['scale'])
        translation = np.dot(np.identity(2) - transform.params[:2, :2], center)
        transform.params[:2, 2] = translation
        concat_img = np.concatenate([img1 / 255.0, img2 / 255.0, flow], axis=-1)

        offset = [ np.random.randint(0, s - ts + 1) for s, ts in zip(concat_img.shape, self._target_shape) ]
        crop_range = [slice(o, o + s) for o, s in zip(offset, self._target_shape)]
        concat_img = skimage.transform.warp(concat_img.astype(np.float64), transform)[crop_range]

        flow = concat_img[..., 6:8]
        rinv = np.linalg.inv(transform.params[:2, :2]).transpose()
        flow[:] = np.reshape(np.matmul(np.reshape(flow, [-1, 2]),rinv),flow.shape)
        return tuple(map(lambda x : np.transpose(x, (2, 0, 1)), [concat_img[..., :3], concat_img[..., 3:6], flow]))