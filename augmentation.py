import skimage.transform
import numpy as np
import math
from mxnet import nd
from mxnet.gluon import nn

class ColorAugmentation(nn.HybridBlock):
    def __init__(self, contrast_range, brightness_sigma, channel_range, batch_size, shape, noise_range, gamma_range = None, **kwargs):
        super().__init__(**kwargs)
        self._contrast_range = contrast_range
        self._brightness_sigma = brightness_sigma
        self._channel_range = channel_range
        self._batch_size = batch_size
        self._shape = shape
        self._noise_range = noise_range
        self._gamma_range = gamma_range

    def hybrid_forward(self, F, img1, img2):
        contrast = F.random.uniform(*self._contrast_range, shape=(self._batch_size, 1, 1, 1)) + 1
        brightness = F.random.normal(scale=self._brightness_sigma, shape=(self._batch_size, 1, 1, 1))
        channel = F.random.uniform(*self._channel_range, shape=(self._batch_size, 3, 1, 1))
        noise_sigma = F.random.uniform(*self._noise_range)
        if self._gamma_range is not None:
            gamma = F.random.uniform(*self._gamma_range)

        contrast = contrast.repeat(repeats=3, axis=1)
        brightness = brightness.repeat(repeats=3, axis=1)
        ret = []
        for img in (img1, img2):
            aug = img
            noise = F.random.normal(scale=1, shape=(self._batch_size, 3) + self._shape)
            aug = aug + noise * noise_sigma
            mean = F.mean(aug, keepdims=True, axis=(2,3))
            aug = F.broadcast_add(F.broadcast_mul(F.broadcast_minus(img, mean), contrast * channel), (mean + brightness) * channel)
            aug = F.clip(aug, 0, 1)
            if self._gamma_range is not None:
                aug = aug ** gamma
            ret.append(aug)

        return ret

class GeometryAugmentation(nn.HybridBlock):
    def __init__(self, angle_range, zoom_range, translation_range, target_shape, orig_shape, batch_size, aspect_range = None, relative_angle=None, relative_scale=None):
        super().__init__()
        self._angle_range = tuple(map(lambda x : x / 180 * math.pi, angle_range) )
        self._scale_range = zoom_range
        try:
            translation_range = tuple(translation_range)
            if len(translation_range) != 2:
                raise ValueError("expect translation range to have shape [2,], but got {}".format(translation_range))
        except TypeError:
            translation_range = (-translation_range, translation_range)
        self._translation_range = tuple(map(lambda x : x * 2, translation_range))
        self._target_shape = np.array(target_shape)
        self._orig_shape = np.array(orig_shape)
        self._batch_size = batch_size
        self._unit = np.flip(self._target_shape - 1, axis=0).reshape([2,1]) / np.flip(self._orig_shape - 1, axis=0).reshape([1,2])
        self._relative = relative_angle is not None 
        if self._relative:
            self._relative_scale = relative_scale
            self._relative_angle = tuple(map(lambda x : x / 180 * math.pi * relative_angle, angle_range) )
        self._aspect_range = aspect_range

    def _get_relative_transform(self, F):
        aspect_ratio = (self._target_shape[0] - 1) / (self._target_shape[1] - 1)
        rotation = F.random.uniform(*self._relative_angle, shape=(self._batch_size))
        scale = F.random.uniform(*self._relative_scale, shape=(self._batch_size))
        # print(rotation.asscalar(), scale.asscalar())
        affine_params = [scale * rotation.cos(), scale * -rotation.sin() * aspect_ratio, F.zeros_like(scale),
                         scale * rotation.sin() / aspect_ratio, scale * rotation.cos(),  F.zeros_like(scale),
                         F.zeros_like(scale), F.zeros_like(scale), F.ones_like(scale)]
        affine = F.reshape(F.stack(*affine_params, axis=1), [0, 3, 3])
        inverse = F.stack(
            rotation.cos() / scale, 
            rotation.sin() / scale,
            -rotation.sin() / scale, 
            rotation.cos() / scale,
            axis=1
        )
        inverse = F.reshape(inverse, [0, 2, 2])
        return affine, inverse
        
    def hybrid_forward(self, F, img1, img2, flow):
        '''
            params : dictionary of augmentation parameters
        '''
        rotation = F.random.uniform(*self._angle_range, shape=(self._batch_size))
        scale = F.random.uniform(*self._scale_range, shape=(self._batch_size))
        if self._aspect_range is not None:
            aspect_ratio = F.random.uniform(*self._aspect_range, shape=(self._batch_size))
        else:
            aspect_ratio = 1
        pad_x, pad_y = 1 - scale * self._unit[0, 0], 1 - scale * self._unit[1, 1]
        translation_x = F.random.uniform(-1, 1, shape=(self._batch_size,)) * pad_x + F.random.uniform(*self._translation_range, shape=(self._batch_size))
        translation_y = F.random.uniform(-1, 1, shape=(self._batch_size,)) * pad_y + F.random.uniform(*self._translation_range, shape=(self._batch_size))
        affine_params = [scale * aspect_ratio * rotation.cos() * self._unit[0, 0], scale * aspect_ratio * -rotation.sin() * self._unit[1, 0], translation_x,
                         scale * rotation.sin() * self._unit[0, 1], scale * rotation.cos() * self._unit[1, 1],  translation_y] 
        affine_params = F.stack(*affine_params, axis=1)
        affine_inverse = F.stack(
                rotation.cos() / (scale * aspect_ratio), 
                rotation.sin() / (scale * aspect_ratio),
                -rotation.sin() / scale, 
                rotation.cos() / scale,
                axis=1
            )
        linv = F.reshape(affine_inverse, [0, 2, 2])

        if not self._relative:
            concat_img = F.concat(img1 / 255.0, img2 / 255.0, flow, dim=1)
            grid = F.GridGenerator(data=affine_params, transform_type='affine', target_shape=list(self._target_shape))
            concat_img = F.BilinearSampler(data=concat_img, grid=grid)

            flow = F.slice_axis(concat_img, axis=1, begin=6, end=8)    
            flow = F.reshape_like(F.batch_dot(linv, F.reshape(flow, (0, 0, -3))), flow)
            img1, img2 = F.slice_axis(concat_img, axis=1, begin=0, end=3), F.slice_axis(concat_img, axis=1, begin=3, end=6)
            return img1, img2, flow
        else:
            # affine_2 = affine * rel_affine  (left matrix multiply)
            # affine^{-1}_2 = rel_inverse * linv
            # x2 = (O - 1) *  [W_2 (x'_2 / ((T - 1) / 2) - (1, 1) ) + b_2 + (1, 1)]
            # x1 = W_1 x'_1 + b_1 
            # x'_2−x'_1  = (W_2^(−1)  W_1−I)  x'_1  + b_1  − b_2  + W_2^(−1)  f(x_1)
            # (W_2^(−1)  W_1−I) * (x'_1 - (Ts - 1) / 2)
            rel_affine, rel_inverse = self._get_relative_transform(F)
            affine_2 = F.reshape(F.batch_dot(F.reshape(affine_params, [0, 2, 3]), rel_affine), [0, 6])

            concat_img = F.concat(img1 / 255.0, flow, dim=1)
            grid = F.GridGenerator(data=affine_params, transform_type='affine', target_shape=list(self._target_shape))
            concat_img = F.BilinearSampler(data=concat_img, grid=grid)
            img1 = F.slice_axis(concat_img, axis=1, begin=0, end=3)    
            flow = F.slice_axis(concat_img, axis=1, begin=3, end=5)

            grid_2 = F.GridGenerator(data=affine_2, transform_type='affine', target_shape=list(self._target_shape))
            img2 = F.BilinearSampler(data=img2 / 255.0, grid=grid_2)

            inverse_2 = F.batch_dot(rel_inverse, linv)
            flow = F.reshape_like(F.batch_dot(inverse_2, F.reshape(flow, (0, 0, -3))), flow)

            scale = F.stack(F.ones([self._batch_size]) * (self._target_shape[1] - 1) / 2,
                            F.zeros([self._batch_size]),
                            F.zeros([self._batch_size]),
                            F.ones([self._batch_size]) * (self._target_shape[0] - 1) / 2,
                            axis=1) 
            scale = F.reshape(scale, [0, 2, 2])
            I = F.reshape(F.one_hot(F.arange(0, 2), depth=2), [1, 2, 2])
            grid = F.GridGenerator(data=F.reshape(F.one_hot(F.arange(0, 2), depth=3), [1, 6]),
                                   transform_type='affine',
                                   target_shape=list(self._target_shape))
            grid = F.reshape(F.repeat(grid, axis=0, repeats=self._batch_size), [0, 0, -3])
            factor = F.batch_dot(F.broadcast_minus(rel_inverse, I), scale)
            flow = flow + F.reshape_like(F.batch_dot(factor, grid), flow)
            return img1, img2, flow
            
class GeometryAugmentationCpu:
    def __init__(self, angle_range, zoom_range, translation_range, target_shape):
        self._angle_range = tuple(map(lambda x : x / 180 * math.pi, angle_range))
        self._scale_range = zoom_range
        try:
            translation_range = tuple(translation_range)
            if len(translation_range) != 2:
                raise ValueError("expect translation range to have shape [2,], but got {}".format(translation_range))
        except TypeError:
            translation_range = (-translation_range, translation_range)
        self._translation_range = translation_range
        self._target_shape = target_shape
        
    def random_params(self):
        ''' Generate a dict of augmentation parameters
        scale : (sx, sy) zoom factor x' = a x, a < 1, zoom in
        translation : (tx, ty) translation relative to image size
        '''
        return dict(
            rotation = np.random.uniform(*self._angle_range),
            scale = np.random.uniform(*self._scale_range, size=(2,)),
            translation = np.random.uniform(*self._translation_range, size=(2,))
        )

    def apply(self, params, data):
        '''
            params : dictionary of augmentation parameters
        '''
        # pylint : disable=E1101

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
        flow = np.reshape(np.matmul(np.reshape(flow, [-1, 2]),rinv),flow.shape)

        img1, img2 = concat_img[..., :3], concat_img[..., 3:6]
        return tuple(map(lambda x : np.transpose(x, (2, 0, 1)), [img1, img2, flow]))