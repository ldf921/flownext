import tensorflow as tf
from network import Network, convolve, convolveLeakyReLU, upconvolve, upconvolveLeakyReLU, MultiGPUs
import numpy as np


def loss_scaled(gt, pred, upsample=False):
    if upsample:
        pred = tf.image.resize_bilinear(pred, tf.shape(gt)[1:3])
    else:
        gt = tf.image.resize_bilinear(gt, tf.shape(pred)[1:3])
    diff = tf.reshape(gt - pred, [-1, 2])
    return tf.user_ops.reduce_sum_2d(tf.expand_dims(tf.sqrt(tf.user_ops.reduce_sum_2d(tf.square(diff))), axis=0)) / tf.cast(tf.shape(diff)[0], tf.float32)


class FlowNetSimple(Network):
    def __init__(self, name, fuse='', **kwargs):
        super().__init__(name)
        self.fuse = fuse

    def build(self, img1, img2, flow):
        concatImgs = tf.concat([img1, img2], 3, 'concatImgs')

        conv1   = convolveLeakyReLU('conv1',   concatImgs, 6,    64,   7, 2)
        conv2   = convolveLeakyReLU('conv2',   conv1,      64,   128,  5, 2)
        conv3   = convolveLeakyReLU('conv3',   conv2,      128,  256,  5, 2)
        conv3_1 = convolveLeakyReLU('conv3_1', conv3,      256,  256,  3, 1)
        conv4   = convolveLeakyReLU('conv4',   conv3_1,    256,  512,  3, 2)
        conv4_1 = convolveLeakyReLU('conv4_1', conv4,      512,  512,  3, 1)
        conv5   = convolveLeakyReLU('conv5',   conv4_1,    512,  512,  3, 2)
        conv5_1 = convolveLeakyReLU('conv5_1', conv5,      512,  512,  3, 1)
        conv6   = convolveLeakyReLU('conv6',   conv5_1,    512,  1024, 3, 2)
        conv6_1 = convolveLeakyReLU('conv6_1', conv6,      1024, 1024, 3, 1)

        # shape0 = concatImgs.shape.as_list()
        # shape1 = conv1.shape.as_list()
        shape2 = conv2.shape.as_list()
        shape3 = conv3.shape.as_list()
        shape4 = conv4.shape.as_list()
        shape5 = conv5.shape.as_list()
        # shape6 = conv6.shape.as_list()

        pred6      = convolve('pred6', conv6_1, 1024, 2, 3, 1)
        upsamp6to5 = upconvolve('upsamp6to5', pred6, 2, 2, 4, 2, shape5[1], shape5[2])
        deconv5    = upconvolveLeakyReLU('deconv5', conv6_1, 1024, 512, 4, 2, shape5[1], shape5[2])
        concat5    = tf.concat([conv5_1, deconv5, upsamp6to5], 3, 'concat5')  # channel = 512+512+2

        pred5      = convolve('pred5', concat5, 512 + 512 + 2, 2, 3, 1)
        upsamp5to4 = upconvolve('upsamp5to4', pred5, 2, 2, 4, 2, shape4[1], shape4[2])
        deconv4    = upconvolveLeakyReLU('deconv4', concat5, 512 + 512 + 2, 256, 4, 2, shape4[1], shape4[2])
        concat4    = tf.concat([conv4_1, deconv4, upsamp5to4], 3, 'concat4')  # channel = 512+256+2

        pred4      = convolve('pred4', concat4, 512 + 256 + 2, 2, 3, 1)
        upsamp4to3 = upconvolve('upsamp4to3', pred4, 2, 2, 4, 2, shape3[1], shape3[2])
        deconv3    = upconvolveLeakyReLU('deconv3', concat4, 512 + 256 + 2, 128, 4, 2, shape3[1], shape3[2])
        concat3    = tf.concat([conv3_1, deconv3, upsamp4to3], 3, 'concat3')  # channel = 256+128+2

        pred3      = convolve('pred3', concat3, 256 + 128 + 2, 2, 3, 1)
        upsamp3to2 = upconvolve('upsamp3to2', pred3, 2, 2, 4, 2, shape2[1], shape2[2])
        deconv2    = upconvolveLeakyReLU('deconv2', concat3, 256 + 128 + 2, 64, 4, 2, shape2[1], shape2[2])
        concat2    = tf.concat([conv2, deconv2, upsamp3to2], 3, 'concat2')  # channel = 128+64+2

        if self.fuse == 'final':
            fuse2      = convolveLeakyReLU('fuse2', concat2, 128 + 64 + 2, 256, 3, 1)
            pred2_1      = convolve('pred2_1', fuse2, 256, 2, 3, 1)
            pred2_2      = convolve('pred2_2', fuse2, 256, 2, 3, 1)
            alpha        = tf.sigmoid(convolve('blend2', fuse2, 256, 1, 3, 1))
            pred2 = pred2_1 * alpha + pred2_2 * (1 - alpha)
        elif self.fuse == '':
            pred2      = convolve('pred2', concat2, 128 + 64 + 2, 2, 3, 1)
        else:
            raise NotImplementedError("{} not implemented".format(self.fuse))

        flow_predictions = {
            'pred2': pred2,
            'pred3': pred3,
            'pred4': pred4,
            'pred5': pred5,
            'pred6': pred6
        }

        loss_weights = {
            'pred6': .32,
            'pred5': .08,
            'pred4': .02,
            'pred3': .01,
            'pred2': .005
        }

        loss = sum([loss_scaled(flow, flow_predictions[k]) * loss_weights[k] for k in loss_weights])

        return {'loss': loss, 'pred2': pred2}


class Framework:
    def __init__(self, devices, network, augmentation=True, train=True, image_shape=[384, 512]):
        self.placeholders = dict()

        # input place holder
        image_shape = list(image_shape)
        img1 = self.get_placeholder(dtype=tf.float32, shape=[None] + image_shape + [3], name='img1') 
        img2 = self.get_placeholder(dtype=tf.float32, shape=[None] + image_shape + [3], name='img2')
        flow = self.get_placeholder(dtype=tf.float32, shape=[None] + image_shape + [2], name='flow')

        # augmentation
        # rotate = tf.get_placeholder(dtype=tf.float32, shape=[None], name='rotate')
        cropX = self.get_placeholder(dtype=tf.int32, shape=[None], name='cropX')
        cropY = self.get_placeholder(dtype=tf.int32, shape=[None], name='cropY')
        gamma = self.get_placeholder(dtype=tf.float32, shape=[None], name='gamma')
        brightness = self.get_placeholder(dtype=tf.float32, shape=[None], name='brightness')
        contrast = self.get_placeholder(dtype=tf.float32, shape=[None], name='contrast')

        # at this time, we do not rotate images
        rotImg1 = img1 / 255.0  # tf.contrib.image.rotate(img1, rotate, 'BILINEAR') / 255.0
        rotImg2 = img2 / 255.0  # tf.contrib.image.rotate(img2, rotate, 'BILINEAR') / 255.0
        rotFlow = flow / 20.0  # tf.contrib.image.rotate(flow, rotate, 'BILINEAR') / 20.0

        dataAugmentOutputTypes = (tf.float32, ) * 3
        if augmentation:
            augImg1, augImg2, augFlow = tf.map_fn(self.dataAugment, (rotImg1, rotImg2, rotFlow, cropX, cropY, gamma, brightness, contrast), dataAugmentOutputTypes)
        else:
            augImg1, augImg2, augFlow = rotImg1, rotImg2, rotFlow

        # Data Processing
        rgbMean = np.array((0.410602, 0.431021, 0.448553)).reshape([1, 1, 1, 3])
        augImg1 = augImg1 - rgbMean
        augImg2 = augImg2 - rgbMean

        gpus = MultiGPUs(devices)
        if train:
            learningRate = self.get_placeholder(tf.float32, [], 'learningRate')
            adamOptimizer = tf.train.AdamOptimizer(learningRate)
            self.predictions, self.adamOpt = gpus(network, [augImg1, augImg2, augFlow], opt=adamOptimizer)
        else:
            self.predictions = gpus(network, [augImg1, augImg2, augFlow])

        self.build_summary(augFlow, self.predictions)

    def get_placeholder(self, dtype, shape, name):
        if name in self.placeholders:
            raise ValueError('Duplicate placeholder "{}"'.format(name))
        h = tf.placeholder(dtype=dtype, shape=shape, name=name)
        self.placeholders[name] = h
        return h

    # the training images are of size 512x384
    # the cropped images are of size 448x320
    # rotation
    # the top-left corner of the crop is in [?, 64) x [?, 64)
    #     where ? depends on whether rotation happened (0 if not, 32 if so)
    # augmentation:
    # 0. rotated    (image + flow, CANNOT be performed as contrib.image.rotate is not working on Windows)
    # 1. cropped    (image + flow)
    # 2. gamma      (image)
    # 3. brightness (image)
    # 4. contrast   (image)
    @staticmethod
    def dataAugment(augTuple):
        img1, img2, flow, cropX, cropY, gamma, brightness, contrast = augTuple
        img1 = tf.image.crop_to_bounding_box(img1, cropX, cropY, 320, 448)
        img1 = img1 ** gamma
        img1 = tf.image.adjust_brightness(img1, brightness)
        img1 = tf.image.adjust_contrast(img1, contrast)

        img2 = tf.image.crop_to_bounding_box(img2, cropX, cropY, 320, 448)
        img2 = img2 ** gamma
        img2 = tf.image.adjust_brightness(img2, brightness)
        img2 = tf.image.adjust_contrast(img2, contrast)

        flow = tf.image.crop_to_bounding_box(flow, cropX, cropY, 320, 448)
        return (img1, img2, flow)

    def feed_dict(self, **kwargs):
        return dict([(self.placeholders[k], v) for k, v in kwargs.items()])

    def build_summary(self, gtFlow, predictions):
        self.loss = tf.reduce_mean(predictions['loss'])
        self.flow = tf.image.resize_bilinear(predictions['pred2'], tf.shape(gtFlow)[1:3])
        self.epe = loss_scaled(gtFlow, self.flow)[0] * 20
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('epe', self.epe)
        self.summaryOp = tf.summary.merge_all()
