import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd
import numpy as np

class Downsample(nn.HybridBlock):
    def __init__(self, factor, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    @staticmethod
    def _kernel2d(F, w):
        kernel = ((w + 1) - F.abs((w - F.arange(0, w * 2 + 1)))) / (2 * w + 1)
        kernel = F.broadcast_mul(F.expand_dims(kernel, axis=0), F.expand_dims(kernel, axis=1))
        return F.reshape(kernel, (1, 1, w * 2 + 1, w * 2 + 1))

    def hybrid_forward(self, F, img):
        batch_img = F.expand_dims(F.reshape(img, [-3, -2]), axis=1)
        factor=self.factor
        kernel = self._kernel2d(F, factor-1)
        nom = F.Convolution(data=F.ones_like(batch_img), 
                            weight=kernel, 
                            no_bias=True, 
                            kernel=(factor*2-1, factor*2-1),
                            stride=(factor, factor),
                            pad=(factor-1, factor-1), 
                            num_filter=1)
        img = F.Convolution(data=batch_img, 
                            weight=kernel, 
                            no_bias=True, 
                            kernel=(factor*2-1, factor*2-1),
                            stride=(factor, factor),
                            pad=(factor-1, factor-1), 
                            num_filter=1)
        return F.broadcast_div(F.reshape(img, [-4, -1, 2, -3, -2]), F.reshape(nom, [-4, -1, 2, -3, -2]))

class Bilinear(mx.initializer.Initializer):
    def _init_weight(self, _, arr):
        weight = np.zeros(arr.shape)
        w = weight.shape[2]
        c = w // 2 
        for k in range(weight.shape[0]):
            for i in range(w):
                for j in range(w):
                    weight[k, k, i, j] = (1 - abs(i - c) / (c + 1)) * (1 - abs(j - c) / (c + 1))
        arr[:] = weight

class Upsample(nn.HybridBlock):
    def __init__(self, channels, factor, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.upsamp = nn.Conv2DTranspose(channels, factor * 2 - 1, strides=factor, padding=factor - 1, weight_initializer=Bilinear())

    def hybrid_forward(self, F, img):
        img = F.pad(img, mode='edge', pad_width=(0, 0, 0, 0, 0, 1, 0, 1))
        return self.upsamp(img)[:, :, :-1, :-1]

class Flownet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.conv1   = nn.Conv2D(64, 7, strides=2, padding=3, prefix='conv1')
            self.conv2   = nn.Conv2D(128, 5, strides=2, padding=2, prefix='conv2')
            self.conv3   = nn.Conv2D(256, 5, strides=2, padding=2, prefix='conv3')
            self.conv3_1 = nn.Conv2D(256, 3, strides=1, padding=1, prefix='conv3_1')
            self.conv4   = nn.Conv2D(512, 3, strides=2, padding=1, prefix='conv4')
            self.conv4_1 = nn.Conv2D(512, 3, strides=1, padding=1, prefix='conv4_1')
            self.conv5   = nn.Conv2D(512, 3, strides=2, padding=1, prefix='conv5')
            self.conv5_1 = nn.Conv2D(512, 3, strides=1, padding=1, prefix='conv5_1')
            self.conv6   = nn.Conv2D(1024, 3, strides=2, padding=1, prefix='conv6')
            self.conv6_1 = nn.Conv2D(1024, 3, strides=1, padding=1, prefix='conv6_1')

            self.pred6   = nn.Conv2D(2, 3, padding=1, prefix='pred6')
            self.pred5   = nn.Conv2D(2, 3, padding=1, prefix='pred5')
            self.pred4   = nn.Conv2D(2, 3, padding=1, prefix='pred4')
            self.pred3   = nn.Conv2D(2, 3, padding=1, prefix='pred3')
            self.pred2   = nn.Conv2D(2, 3, padding=1, prefix='pred2')

            self.upsamp5 = nn.Conv2DTranspose(2, 4, strides=2, padding=1, prefix='upsamp5')
            self.upsamp4 = nn.Conv2DTranspose(2, 4, strides=2, padding=1,  prefix='upsamp4')
            self.upsamp3 = nn.Conv2DTranspose(2, 4, strides=2, padding=1,  prefix='upsamp3')
            self.upsamp2 = nn.Conv2DTranspose(2, 4, strides=2, padding=1,  prefix='upsamp2')

            self.deconv5 = nn.Conv2DTranspose(512, 4, strides=2, padding=1,  prefix='deconv5')
            self.deconv4 = nn.Conv2DTranspose(256, 4, strides=2, padding=1,  prefix='deconv4')
            self.deconv3 = nn.Conv2DTranspose(128, 4, strides=2, padding=1,  prefix='deconv3')
            self.deconv2 = nn.Conv2DTranspose(64,  4, strides=2, padding=1,  prefix='deconv2')

    def hybrid_forward(self, F, img1, img2):
        concat_img = F.concat(img1, img2, dim=1)
        conv1 = nn.LeakyReLU(0.1)(self.conv1(concat_img))
        conv2 = nn.LeakyReLU(0.1)(self.conv2(conv1))
        conv3 = nn.LeakyReLU(0.1)(self.conv3_1(nn.LeakyReLU(0.1)(self.conv3(conv2))))
        conv4 = nn.LeakyReLU(0.1)(self.conv4_1(nn.LeakyReLU(0.1)(self.conv4(conv3))))
        conv5 = nn.LeakyReLU(0.1)(self.conv5_1(nn.LeakyReLU(0.1)(self.conv5(conv4))))
        conv6 = nn.LeakyReLU(0.1)(self.conv6_1(nn.LeakyReLU(0.1)(self.conv6(conv5))))

        pred6 = self.pred6(conv6)

        concat5 = F.concat(self.upsamp5(pred6), nn.LeakyReLU(0.1)(self.deconv5(conv6)), conv5, dim=1)
        pred5 = self.pred5(concat5)

        concat4 = F.concat(self.upsamp4(pred5), nn.LeakyReLU(0.1)(self.deconv4(concat5)), conv4, dim=1)
        pred4 = self.pred4(concat4)

        concat3 = F.concat(self.upsamp3(pred4), nn.LeakyReLU(0.1)(self.deconv3(concat4)), conv3, dim=1)
        pred3 = self.pred3(concat3)

        concat2 = F.concat(self.upsamp2(pred3), nn.LeakyReLU(0.1)(self.deconv2(concat3)), conv2, dim=1)
        pred2 = self.pred2(concat2)

        return pred6, pred5, pred4, pred3, pred2


class EpeLoss(nn.HybridBlock):
    ''' Compute Endpoint Error Loss
    Arguments
    ==============
    - pred [N, C, H, W] : predictions
    - label [N, C, H, W] : flow_groundtruth
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def hybrid_forward(self, F, pred, label):
        loss = F.sqrt(F.sum(F.square(pred - label), axis=1))
        return F.mean(loss, axis=0, exclude=True)

from mxnet import nd as F
def multiscale_epe(flow, predictions):
    scales = [64, 32, 16, 8, 4]
    weights = [.01, .01, .01, .02, .04]
    losses = [EpeLoss()(p, Downsample(s)(flow)) * w for p, w, s in zip(predictions, weights, scales)]
    return F.add_n(*losses)

class MultiscaleEpe(nn.HybridBlock):
    def __init__(self, scales, weights, **kwargs):
        self.scales = scales
        self.weights = weights
        super().__init__(**kwargs)

    def hybrid_forward(self, F, flow, *predictions):
        losses = [EpeLoss()(p, Downsample(s)(flow)) * w for p, w, s in zip(predictions, self.weights, self.scales)]
        return F.add_n(*losses)

if __name__ == '__main__':

    img = nd.arange(16).reshape((1, 1, 4, 4))
    img = nd.repeat(img, repeats=2, axis=1)
    upsamp = Upsample(2, 4)
    upsamp.collect_params().initialize()
    print(img)
    print(upsamp(img))    

    # flownet = Flownet()
    # img1 = nd.random_uniform(shape=(5, 3, 320, 448))
    # img2 = nd.random_uniform(shape=(5, 3, 320, 448))
    # flow = nd.random_uniform(shape=(5, 2, 320, 448))
    # flownet = Flownet()
    # flownet.collect_params().initialize()
    # pred = flownet(img1, img2)
    # loss = multiscale_epe(flow, pred)
    # print(loss.asnumpy())