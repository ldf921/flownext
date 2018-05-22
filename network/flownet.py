import mxnet as mx
import numpy as np
from mxnet import nd as F
from mxnet.gluon import nn

from . import layer


class SequentialFeatures(nn.HybridBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layers = []
        with self.name_scope():
            self.net = nn.HybridSequential()
    
    def _output(self):
        self.layers.append(len(self.net) - 1)

    def hybrid_forward(self, F, img):
        feature = img
        ret = []
        for i, layer in enumerate(self.net):
            feature = layer(feature)
            if i in self.layers:
                ret.append(feature)
        return ret


class Downsample(nn.HybridBlock):
    def __init__(self, factor, channels=2, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        self.channels = channels

    @staticmethod
    def _kernel2d(F, w):
        kernel = ((w + 1) - F.abs((w - F.arange(0, w * 2 + 1)))) / (2 * w + 1)
        kernel = F.broadcast_mul(F.expand_dims(kernel, axis=0), F.expand_dims(kernel, axis=1))
        return F.reshape(kernel, (1, 1, w * 2 + 1, w * 2 + 1))

    def hybrid_forward(self, F, img):
        batch_img = F.expand_dims(F.reshape(img, [-3, -2]), axis=1)
        factor=self.factor
        kernel = self._kernel2d(F, factor // 2)
        conv_args = dict(
            weight=kernel, 
            no_bias=True, 
            kernel=(factor + 1,) * 2,
            stride=(factor,) * 2,
            pad=(factor // 2,) * 2,
            num_filter=1)
        nom = F.Convolution(data=F.ones_like(batch_img), **conv_args)
        img = F.Convolution(data=batch_img, **conv_args)
        return F.broadcast_div(F.reshape(img, [-4, -1, self.channels, -3, -2]), F.reshape(nom, [-4, -1, self.channels, -3, -2]))

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
        ''' Upsample a feature map
        Arguments:
            channels : channels of the feature map
            factor : scale to upsample
        '''
        super().__init__(**kwargs)
        with self.name_scope():
            self.upsamp = nn.Conv2DTranspose(channels, factor * 2 - 1, strides=factor, padding=factor - 1, use_bias=False, weight_initializer=Bilinear())
            self.upsamp.weight.lr_mult = 0
            self.upsamp.weight.wd_mult = 0

    def hybrid_forward(self, F, img):
        img = F.pad(img, mode='edge', pad_width=(0, 0, 0, 0, 0, 1, 0, 1))
        return F.slice(self.upsamp(img), begin=(None, None, None, None), end=(None, None, -1, -1))

class ASSP(nn.HybridBlock):
    def __init__(self, channels, conv_params, pooling, **kwargs):
        super().__init__(**kwargs)
        self.conv_params = conv_params
        self.pooling = pooling
        with self.name_scope():
            for kernel_size, r in conv_params:
                net = nn.HybridSequential()
                net.add(nn.Conv2D(channels, kernel_size, padding=r * (kernel_size // 2), dilation=r, prefix='atrous_%d' % r))
                net.add(nn.LeakyReLU(0.1))
                setattr(self, 'atrous_%d' % r, net)
        if pooling:
            self.pool = net = nn.HybridSequential()
            net.add(nn.GlobalAvgPool2D())
            net.add(nn.Dense(channels))
            net.add(nn.LeakyReLU(0.1))

        self.merge = net = nn.HybridSequential()
        net.add(nn.Conv2D(channels, 1))
        net.add(nn.LeakyReLU(0.1))

    def hybrid_forward(self, F, x):
        features = [ getattr(self, 'atrous_%s' % r)(x) for _, r in self.conv_params]
        if hasattr(self, 'pool'):
            global_feature = self.pool(x)
            global_feature = F.reshape(global_feature, (0, 0, 1, 1))
            features.append(F.broadcast_add(global_feature, F.zeros_like(x)))
        return self.merge(F.concat(*features, dim=1))


class ConvHead(SequentialFeatures):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self._conv_bn_relu(64, 7, strides=2, prefix='conv1')
            self.net.add(nn.MaxPool2D(3, 2, 1)) # stride = 4
            self._conv_bn_relu(64, 3, strides=1, prefix='conv2')
            self._conv_bn_relu(64, 3, strides=2, prefix='conv3')
            self._conv_bn_relu(64, 3, strides=1, prefix='conv3_1') # stride = 8
            self._output()

    def _conv_bn_relu(self, channels, kernel, strides=1, **kwargs):
        self.net.add(nn.Conv2D(channels, kernel, strides=strides, padding=(kernel - 1) // 2, use_bias=False, 
            weight_initializer=mx.initializer.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2), 
            **kwargs))
        self.net.add(nn.BatchNorm())
        self.net.add(nn.LeakyReLU(0.1))

class ExtraFlowEncoder(nn.HybridBlock):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.stride = 8
            if config.downsample.value == 'conv':
                self.encoder = ConvHead()
            elif config.downsample.value == 'bilinear':
                self.encoder = Downsample(self.stride, channels=3)
            self.conv = net = nn.HybridSequential()
            net.add(nn.Conv2D(64, 7, padding=3)) 
            net.add(nn.LeakyReLU(0.1))

    def hybrid_forward(self, F, img1, img2):
        feature = F.concat(self.encoder(img1), 
                           self.encoder(img2), 
                           dim=1)
        return self.conv(feature)


class FuseConv(nn.HybridBlock):
    def __init__(self, channels, output_stride, dilation=1, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.pred = nn.Conv2D(2, 3, padding=1, prefix='pred')
            self.conv = nn.Conv2D(channels, 3, strides=1, padding=dilation, dilation=dilation, prefix='conv')
            self.deform = layer.DeformableConv2D(channels, 3, strides=1, padding=dilation, dilation=dilation, use_bias=False, prefix='deform')
            self.upsamp = nn.Conv2D(2, 3, strides=1, padding=dilation, dilation=dilation, prefix='upsamp')
            self.output_stride = output_stride

    def hybrid_forward(self, F, flow, feature, shortcut):
        concat = F.concat(flow, feature, shortcut, dim=1)
        flow = self.pred(concat)

        offset = F.repeat(F.expand_dims(F.flip(flow * 20 / self.output_stride, axis=1), axis=1), 9, axis=1)
        offset = F.reshape(offset, (0, -3, -2))

        deconv = nn.LeakyReLU(0.1)(self.conv(concat) + self.deform(shortcut, offset))
        return self.upsamp(flow), deconv, flow
        

class FlownetDilation(nn.HybridBlock):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.strides = None
        
        with self.name_scope():
            self.conv1   = nn.Conv2D(64, 7, strides=2, padding=3, prefix='conv1')
            self.conv2   = nn.Conv2D(128, 5, strides=2, padding=2, prefix='conv2')
            self.conv3   = nn.Conv2D(256, 5, strides=2, padding=2, prefix='conv3')
            if config.encoder.get(None) is not None:
                self.extra_flow = ExtraFlowEncoder(config.encoder, prefix='flowenc')
            self.conv3_1 = nn.Conv2D(256, 3, strides=1, padding=1, prefix='conv3_1')
            self.conv4   = nn.Conv2D(512, 3, strides=2, padding=1, prefix='conv4')
            self.conv4_1 = nn.Conv2D(512, 3, strides=1, padding=1, prefix='conv4_1')

            dilation = 1

            dilation_5 = config.network.conv5.dilation.get(False)
            self.conv5_net = nn.HybridSequential()
            if dilation_5:
                self.conv5_net.add(nn.Conv2D(512, 3, strides=1, padding=1, prefix='conv5'))
                self.conv5_net.add(nn.LeakyReLU(0.1))
                dilation *= 2
            else:
                self.conv5_net.add(nn.Conv2D(512, 3, strides=2, padding=1, prefix='conv5'))
                self.conv5_net.add(nn.LeakyReLU(0.1))
            
            for i in range(1, config.network.conv5.layers.get(2)):
                self.conv5_net.add(nn.Conv2D(512, 3, strides=1, padding=dilation, dilation=dilation, prefix='conv5_%d' % i))
                self.conv5_net.add(nn.LeakyReLU(0.1))
                print('conv5_%d Dilation%d' % (i, dilation))

            dilation_6 = config.network.conv6.dilation.get(False)
            self.conv6_net = nn.HybridSequential()
            if dilation_6:
                self.conv6_net.add(nn.Conv2D(512, 3, strides=1, padding=dilation, dilation=dilation, prefix='conv6'))
                self.conv6_net.add(nn.LeakyReLU(0.1))
                dilation *= 2
            else:
                assert(not dilation_5)
                self.conv6_net.add(nn.Conv2D(512, 3, strides=2, padding=1, prefix='conv6'))
                self.conv6_net.add(nn.LeakyReLU(0.1))

            if config.network.conv6.assp.get(None) is not None:
                self.conv6_net.add(ASSP(512, config.network.conv6.assp.conv_params.value, config.network.conv6.assp.pooling.value, prefix='assp'))
            else:
                for i in range(1, config.network.conv6.layers.get(2)):
                    self.conv6_net.add(nn.Conv2D(512, 3, strides=1, padding=dilation, dilation=dilation, prefix='conv6_%d' % i))
                    self.conv6_net.add(nn.LeakyReLU(0.1))
                    print('conv6_%d Dilation%d' % (i, dilation))

            if dilation == 4:
                self.strides = [16, 16, 16, 8, 4]
            elif dilation == 2:
                self.strides = [32, 32, 16, 8, 4]
            else:
                self.strides = [64, 32, 16, 8, 4]
            print('Strides', self.strides)

            deform_decode5 = config.network.decode5.deform.get(False)
            self.pred6   = nn.Conv2D(2, 3, padding=1, prefix='pred6')
            if not deform_decode5:
                self.pred5   = nn.Conv2D(2, 3, padding=1, prefix='pred5')
            self.pred4   = nn.Conv2D(2, 3, padding=1, prefix='pred4')
            self.pred3   = nn.Conv2D(2, 3, padding=1, prefix='pred3')
            self.pred2   = nn.Conv2D(2, 3, padding=1, prefix='pred2')

            if self.strides[0] == self.strides[1]:
                dilation = dilation // 2
                self.upsamp5 = nn.Conv2D(2, 3, strides=1, padding=dilation, dilation=dilation, prefix='upsamp5')
                self.deconv5 = nn.Conv2D(512, 3, strides=1, padding=dilation, dilation=dilation,  prefix='deconv5')
            else:
                self.upsamp5 = nn.Conv2DTranspose(2, 4, strides=2, padding=1, prefix='upsamp5')
                self.deconv5 = nn.Conv2DTranspose(512, 4, strides=2, padding=1,  prefix='deconv5')

            if deform_decode5:
                self.fuse5 = FuseConv(256, output_stride=self.strides[1], prefix='fuse5')
                print('deformable fusing')
            else:
                if self.strides[1] == self.strides[2]:
                    dilation = dilation // 2
                    self.upsamp4 = nn.Conv2D(2, 3, strides=1, padding=dilation, dilation=dilation, prefix='upsamp4')
                    self.deconv4 = nn.Conv2D(256, 3, strides=1, padding=dilation, dilation=dilation, prefix='deconv4')
                else:
                    self.upsamp4 = nn.Conv2DTranspose(2, 4, strides=2, padding=1,  prefix='upsamp4')
                    self.deconv4 = nn.Conv2DTranspose(256, 4, strides=2, padding=1,  prefix='deconv4')
            
            self.deform_decode5 = deform_decode5

            self.upsamp3 = nn.Conv2DTranspose(2, 4, strides=2, padding=1,  prefix='upsamp3')
            self.upsamp2 = nn.Conv2DTranspose(2, 4, strides=2, padding=1,  prefix='upsamp2')

            self.deconv3 = nn.Conv2DTranspose(128, 4, strides=2, padding=1,  prefix='deconv3')
            self.deconv2 = nn.Conv2DTranspose(64,  4, strides=2, padding=1,  prefix='deconv2')

    def hybrid_forward(self, F, img1, img2):
        concat_img = F.concat(img1, img2, dim=1)
        conv1 = nn.LeakyReLU(0.1)(self.conv1(concat_img))
        conv2 = nn.LeakyReLU(0.1)(self.conv2(conv1))
        conv3 = nn.LeakyReLU(0.1)(self.conv3(conv2))
        if hasattr(self, 'extra_flow'):
            conv3 = F.concat(conv3, self.extra_flow(img1, img2), dim=1)
        conv3 = nn.LeakyReLU(0.1)(self.conv3_1(conv3))
        conv4 = nn.LeakyReLU(0.1)(self.conv4_1(nn.LeakyReLU(0.1)(self.conv4(conv3))))
        conv5 = self.conv5_net(conv4)
        conv6 = self.conv6_net(conv5)

        pred6 = self.pred6(conv6)

        if self.deform_decode5:
            upsamp4, deconv4, pred5 = self.fuse5(self.upsamp5(pred6), nn.LeakyReLU(0.1)(self.deconv5(conv6)), conv5)
        else:
            concat5 = F.concat(self.upsamp5(pred6), nn.LeakyReLU(0.1)(self.deconv5(conv6)), conv5, dim=1)
            pred5 = self.pred5(concat5)

            upsamp4 = self.upsamp4(pred5)
            deconv4 = nn.LeakyReLU(0.1)(self.deconv4(concat5))

        concat4 = F.concat(upsamp4, deconv4, conv4, dim=1)
        pred4 = self.pred4(concat4)

        concat3 = F.concat(self.upsamp3(pred4), nn.LeakyReLU(0.1)(self.deconv3(concat4)), conv3, dim=1)
        pred3 = self.pred3(concat3)

        concat2 = F.concat(self.upsamp2(pred3), nn.LeakyReLU(0.1)(self.deconv2(concat3)), conv2, dim=1)
        pred2 = self.pred2(concat2)

        return pred6, pred5, pred4, pred3, pred2    

class Flownet(nn.HybridBlock):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.strides = None
        
        with self.name_scope():
            self.conv1   = nn.Conv2D(64, 7, strides=2, padding=3, prefix='conv1')
            self.conv2   = nn.Conv2D(128, 5, strides=2, padding=2, prefix='conv2')
            self.conv3   = nn.Conv2D(256, 5, strides=2, padding=2, prefix='conv3')
            self.conv3_1 = nn.Conv2D(256, 3, strides=1, padding=1, prefix='conv3_1')
            self.conv4   = nn.Conv2D(512, 3, strides=2, padding=1, prefix='conv4')
            self.conv4_1 = nn.Conv2D(512, 3, strides=1, padding=1, prefix='conv4_1')

            dilation = config.network.conv6.dilation.get(1)
            self.impl = config.network.implementation.get('normal')
            if self.impl == 'normal':
                self.conv5   = nn.Conv2D(512, 3, strides=2, padding=1, prefix='conv5')
                self.conv5_1 = nn.Conv2D(512, 3, strides=1, padding=1, prefix='conv5_1')

                if dilation == 1:
                    self.conv6   = nn.Conv2D(1024, 3, strides=2, padding=1, prefix='conv6')
                    self.conv6_1 = nn.Conv2D(1024, 3, strides=1, padding=1, prefix='conv6_1')
                    self.strides = [64, 32, 16, 8, 4]
                else:
                    self.conv6   = nn.Conv2D(1024, 3, strides=1, padding=dilation, dilation=dilation, prefix='conv6')
                    self.conv6_1 = nn.Conv2D(1024, 3, strides=1, padding=dilation, dilation=dilation, prefix='conv6_1')
                    self.strides = [32, 32, 16, 8, 4]
                    print(self.strides)
            elif self.impl == 'dilation':
                if not config.network.conv5.dilation.get(False):
                    self.conv5_net = nn.HybridSequential()
                    self.conv5_net.add(nn.Conv2D(512, 3, strides=2, padding=1, prefix='conv5'))
                    self.conv5_net.add(nn.LeakyReLU(0.1))
                    self.conv5_net.add(nn.Conv2D(512, 3, strides=1, padding=1, prefix='conv5_1'))
                    self.conv5_net.add(nn.LeakyReLU(0.1))
                    self.conv5_net.add(nn.Conv2D(512, 3, strides=1, padding=1, prefix='conv5_2'))
                    self.conv5_net.add(nn.LeakyReLU(0.1))
                else:
                    assert dilation == 2
                    self.conv5_net = nn.HybridSequential()
                    self.conv5_net.add(nn.Conv2D(512, 3, strides=1, padding=1, prefix='conv5'))
                    self.conv5_net.add(nn.LeakyReLU(0.1))
                    self.conv5_net.add(nn.Conv2D(512, 3, strides=1, padding=dilation, dilation=dilation, prefix='conv5_1'))
                    self.conv5_net.add(nn.LeakyReLU(0.1))
                    self.conv5_net.add(nn.Conv2D(512, 3, strides=1, padding=dilation, dilation=dilation, prefix='conv5_2'))
                    self.conv5_net.add(nn.LeakyReLU(0.1))
                    dilation = dilation * 2

                self.conv6_net = nn.HybridSequential()
                self.conv6_net.add(nn.Conv2D(512, 3, strides=1, padding=dilation, dilation=dilation, prefix='conv6'))
                self.conv6_net.add(nn.LeakyReLU(0.1))
                self.conv6_net.add(nn.Conv2D(512, 3, strides=1, padding=dilation, dilation=dilation, prefix='conv6_1'))
                self.conv6_net.add(nn.LeakyReLU(0.1))

                if dilation == 4:
                    self.strides = [16, 16, 16, 8, 4]
                else:
                    self.strides = [32, 32, 16, 8, 4]
                print('Dilation', self.strides)


            self.pred6   = nn.Conv2D(2, 3, padding=1, prefix='pred6')
            self.pred5   = nn.Conv2D(2, 3, padding=1, prefix='pred5')
            self.pred4   = nn.Conv2D(2, 3, padding=1, prefix='pred4')
            self.pred3   = nn.Conv2D(2, 3, padding=1, prefix='pred3')
            self.pred2   = nn.Conv2D(2, 3, padding=1, prefix='pred2')

            if self.strides[0] == self.strides[1]:
                self.upsamp5 = nn.Conv2D(2, 3, strides=1, padding=1, prefix='upsamp5')
                self.deconv5 = nn.Conv2D(512, 3, strides=1, padding=1,  prefix='deconv5')
            else:
                self.upsamp5 = nn.Conv2DTranspose(2, 4, strides=2, padding=1, prefix='upsamp5')
                self.deconv5 = nn.Conv2DTranspose(512, 4, strides=2, padding=1,  prefix='deconv5')

            if self.strides[1] == self.strides[2]:
                self.upsamp4 = nn.Conv2D(2, 3, strides=1, padding=1, prefix='upsamp4')
                self.deconv4 = nn.Conv2D(256, 3, strides=1, padding=1, prefix='deconv4')
            else:
                self.upsamp4 = nn.Conv2DTranspose(2, 4, strides=2, padding=1,  prefix='upsamp4')
                self.deconv4 = nn.Conv2DTranspose(256, 4, strides=2, padding=1,  prefix='deconv4')

            self.upsamp3 = nn.Conv2DTranspose(2, 4, strides=2, padding=1,  prefix='upsamp3')
            self.upsamp2 = nn.Conv2DTranspose(2, 4, strides=2, padding=1,  prefix='upsamp2')

            self.deconv3 = nn.Conv2DTranspose(128, 4, strides=2, padding=1,  prefix='deconv3')
            self.deconv2 = nn.Conv2DTranspose(64,  4, strides=2, padding=1,  prefix='deconv2')

            self.fuse5_impl = config.network.fuse5.implementation.get('none')
            self.fuse5_basline = config.network.fuse5.basline.get(False)
            if self.fuse5_impl == 'deformable':
                if not self.fuse5_basline:
                    self.fuse5_deform = layer.BrunchDeform(512, 3, num_deformable_group=4, split_channels=(384, 128), prefix='fuse5_0', project=True)
                else:
                    net = self.fuse5_deform = nn.HybridSequential()
                    net.add(nn.Conv2D(512, 1, prefix='fuse5_0r'))
                    net.add(nn.LeakyReLU(0.1))
                    net.add(nn.Conv2D(512, 3, padding=1, prefix='fuse5_0'))

                net = self.fuse5 = nn.HybridSequential()
                net.add(nn.LeakyReLU(0.1))
                net.add(nn.Conv2D(512, 3, strides=1, padding=1, prefix='fuse5_1'))
                net = self.fuse5_project = nn.HybridSequential() 
                net.add(nn.Conv2D(512, 3, padding=1, prefix='fuse5_project', use_bias=False))

    def hybrid_forward(self, F, img1, img2):
        concat_img = F.concat(img1, img2, dim=1)
        conv1 = nn.LeakyReLU(0.1)(self.conv1(concat_img))
        conv2 = nn.LeakyReLU(0.1)(self.conv2(conv1))
        conv3 = nn.LeakyReLU(0.1)(self.conv3_1(nn.LeakyReLU(0.1)(self.conv3(conv2))))
        conv4 = nn.LeakyReLU(0.1)(self.conv4_1(nn.LeakyReLU(0.1)(self.conv4(conv3))))

        if self.impl == 'normal':
            conv5 = nn.LeakyReLU(0.1)(self.conv5_1(nn.LeakyReLU(0.1)(self.conv5(conv4))))
            conv6 = nn.LeakyReLU(0.1)(self.conv6_1(nn.LeakyReLU(0.1)(self.conv6(conv5))))
        elif self.impl == 'dilation':
            conv5 = self.conv5_net(conv4)
            conv6 = self.conv6_net(conv5)

        pred6 = self.pred6(conv6)

        concat5 = F.concat(self.upsamp5(pred6), nn.LeakyReLU(0.1)(self.deconv5(conv6)), conv5, dim=1)
        if self.fuse5_impl == 'deformable':
            if self.fuse5_basline:
                fuse5_feature = self.fuse5_deform(concat5)
            else:
                fuse5_feature = self.fuse5_deform(concat5, concat5)
            concat5 = F.LeakyReLU(self.fuse5_project(concat5) + self.fuse5(fuse5_feature), slope=0.1)
        pred5 = self.pred5(concat5)

        concat4 = F.concat(self.upsamp4(pred5), nn.LeakyReLU(0.1)(self.deconv4(concat5)), conv4, dim=1)
        pred4 = self.pred4(concat4)

        concat3 = F.concat(self.upsamp3(pred4), nn.LeakyReLU(0.1)(self.deconv3(concat4)), conv3, dim=1)
        pred3 = self.pred3(concat3)

        concat2 = F.concat(self.upsamp2(pred3), nn.LeakyReLU(0.1)(self.deconv2(concat3)), conv2, dim=1)
        pred2 = self.pred2(concat2)

        return pred6, pred5, pred4, pred3, pred2

class ConvEncoder(SequentialFeatures):
    def __init__(self, config=dict(), **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self._conv_bn_relu(32, 7, strides=2, prefix='conv1')
            self._output()
            self._conv_bn_relu(64, 3, strides=2, prefix='conv2')
            self._conv_bn_relu(64, 3, strides=1, prefix='conv2_1')
            self._output()
            self._conv_bn_relu(128, 3, strides=2, prefix='conv3')
            self._conv_bn_relu(128, 3, strides=1, prefix='conv3_1')
            self._conv_bn_relu(128, 3, strides=1, prefix='conv3_2')
            self._output()
            self._conv_bn_relu(256, 3, strides=2, prefix='conv4')
            self._conv_bn_relu(256, 3, strides=1, prefix='conv4_1')
            self._conv_bn_relu(256, 3, strides=1, prefix='conv4_2')
            self._output()

    def _conv_bn_relu(self, channels, kernel, strides=1, **kwargs):
        self.net.add(nn.Conv2D(channels, kernel, strides=strides, padding=(kernel - 1) // 2, use_bias=False, 
            weight_initializer=mx.initializer.MSRAPrelu(slope=0.1), **kwargs))
        self.net.add(nn.BatchNorm())
        self.net.add(nn.LeakyReLU(0.1))
        

class FlownetEncoder(nn.HybridBlock):
    def __init__(self, config, **kwargs):
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

            dilation = config.network.conv6.dilation.get(1)
            if dilation == 1:
                self.conv6   = nn.Conv2D(1024, 3, strides=2, padding=1, prefix='conv6')
                self.conv6_1 = nn.Conv2D(1024, 3, strides=1, padding=1, prefix='conv6_1')
                self.strides = [64, 32, 16, 8, 4]
            else:
                self.conv6   = nn.Conv2D(1024, 3, strides=1, padding=dilation, dilation=dilation, prefix='conv6')
                self.conv6_1 = nn.Conv2D(1024, 3, strides=1, padding=dilation, dilation=dilation, prefix='conv6_1')
                self.strides = [32, 32, 16, 8, 4]
                print(self.strides)

            self.pred6   = nn.Conv2D(2, 3, padding=1, prefix='pred6')
            self.pred5   = nn.Conv2D(2, 3, padding=1, prefix='pred5')
            self.pred4   = nn.Conv2D(2, 3, padding=1, prefix='pred4')
            self.pred3   = nn.Conv2D(2, 3, padding=1, prefix='pred3')
            self.pred2   = nn.Conv2D(2, 3, padding=1, prefix='pred2')

            if self.strides[0] == self.strides[1]:
                self.upsamp5 = nn.Conv2D(2, 3, strides=1, padding=1, prefix='upsamp5')
                self.deconv5 = nn.Conv2D(512, 3, strides=1, padding=1,  prefix='deconv5')
            else:
                self.upsamp5 = nn.Conv2DTranspose(2, 4, strides=2, padding=1, prefix='upsamp5')
                self.deconv5 = nn.Conv2DTranspose(512, 4, strides=2, padding=1,  prefix='deconv5')

            self.upsamp4 = nn.Conv2DTranspose(2, 4, strides=2, padding=1,  prefix='upsamp4')
            self.upsamp3 = nn.Conv2DTranspose(2, 4, strides=2, padding=1,  prefix='upsamp3')
            self.upsamp2 = nn.Conv2DTranspose(2, 4, strides=2, padding=1,  prefix='upsamp2')

            self.deconv4 = nn.Conv2DTranspose(256, 4, strides=2, padding=1,  prefix='deconv4')
            self.deconv3 = nn.Conv2DTranspose(128, 4, strides=2, padding=1,  prefix='deconv3')
            self.deconv2 = nn.Conv2DTranspose(64,  4, strides=2, padding=1,  prefix='deconv2')

            self.fuse4 = nn.HybridSequential()
            self.fuse4.add(nn.Conv2D(256, 3, padding=1, prefix='fuse4'))
            self.fuse4.add(nn.LeakyReLU(0.1))
            
            self.fuse3 = nn.HybridSequential()
            self.fuse3.add(nn.Conv2D(128, 3, padding=1, prefix='fuse3'))
            self.fuse3.add(nn.LeakyReLU(0.1))

            self.fuse2 = nn.HybridSequential()
            self.fuse2.add(nn.Conv2D(64, 3, padding=1, prefix='fuse2'))
            self.fuse2.add(nn.LeakyReLU(0.1))

            if config.network.encoder.get(True):
                self.encoder = ConvEncoder()
            else:
                self.encoder = None

    def hybrid_forward(self, F, img1, img2):
        concat_img = F.concat(img1, img2, dim=1)
        conv1 = nn.LeakyReLU(0.1)(self.conv1(concat_img))
        conv2 = nn.LeakyReLU(0.1)(self.conv2(conv1))
        conv3 = nn.LeakyReLU(0.1)(self.conv3_1(nn.LeakyReLU(0.1)(self.conv3(conv2))))
        conv4 = nn.LeakyReLU(0.1)(self.conv4_1(nn.LeakyReLU(0.1)(self.conv4(conv3))))
        conv5 = nn.LeakyReLU(0.1)(self.conv5_1(nn.LeakyReLU(0.1)(self.conv5(conv4))))
        conv6 = nn.LeakyReLU(0.1)(self.conv6_1(nn.LeakyReLU(0.1)(self.conv6(conv5))))

        if self.encoder is not None:
            feature1 = self.encoder(img1)

        pred6 = self.pred6(conv6)

        concat5 = F.concat(self.upsamp5(pred6), nn.LeakyReLU(0.1)(self.deconv5(conv6)), conv5, dim=1)
        pred5 = self.pred5(concat5)

        deconv4 = nn.LeakyReLU(0.1)(self.deconv4(concat5))
        if self.encoder is not None:
            deconv4 = F.concat(deconv4, feature1[3], dim=1)
        concat4 = F.concat(self.upsamp4(pred5), self.fuse4(deconv4), conv4, dim=1)
        pred4 = self.pred4(concat4)

        deconv3 = nn.LeakyReLU(0.1)(self.deconv3(concat4))
        if self.encoder is not None:
            deconv3 = F.concat(deconv3, feature1[2], dim=1)
        concat3 = F.concat(self.upsamp3(pred4), self.fuse3(deconv3), conv3, dim=1)
        pred3 = self.pred3(concat3)

        deconv2 = nn.LeakyReLU(0.1)(self.deconv2(concat3))
        if self.encoder is not None:
            deconv2 = F.concat(deconv2, feature1[1], dim=1)
        concat2 = F.concat(self.upsamp2(pred3), self.fuse2(deconv2), conv2, dim=1)
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

# def multiscale_epe(flow, predictions):
#     scales = [64, 32, 16, 8, 4]
#     weights = [.01, .01, .01, .02, .04]
#     losses = [EpeLoss()(p, Downsample(s)(flow)) * w for p, w, s in zip(predictions, weights, scales)]
#     return F.add_n(*losses)

class MultiscaleEpe(nn.HybridBlock):
    def __init__(self, scales, weights, match, **kwargs):
        super().__init__(**kwargs)

        self.scales = scales
        self.weights = weights
        self.match = match
        if match == 'upsampling':
            with self.name_scope():
                for s in self.scales:
                    setattr(self, 'upsampler_{}'.format(s), Upsample(2, s))

    def _get_upsampler(self, s):
        return getattr(self, 'upsampler_{}'.format(s))

    def hybrid_forward(self, F, flow, *predictions):
        if self.match == 'upsampling':
            losses = [EpeLoss()(self._get_upsampler(s)(p), (flow)) * w for p, w, s in zip(predictions, self.weights, self.scales)]
        elif self.match == 'downsampling':
            losses = [EpeLoss()(p, Downsample(s)(flow)) * w for p, w, s in zip(predictions, self.weights, self.scales)]
        else:
            raise NotImplementedError
        return F.add_n(*losses)

def build_network(name):
    network_classes = {
        'Flownet' : Flownet,
        'FlownetEncoder' : FlownetEncoder,
        'FlownetDilation' : FlownetDilation
    }
    if name in network_classes:
        return network_classes[name]
    else:
        raise NotImplementedError

if __name__ == '__main__':
    pass

    # img = nd.arange(16).reshape((1, 1, 4, 4))
    # img = nd.repeat(img, repeats=2, axis=1)
    # upsamp = Upsample(2, 4)
    # upsamp.collect_params().initialize()
    # print(img)
    # print(upsamp(img))    

    # flownet = Flownet()
    # img1 = nd.random_uniform(shape=(5, 3, 320, 448))
    # img2 = nd.random_uniform(shape=(5, 3, 320, 448))
    # flow = nd.random_uniform(shape=(5, 2, 320, 448))
    # flownet = Flownet()
    # flownet.collect_params().initialize()
    # pred = flownet(img1, img2)
    # loss = multiscale_epe(flow, pred)
    # print(loss.asnumpy())
