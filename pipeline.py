import mxnet as mx
import flownet
import numpy as np

from mxnet import gluon, autograd, nd

class Pipeline:
    def __init__(self, ctx):
        self.ctx = ctx
        self.network = flownet.Flownet()
        self.network.hybridize()
        self.network.collect_params().initialize(init=mx.initializer.MSRAPrelu(slope=0.1), ctx=self.ctx)
        self.trainer = gluon.Trainer(self.network.collect_params(), 'adam', {'learning_rate': 1e-4})
        self.rgb_mean = nd.array(np.reshape([0.411451, 0.432060, 0.450141], [1, 3, 1, 1]))
        self.scale = 20

        self.upsampler = flownet.Upsample(2, 4)
        self.upsampler.collect_params().initialize(ctx=self.ctx)

        self.epeloss = flownet.EpeLoss()
        self.epeloss.hybridize()

    def train_batch(self, img1, img2, flow, aug, geo_aug):
        losses = []
        epes = []
        batch_size = img1.shape[0]
        img1, img2, flow = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2, flow))
        with autograd.record():
            for img1s, img2s, flows in zip(img1, img2, flow):
                img1s, img2s, flows = geo_aug(img1s, img2s, flows)
                img1s, img2s = aug(img1s, img2s)
                rgb_mean = self.rgb_mean.as_in_context(img1s.context)
                pred = self.network(img1s - rgb_mean, img2s - rgb_mean)
                loss = flownet.multiscale_epe(flows / self.scale, pred)
                epe = nd.mean(self.epeloss(pred[-1] * self.scale, flownet.Downsample(4)(flows)))

                losses.append(loss)
                epes.append(epe)
        for loss in losses:
            loss.backward()
        epe = nd.mean(epe)
        self.trainer.step(batch_size)
        return epe

    def validate_batch(self, img1, img2, flow):
        shape = img1.shape
        pad_h = (64 - shape[2] % 64) % 64
        pad_w = (64 - shape[3] % 64) % 64
        if pad_h != 0 or pad_w != 0:
            img1 = nd.pad(img1, mode='constant', constant_value=0, pad_width=(0, 0, 0, 0, 0, pad_h, 0, pad_w))
            img2 = nd.pad(img2, mode='constant', constant_value=0, pad_width=(0, 0, 0, 0, 0, pad_h, 0, pad_w))
        rgb_mean = self.rgb_mean.as_in_context(img1.context)
        pred = self.network(img1 - rgb_mean, img2 - rgb_mean)
        epe = self.epeloss(self.upsampler(pred[-1])[:, :, :shape[2], :shape[3]] * self.scale, flow)
        return epe.asnumpy()

    def validate(self, img1, img2, flow, batch_size):
        ''' validate the whole dataset
        '''
        batchEpe = []
        size = len(img1)
        for j in range(0, size, batch_size):
            batch_img1 = img1[j: j + batch_size]
            batch_img2 = img2[j: j + batch_size]
            batch_flow = flow[j: j + batch_size]

            batch_img1 = nd.array(np.transpose(np.stack(batch_img1, axis=0), (0, 3, 1, 2)) / 255.0, ctx=self.ctx[0])
            batch_img2 = nd.array(np.transpose(np.stack(batch_img2, axis=0), (0, 3, 1, 2)) / 255.0, ctx=self.ctx[0])
            batch_flow = nd.array(np.transpose(np.stack(batch_flow, axis=0), (0, 3, 1, 2)), ctx=self.ctx[0])

            batchEpe.append(self.validate_batch(batch_img1, batch_img2, batch_flow))
        return np.mean(np.concatenate(batchEpe))