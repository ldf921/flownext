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
        self.rgb_mean = nd.array(np.reshape([0.411451, 0.432060, 0.450141], [1, 3, 1, 1]), ctx=self.ctx)
        self.scale = 20

    def train_batch(self, img1, img2, flow):
        with autograd.record():
            pred = self.network(img1 - self.rgb_mean, img2 - self.rgb_mean)
            loss = flownet.multiscale_epe(flow / self.scale, pred)
        epe = nd.mean(flownet.EpeLoss()(pred[-1] * self.scale, flownet.Downsample(4)(flow)))
        loss.backward()
        self.trainer.step(img1.shape[0])
        return epe

    def validate_batch(self, img1, img2, flow):
        pred = self.network(img1 - self.rgb_mean, img2 - self.rgb_mean)
        epe = flownet.EpeLoss()(pred[-1] * self.scale, flownet.Downsample(4)(flow))
        return epe.asnumpy()