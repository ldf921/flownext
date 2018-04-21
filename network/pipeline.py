import mxnet as mx
import numpy as np
from mxnet import nd, gluon, autograd

from .flownet import EpeLoss, Upsample, Flownet, multiscale_epe, Downsample


class PipelineBase:
    def __init__(self, network, trainer, lr_schedule, ctx):
        self.network = network
        self.trainer = trainer
        self.lr_schedule = lr_schedule
        self.ctx = ctx
        pass

    def preprocess(self, img):
        ''' Preprocessing a batch of images with range [0, 1]
        '''
        pass

    def loss(self, pred, label):
        ''' Computing the loss
        '''
        pass

    def metrics(self, pred, label):
        ''' Computing the metrics
        '''
        pass

    def save(self, prefix):
        ''' Saving the state of the pipeline
        Arguments:
            prefix : the file name prefix for the saved file
        '''
        self.network.collect_params().save(prefix + '.params')
        self.trainer.save_states(prefix + '.states')

    def set_learning_rate(self, steps):
        i = 0
        while i < len(self.lr_schedule) and steps > self.lr_schedule[i][0]:
            i += 1
        try:
            lr = self.lr_schedule[i][1]
        except IndexError:
            return False
        self.trainer.set_learning_rate(lr)
        return True

    def train_batch(self, img1, img2, flow, aug, geo_aug):
        losses = []
        epes = []
        batch_size = img1.shape[0]
        img1, img2, flow = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2, flow))
        with autograd.record():
            for img1s, img2s, flows in zip(img1, img2, flow):
                img1s, img2s, flows = geo_aug(img1s, img2s, flows)
                img1s, img2s = aug(img1s, img2s)
                pred = self.network(self.preprocess(img1s), self.preprocess(img2s))
                loss = self.loss(pred, flows)
                epe = self.metrics(pred, flows)

                losses.append(loss)
                epes.append(epe)
        for loss in losses:
            loss.backward()
        # print(losses[0].asnumpy())
        # for k, v in self.network.collect_params().items():
        #     grad = v.grad(ctx=self.ctx[0])
        #     std = nd.sqrt(nd.mean(nd.square(grad)))
        #     print(k, std.asscalar())
        epe = nd.mean(epe)
        self.trainer.step(batch_size, ignore_stale_grad=True)
        return epe

    def validate_batch(self, img1, img2, flow):
        shape = img1.shape
        pad_h = (64 - shape[2] % 64) % 64
        pad_w = (64 - shape[3] % 64) % 64
        if pad_h != 0 or pad_w != 0:
            img1 = nd.pad(img1, mode='constant', constant_value=0, pad_width=(0, 0, 0, 0, 0, pad_h, 0, pad_w))
            img2 = nd.pad(img2, mode='constant', constant_value=0, pad_width=(0, 0, 0, 0, 0, pad_h, 0, pad_w))
        pred = self.network(self.preprocess(img1), self.preprocess(img2))
        epe = self.metrics(pred, flow)
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


class PipelineFlownet:
    def __init__(self, ctx, config):
        self.ctx = ctx
        self.network = Flownet(config=config)
        self.network.hybridize()
        self.network.collect_params().initialize(init=mx.initializer.MSRAPrelu(slope=0.1), ctx=self.ctx)
        self.trainer = gluon.Trainer(self.network.collect_params(), 'adam', {'learning_rate': 1e-4})
        self.rgb_mean = nd.array(np.reshape([0.411451, 0.432060, 0.450141], [1, 3, 1, 1]))
        self.scale = 20

        self.upsampler = Upsample(2, 4)
        self.upsampler.collect_params().initialize(ctx=self.ctx)

        self.epeloss = EpeLoss()
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
                loss = multiscale_epe(flows / self.scale, pred)
                epe = nd.mean(self.epeloss(pred[-1] * self.scale, Downsample(4)(flows)))

                losses.append(loss)
                epes.append(epe)
        for loss in losses:
            loss.backward()
        epe = nd.mean(epe)
        self.trainer.step(batch_size)
        return epe

    def predict_batch_mx(self, img1, img2, flow):
        ''' Predict a batch of samples range [0,1] with network preprocessing and padding
        '''
        shape = img1.shape
        pad_h = (64 - shape[2] % 64) % 64
        pad_w = (64 - shape[3] % 64) % 64
        if pad_h != 0 or pad_w != 0:
            img1 = nd.pad(img1, mode='constant', constant_value=0, pad_width=(0, 0, 0, 0, 0, pad_h, 0, pad_w))
            img2 = nd.pad(img2, mode='constant', constant_value=0, pad_width=(0, 0, 0, 0, 0, pad_h, 0, pad_w))
        rgb_mean = self.rgb_mean.as_in_context(img1.context)
        pred = self.network(img1 - rgb_mean, img2 - rgb_mean)
        return pred

    def validate_batch(self, img1, img2, flow):
        shape = img1.shape
        pred = self.predict_batch_mx(img1, img2, flow)
        epe = self.epeloss(self.upsampler(pred[-1])[:, :, :shape[2], :shape[3]] * self.scale, flow)
        return epe.asnumpy()

    def validate_batch_levels(self, img1, img2, flow, upsamplers):
        shape = img1.shape
        predictions = self.predict_batch_mx(img1, img2, flow)
        epes = [self.epeloss(upsampler(pred)[:, :, :shape[2], :shape[3]] * self.scale, flow).asnumpy() for pred, upsampler in zip(predictions, upsamplers)]
        return epes

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

    def validate_levels(self, img1, img2, flow, batch_size):
        ''' validate the whole dataset at each sample level
        '''
        scales = [64, 32, 16, 8, 4]
        upsamplers = [Upsample(2, scale) for scale in scales]
        for upsampler in upsamplers:
            upsampler.collect_params().initialize(ctx=self.ctx)

        batchEpes = [[] for _ in scales]
        size = len(img1)
        for j in range(0, size, batch_size):
            batch_img1 = img1[j: j + batch_size]
            batch_img2 = img2[j: j + batch_size]
            batch_flow = flow[j: j + batch_size]

            batch_img1 = nd.array(np.transpose(np.stack(batch_img1, axis=0), (0, 3, 1, 2)) / 255.0, ctx=self.ctx[0])
            batch_img2 = nd.array(np.transpose(np.stack(batch_img2, axis=0), (0, 3, 1, 2)) / 255.0, ctx=self.ctx[0])
            batch_flow = nd.array(np.transpose(np.stack(batch_flow, axis=0), (0, 3, 1, 2)), ctx=self.ctx[0])

            for agg, v in zip(batchEpes, self.validate_batch_levels(batch_img1, batch_img2, batch_flow, upsamplers)):
                agg.append(v)

        return [np.mean(np.concatenate(epe)) for epe in batchEpes]