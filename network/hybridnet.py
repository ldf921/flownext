from .flownet import EpeLoss, Upsample, Downsample, MultiscaleEpe
from .pipeline import PipelineBase

import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision, model_store
from mxnet.initializer import Xavier

def get_features(net, input, layers):
    feature = input
    features = []
    for l in net:
        feature = l(feature)
        features.append(feature)
    return [features[i] for i in layers]

class FlowPrediction(nn.HybridBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            pass


class TopDownBranch(nn.HybridBlock):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.deconv = nn.Conv2DTranspose(channels, 4, strides=2, padding=1,
                weight_initializer=Xavier(rnd_type='gaussian', magnitude=2))
            self.reduce_dim = nn.Conv2D(channels, 1, use_bias=False,
                weight_initializer=Xavier(rnd_type='gaussian', magnitude=2))
            self.conv = nn.Conv2D(channels, 3, padding=1,
                weight_initializer=Xavier(rnd_type='gaussian', magnitude=2))

    def hybrid_forward(self, F, input, skip):
        fm = F.concat(nn.Activation('relu')(self.deconv(input)), self.reduce_dim(skip))
        fm = self.conv(fm)
        fm = nn.Activation('relu')(fm)
        return fm


class HybridNet(nn.HybridBlock):
    '''
    Comment:
        Resnet50 Feature Dims [4, 256] [7, 2048]
    '''
    def __init__(self, features, **kwargs):
        super().__init__(**kwargs)
        self.feature_layers = [4, 5, 6, 7]
        self.feature_dims = [64, 128, 256, 512]
        self.features = features
        with self.name_scope():
            # self.backbone = vision.resnet50_v1()
            self.reduce_dim = nn.Conv2D(1024, 1, use_bias=False)

            self.flow = nn.HybridSequential(prefix='flow')
            self.flow.add(nn.Conv2D(1024, 3, padding=1,
                weight_initializer=Xavier(rnd_type='gaussian', magnitude=2)))
            self.flow.add(nn.Activation('relu'))
            self.flow.add(nn.Conv2D(1024, 3, padding=1,
                weight_initializer=Xavier(rnd_type='gaussian', magnitude=2)))
            self.flow.add(nn.Activation('relu'))
            self.flow.add(nn.Conv2D(512, 3, padding=1,
                weight_initializer=Xavier(rnd_type='gaussian', magnitude=2)))
            self.flow.add(nn.Activation('relu'))

            self.upbranch4 = TopDownBranch(self.feature_dims[2]) 
            self.upbranch3 = TopDownBranch(self.feature_dims[1]) 
            self.upbranch2 = TopDownBranch(self.feature_dims[0]) 

            self.output5 = nn.Conv2D(2, 3, padding=1)
            self.output4 = nn.Conv2D(2, 3, padding=1)
            self.output3 = nn.Conv2D(2, 3, padding=1)
            self.output2 = nn.Conv2D(2, 3, padding=1)
        
    def hybrid_forward(self, F, img1, img2):
        features1 = get_features(self.features, img1, self.feature_layers)
        features2 = get_features(self.features, img2, self.feature_layers)
        concat_feature = F.concat(self.reduce_dim(features1[-1]), self.reduce_dim(features2[-1]), dim=1)
        
        preds = []
        flow = self.flow(concat_feature)
        preds.append(self.output5(flow))

        flow = self.upbranch4(flow, features1[2])
        preds.append(self.output4(flow))

        flow = self.upbranch3(flow, features1[1])
        preds.append(self.output3(flow))

        flow = self.upbranch2(flow, features1[0])
        preds.append(self.output2(flow))

        return preds


class Pipeline(PipelineBase):
    def __init__(self, ctx, lr_mult):
        model = vision.resnet50_v1(root=r'\\msralab\ProjectData\ScratchSSD\Users\v-dinliu\.mxnet\models', pretrained=True, ctx=ctx)
        # network.backbone.load_params(model_store.get_model_file('resnet50_v1', 
        #     root=r'\\msralab\ProjectData\ScratchSSD\Users\v-dinliu\.mxnet\models'), ctx=ctx)
        network = HybridNet(model.features)
        network.hybridize()
        for k, v in network.collect_params().items():
            if k.startswith(network.prefix):
                v.initialize(ctx=ctx)
        
        trainer = gluon.trainer.Trainer(network.collect_params(), 'sgd', {'learning_rate' : 1e-3, 'momentum' : 0.9, 'wd' : 2e-4})
        lr_schedule = [(200_000, 1e-3), (300_000, 5e-4), (400_000, 1e-4)]
        lr_schedule = [ (s, lr * lr_mult) for s, lr in lr_schedule ]
        super().__init__(network, trainer, lr_schedule, ctx)

        self.color_mean =nd.reshape(nd.array([0.485, 0.456, 0.406]), [1, 3, 1, 1])
        self.color_std = nd.reshape(nd.array([0.229, 0.224, 0.225]), [1, 3, 1, 1])
        self.epeloss = EpeLoss()
        self.epeloss.hybridize()
        self.upsampler = Upsample(2, 4)
        self.upsampler.collect_params().initialize(ctx=ctx)
        self.scale = 20

        self.msloss = MultiscaleEpe([32, 16, 8, 4], [0.1, 0.1, 0.2, 0.4])
        self.msloss.hybridize()

    def loss(self, pred, label):
        return self.msloss(label, *pred)

    def metrics(self, pred, label):
        shape = label.shape
        epe = self.epeloss(nd.slice(self.upsampler(pred[-1]), 
            begin=(None, None, 0, 0), 
            end=(None, None, shape[2], shape[3])) * self.scale, label)
        return epe
        
    def preprocess(self, img):
        return nd.broadcast_div(nd.broadcast_minus(img, self.color_mean.as_in_context(img.context)),
            self.color_std.as_in_context(img.context))