from .flownet import EpeLoss, Upsample, Downsample, MultiscaleEpe, Flownet, Bilinear
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


class HybridNetCoarse(nn.HybridBlock):
    '''
    Comment:
        Resnet50 Feature Dims [4, 256] [7, 2048]
    '''
    def __init__(self, features, config, **kwargs):
        super().__init__(**kwargs)
        self.feature_layers = [4, 5, 6, 7]
        self.feature_dims = [64, 128, 256, 512]
        self.features = features
        with self.name_scope():
            # self.backbone = vision.resnet50_v1()
            self.reduce_dim = nn.HybridSequential()
            self.reduce_dim.add(nn.Conv2D(64, 1))
            self.reduce_dim.add(nn.BatchNorm()) 
            self.reduce_dim.add(nn.Activation('relu')) 

            self.flow = nn.HybridSequential(prefix='flow')
            self.flow.add(nn.Conv2D(64, 7, padding=3,
                weight_initializer=Xavier(rnd_type='gaussian', magnitude=2)))
            self.flow.add(nn.LeakyReLU(0.1))
            self.flow.add(nn.Conv2D(32, 7, padding=3,
                weight_initializer=Xavier(rnd_type='gaussian', magnitude=2)))
            self.flow.add(nn.LeakyReLU(0.1))
            self.flow.add(nn.Conv2D(16, 7, padding=3,
                weight_initializer=Xavier(rnd_type='gaussian', magnitude=2)))
            self.flow.add(nn.LeakyReLU(0.1))
            self.flow.add(nn.Conv2D(2, 7, padding=3,
                weight_initializer=Xavier(rnd_type='gaussian', magnitude=2)))
        
    def hybrid_forward(self, F, img1, img2):
        features1 = get_features(self.features, img1, self.feature_layers)
        features2 = get_features(self.features, img2, self.feature_layers)
        concat_feature = F.concat(self.reduce_dim(features1[-1]), self.reduce_dim(features2[-1]), dim=1)
        
        flow = self.flow(concat_feature)
        return flow,


class HybridNetS16(nn.HybridBlock):
    '''
    Comment:
        Resnet50 Feature Dims [4, 256] [7, 2048]
    '''
    def __init__(self, features, config, **kwargs):
        super().__init__(**kwargs)
        self.feature_layers = [4, 5, 6, 7]
        self.feature_dims = [64, 128, 256, 512]
        self.features = features
        self.dilation = get_param(config, 'network.dilation', 1)
        with self.name_scope():
            # self.backbone = vision.resnet50_v1()
            self.reduce_dim = nn.HybridSequential()
            self.reduce_dim.add(nn.Conv2D(64, 1))
            self.reduce_dim.add(nn.BatchNorm()) 
            self.reduce_dim.add(nn.Activation('relu')) 

            self.reduce_dim_2 = nn.HybridSequential()
            self.reduce_dim_2.add(nn.Conv2D(64, 1))
            self.reduce_dim_2.add(nn.BatchNorm()) 

            self.upsampler = Upsample(64, 2)

            self.flow = nn.HybridSequential(prefix='flow')
            self.flow.add(nn.Conv2D(64, 7, padding=3 * self.dilation, dilation=self.dilation,
                weight_initializer=Xavier(rnd_type='gaussian', magnitude=2)))
            self.flow.add(nn.LeakyReLU(0.1))
            self.flow.add(nn.Conv2D(32, 7, padding=3 * self.dilation, dilation=self.dilation,
                weight_initializer=Xavier(rnd_type='gaussian', magnitude=2)))
            self.flow.add(nn.LeakyReLU(0.1))
            self.flow.add(nn.Conv2D(16, 7, padding=3 * self.dilation, dilation=self.dilation,
                weight_initializer=Xavier(rnd_type='gaussian', magnitude=2)))
            self.flow.add(nn.LeakyReLU(0.1))
            self.flow.add(nn.Conv2D(2, 7, padding=3 * self.dilation, dilation=self.dilation,
                weight_initializer=Xavier(rnd_type='gaussian', magnitude=2)))

    def _top_down_features(self, F, features):    
        net = self.reduce_dim(features[-1])
        net = self.upsampler(net) 
        net = F.Activation(net + self.reduce_dim_2(features[-2]), 'relu')
        return net

    def hybrid_forward(self, F, img1, img2):
        features1 = get_features(self.features, img1, self.feature_layers)
        features2 = get_features(self.features, img2, self.feature_layers)

        concat_feature = F.concat(self._top_down_features(F, features1), self._top_down_features(F, features2), dim=1)
      
        flow = self.flow(concat_feature)
        return flow,

def conv_bn_relu(channels, kernel_size, strides=1, padding=None):
    if padding is None:
        padding = (kernel_size - 1) // 2
    net = nn.HybridSequential()
    net.add(nn.Conv2D(channels, kernel_size, strides=strides, padding=padding, use_bias=False))
    net.add(nn.BatchNorm()) 
    net.add(nn.Activation('relu'))
    return net

class HybridNetDecoder(nn.HybridBlock):
    '''
    Comment:
        Resnet50 Feature Dims [4, 256] [7, 2048]
    '''
    def __init__(self, features, config, **kwargs):
        super().__init__(**kwargs)
        self.feature_layers = [4, 5, 6, 7]
        self.feature_dims = [64, 128, 256, 512]
        self.features = features
        with self.name_scope():
            # self.backbone = vision.resnet50_v1()
            self.add_layers('reduce_dim', 
                conv_bn_relu(64, 1),
                conv_bn_relu(64, 1),
                conv_bn_relu(64, 1),
                conv_bn_relu(64, 1)
            )

            self.add_layers('deconv', 
                nn.Conv2DTranspose(64, 4, strides=2, padding=1, weight_initializer=mx.initializer.MSRAPrelu(slope=0.1)),
                nn.Conv2DTranspose(64, 4, strides=2, padding=1, weight_initializer=mx.initializer.MSRAPrelu(slope=0.1)),
                nn.Conv2DTranspose(64, 4, strides=2, padding=1, weight_initializer=mx.initializer.MSRAPrelu(slope=0.1))
            )

            self.add_layers('pred',
                self._prediction_head(),
                self._prediction_head(),
                self._prediction_head(),
                self._prediction_head()
            )

            self.flow = nn.HybridSequential(prefix='flow')
            self.flow.add(nn.Conv2D(64, 7, padding=3,
                weight_initializer=Xavier(rnd_type='gaussian', magnitude=2)))
            self.flow.add(nn.LeakyReLU(0.1))
            self.flow.add(nn.Conv2D(64, 7, padding=3,
                weight_initializer=Xavier(rnd_type='gaussian', magnitude=2)))
            self.flow.add(nn.LeakyReLU(0.1))
            self.flow.add(nn.Conv2D(64, 7, padding=3,
                weight_initializer=Xavier(rnd_type='gaussian', magnitude=2)))
            self.flow.add(nn.LeakyReLU(0.1))

    def _prediction_head(self):
        net = nn.HybridSequential(prefix='pred')
        net.add(nn.Conv2D(32, 3, padding=1, weight_initializer=mx.initializer.MSRAPrelu(slope=0.1)))
        net.add(nn.LeakyReLU(0.1))
        net.add(nn.Conv2D(2, 3, padding=1))
        return net


    def add_layers(self, name, *layers):
        for i, l in enumerate(layers):
            setattr(self, name + '%d' % i, l)

    def get_layer(self, name, index):
        return getattr(self, name + '%d' % index)
        
    def hybrid_forward(self, F, img1, img2):
        features1 = get_features(self.features, img1, self.feature_layers)
        features2 = get_features(self.features, img2, self.feature_layers)

        concat_feature = F.concat(self.get_layer('reduce_dim', 3)(features1[3]), self.get_layer('reduce_dim', 3)(features2[3]), dim=1)
        concat_feature = self.flow(concat_feature)
        
        concat_features = [concat_feature]

        for i in reversed(range(3)):
            conv = self.get_layer('reduce_dim', i)(features1[i])
            concat_feature = F.LeakyReLU(self.get_layer('deconv', i)(concat_feature), slope=0.1)
            concat_feature = F.concat(conv, concat_feature, dim = 1)
            concat_features.append(concat_feature)

        pred = [self.get_layer('pred', 3 - i)(concat_features[i]) for i in range(4)]
        returdn pred

class Spynet(nn.HybridBlock):
    ''' Implementation of SpyNet block https://arxiv.org/pdf/1611.00850.pdf
    Comment:
    L1(/16), L2(/8), L3(/4), L2(/2), L5(/1)
    '''
    def __init__(self, features, config, **kwargs):
        super().__init__(**kwargs)
        self.scale = get_param(config, 'network.scale')

        with self.name_scope():
            self.flow = nn.HybridSequential(prefix='flow')

            act_param = get_param(config, 'network.activation', dict())
            act_func = self._builder(lambda : nn.Activation('relu'), **act_param)

            weight_init_param = get_param(config, 'network.weight_init', dict())
            weight_init_func = self._builder(lambda : Xavier(rnd_type='gaussian', magnitude=2), **weight_init_param)

            self.flow.add(nn.Conv2D(32, 7, padding=3,
                weight_initializer=weight_init_func()))
            self.flow.add(act_func())
            self.flow.add(nn.Conv2D(64, 7, padding=3,
                weight_initializer=weight_init_func()))
            self.flow.add(act_func())
            self.flow.add(nn.Conv2D(32, 7, padding=3,
                weight_initializer=weight_init_func()))
            self.flow.add(act_func())
            self.flow.add(nn.Conv2D(16, 7, padding=3,
                weight_initializer=weight_init_func()))
            self.flow.add(act_func())
            self.flow.add(nn.Conv2D(2, 7, padding=3,
                weight_initializer=weight_init_func()))
        
    def hybrid_forward(self, F, img1, img2):
        concat_feature = F.concat(Downsample(self.scale, channels=3)(img1), Downsample(self.scale, channels=3)(img2), dim=1)
        # print(concat_feature.shape)
        flow = self.flow(concat_feature)
        return flow,

    @staticmethod
    def _builder(default=None, typename=None, args=[], kwargs={}):
        if typename is None:
            return default
        else:
            init = eval(typename)
            return lambda : init(*args, **kwargs)


def get_param(config, key, default=None):
    for k in key.split('.'):
        if k not in config:
            print('Default {} to {}'.format(key, default))
            return default
        config = config[k]
    return config


# class Pipeline(PipelineBase):
#     def __init__(self, ctx, config, lr_mult=None):
#         lr_mult = config['optimizer']['lr_mult']
#         lr_schedule = config['optimizer']['learning_rate']
#         lr_schedule = [(s, lr * lr_mult) for s, lr in lr_schedule]

#         model = vision.resnet50_v1(root=r'\\msralab\ProjectData\ScratchSSD\Users\v-dinliu\.mxnet\models', pretrained=True, ctx=ctx)
#         # network.backbone.load_params(model_store.get_model_file('resnet50_v1', 
#         #     root=r'\\msralab\ProjectData\ScratchSSD\Users\v-dinliu\.mxnet\models'), ctx=ctx)
#         network = HybridNet(model.features)
#         network.hybridize()
#         for k, v in network.collect_params().items():
#             if k.startswith(network.prefix):
#                 v.initialize(ctx=ctx)
#         for k, v in config['optimizer'].get('lr_mult_layer', dict()):
#             for _, param in getattr(network, k).collect_params().items():
#                 param.lr_mult = v
        
#         trainer = gluon.trainer.Trainer(network.collect_params(), 'sgd', {'learning_rate' : 1e-3, 'momentum' : 0.9, 'wd' : 2e-4})
        
#         super().__init__(network, trainer, lr_schedule, ctx)

#         self.color_mean =nd.reshape(nd.array([0.485, 0.456, 0.406]), [1, 3, 1, 1])
#         self.color_std = nd.reshape(nd.array([0.229, 0.224, 0.225]), [1, 3, 1, 1])
#         self.epeloss = EpeLoss()
#         self.epeloss.hybridize()
#         self.upsampler = Upsample(2, 4)
#         self.upsampler.collect_params().initialize(ctx=ctx)
#         self.scale = get_param(config, 'network.scale', 20)

#         self.msloss = MultiscaleEpe([32, 16, 8, 4], [0.1, 0.1, 0.2, 0.4])
#         self.msloss.hybridize()

#     def loss(self, pred, label):
#         return self.msloss(label / self.scale, *pred)

#     def metrics(self, pred, label):
#         shape = label.shape
#         epe = self.epeloss(nd.slice(self.upsampler(pred[-1]), 
#             begin=(None, None, 0, 0), 
#             end=(None, None, shape[2], shape[3])) * self.scale, label)
#         return epe
        
#     def preprocess(self, img):
#         return nd.broadcast_div(nd.broadcast_minus(img, self.color_mean.as_in_context(img.context)),
#             self.color_std.as_in_context(img.context))


class PipelineCoarse(PipelineBase):
    def __init__(self, ctx, config, lr_mult=None):
        lr_mult = config['optimizer']['lr_mult']
        lr_schedule = config['optimizer']['learning_rate']
        lr_schedule = [(s, lr * lr_mult) for s, lr in lr_schedule]
        optimizer_type = config['optimizer'].get('type', 'sgd')
        optimizer_params = {'learning_rate' : 1e-3, 'wd' : 2e-4}
        if optimizer_type == 'sgd':
            optimizer_params['momentum'] = 0.9
        optimizer_params.update(config['optimizer'].get('params', dict()))

        model = vision.resnet50_v1(root=r'\\msralab\ProjectData\ScratchSSD\Users\v-dinliu\.mxnet\models', pretrained=True, ctx=ctx)
        
        # HybridNetCoarse
        Network = eval(config['network']['class'])
        scale = get_param(config, 'network.scale', 20)
        network = Network(model.features, config)
        network.hybridize()
        for k, v in network.collect_params().items():
            if k.startswith(network.prefix):
                v.initialize(ctx=ctx)
        
        for k, v in config['optimizer'].get('lr_mult_layer', dict()).items():
            for _, param in getattr(network, k).collect_params().items():
                param.lr_mult = param.lr_mult * v
        
        trainer = gluon.trainer.Trainer(network.collect_params(), optimizer_type, optimizer_params)
        
        super().__init__(network, trainer, lr_schedule, ctx)
        self.epeloss = EpeLoss()
        self.epeloss.hybridize()
        self.color_mean =nd.reshape(nd.array([0.485, 0.456, 0.406]), [1, 3, 1, 1])
        self.color_std = nd.reshape(nd.array([0.229, 0.224, 0.225]), [1, 3, 1, 1])
        self.upsampler = Upsample(2, 32)
        self.upsampler.collect_params().initialize(ctx=ctx)
        self.scale = scale

        loss_scales = get_param(config, 'loss.scales', [32])
        loss_weights = get_param(config, 'loss.weights', [1 for _ in loss_scales])
        self.msloss = MultiscaleEpe(loss_scales, loss_weights, match=get_param(config, 'loss.match', 'downsampling'))
        self.msloss.hybridize()
        self.msloss.collect_params().initialize(ctx=self.ctx)


    def loss(self, pred, label):
        return self.msloss(label / self.scale, *pred)

    def metrics(self, pred, label):
        shape = label.shape
        epe = self.epeloss(nd.slice(self.upsampler(pred[-1]), 
            begin=(None, None, 0, 0), 
            end=(None, None, shape[2], shape[3])) * self.scale, label)
        return epe
        
    def preprocess(self, img):
        #pylint: disable=E1101
        return nd.broadcast_div(nd.broadcast_minus(img, self.color_mean.as_in_context(img.context)),
            self.color_std.as_in_context(img.context))