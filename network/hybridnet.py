from .flownet import EpeLoss, Upsample, Downsample

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.model_zoo.vision import vgg16

class FlowPrediction:
    

class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.backbone = vgg16()
                

    def hybrid_forward(self, F, img1, img2):
        
        