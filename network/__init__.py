from . import layer
from . import flownet
from . import hybridnet
from . import pipeline

def get_pipeline(network, **kwargs):
    if network == 'flownet':
        return pipeline.PipelineFlownet(**kwargs)
    elif network == 'hybridnet-coarse':
        return hybridnet.PipelineCoarse(**kwargs)
    else:
        raise NotImplementedError