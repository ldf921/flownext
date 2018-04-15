from . import hybridnet
from . import flownet

def get_pipeline(network, **kwargs):
    if network == 'hybridnet':
        return hybridnet.Pipeline(**kwargs)
    elif network == 'hybridnet-coarse':
        return hybridnet.PipelineCoarse(**kwargs)
    else:
        raise NotImplementedError