from . import hybridnet
from . import flownet

def get_pipeline(network, **kwargs):
    if network == 'hybridnet':
        return hybridnet.Pipeline(**kwargs)
    else:
        raise NotImplementedError