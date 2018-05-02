import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd
import numpy as np
from mxnet.base import numeric_types
from mxnet import symbol

class DeformableConv2D(nn.HybridBlock):
    """ Deformable Convolution 2D

    Parameters
    ----------
    channels : int
        The dimensionality of the output space
        i.e. the number of output channels in the convolution.
    kernel_size : int or tuple/list of n ints
        Specifies the dimensions of the convolution window.
    strides: int or tuple/list of n ints,
        Specifies the strides of the convolution.
    padding : int or tuple/list of n ints,
        If padding is non-zero, then the input is implicitly zero-padded
        on both sides for padding number of points
    dilation: int or tuple/list of n ints,
        Specifies the dilation rate to use for dilated convolution.
    groups : int
        Controls the connections between inputs and outputs.
        At groups=1, all inputs are convolved to all outputs.
        At groups=2, the operation becomes equivalent to having two convolution
        layers side by side, each seeing half the input channels, and producing
        half the output channels, and both subsequently concatenated.
    layout : str,
        Dimension ordering of data and weight. Can be 'NCW', 'NWC', 'NCHW',
        'NHWC', 'NCDHW', 'NDHWC', etc. 'N', 'C', 'H', 'W', 'D' stands for
        batch, channel, height, width and depth dimensions respectively.
        Convolution is performed over 'D', 'H', and 'W' dimensions.
    in_channels : int, default 0
        The number of input channels to this layer. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    activation : str
        Activation function to use. See :func:`~mxnet.ndarray.Activation`.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias: bool
        Whether the layer uses a bias vector.
    weight_initializer : str or `Initializer`
        Initializer for the `weight` weights matrix.
    bias_initializer: str or `Initializer`
        Initializer for the bias vector.
    """
    def __init__(self, channels, kernel_size, strides=1, padding=0, dilation=1,
                 groups=1, layout='NCHW', num_deformable_group=1, in_channels=0, activation=None, use_bias=True,
                 weight_initializer=None, bias_initializer='zeros', 
                 prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        with self.name_scope():
            self._channels = channels
            self._in_channels = in_channels
            if isinstance(kernel_size, numeric_types):
                kernel_size = (kernel_size,)*2
            if isinstance(strides, numeric_types):
                strides = (strides,)*len(kernel_size)
            if isinstance(padding, numeric_types):
                padding = (padding,)*len(kernel_size)
            if isinstance(dilation, numeric_types):
                dilation = (dilation,)*len(kernel_size)
            self._kwargs = {
                'kernel': kernel_size, 'stride': strides, 'dilate': dilation,
                'pad': padding, 'num_filter': channels, 'num_group': groups,
                'no_bias': not use_bias, 'layout': layout, 
                'num_deformable_group' : num_deformable_group}

            wshapes = [
                (),
                (channels, in_channels) + kernel_size,
                (channels,) 
            ]
            self.weight = self.params.get('weight', shape=wshapes[1],
                                          init=weight_initializer,
                                          allow_deferred_init=True)
            if use_bias:
                self.bias = self.params.get('bias', shape=wshapes[2],
                                            init=bias_initializer,
                                            allow_deferred_init=True)
            else:
                self.bias = None

            if activation is not None:
                self.act = nn.Activation(activation, prefix=activation+'_')
            else:
                self.act = None

    def hybrid_forward(self, F, x, offset, weight, bias=None):
        #pylint: disable=E1101
        if bias is None:
            act = F.contrib.DeformableConvolution(x, offset, weight, name='fwd', **self._kwargs)
        else:
            act = F.contrib.DeformableConvolution(x, offset, weight, bias, name='fwd', **self._kwargs)
        if self.act is not None:
            act = self.act(act)
        return act

    def _alias(self):
        return 'deformable_conv'

    def __repr__(self):
        s = '{name}({mapping}, kernel_size={kernel}, stride={stride}'
        len_kernel_size = len(self._kwargs['kernel'])
        if self._kwargs['pad'] != (0,) * len_kernel_size:
            s += ', padding={pad}'
        if self._kwargs['dilate'] != (1,) * len_kernel_size:
            s += ', dilation={dilate}'
        if self._kwargs['num_group'] != 1:
            s += ', groups={num_group}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        shape = self.weight.shape
        return s.format(name=self.__class__.__name__,
                        mapping='{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0]),
                        **self._kwargs)


class BrunchDeform(nn.HybridBlock):
    def __init__(self, channels, kernel_size, num_deformable_group, split_channels, project=False, **kwargs):
        super().__init__(**kwargs)
        padding = kernel_size // 2
        self.project = project
        with self.name_scope():
            self.split_channels = split_channels
            self.conv = nn.Conv2D(channels, kernel_size, 
                padding=padding, use_bias=True, in_channels=split_channels[0])
            self.offset = nn.Conv2D(num_deformable_group * 2 * kernel_size * kernel_size, 3, 
                padding=1, use_bias=False, weight_initializer='zeros', prefix='offset')
            self.deform_conv = DeformableConv2D(channels, kernel_size, 
                padding=padding, num_deformable_group=num_deformable_group, 
                use_bias=False, in_channels=split_channels[1])
            if self.project:
                net = self.conv_brunch_0 = nn.HybridSequential()
                net.add(nn.Conv2D(split_channels[0], 1))
                net.add(nn.LeakyReLU(0.1))
                net = self.conv_brunch_1 = nn.HybridSequential()
                net.add(nn.Conv2D(split_channels[1], 1))
                net.add(nn.LeakyReLU(0.1))

    def hybrid_forward(self, F, data, control):
        offset = self.offset(control)
        if self.project:
            return self.conv(self.conv_brunch_0(data)) + self.deform_conv(self.conv_brunch_1(data), offset)
        else:
            return (self.conv(F.slice_axis(data, axis=1, begin=None, end=self.split_channels[0]))
            + self.deform_conv(F.slice_axis(data, axis=1, begin=self.split_channels[0], end=None), offset) )

        