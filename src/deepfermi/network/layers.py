"""
This module defines custom neural network layers, activation functions,
normalization techniques, and building blocks for constructing models
in PyTorch.

Classes include:
- Dense, Identity, ConvLayer, TransposeConv2DLayer, and Activation functions
  (ReLU, LeakyReLU, Sigmoid, etc.).
- Dropout, Pooling, Upsampling, and normalization layers (BatchNorm, LayerNorm,
  InstanceNorm).
- Conditional layers (e.g., ConditionalInstance2DNorm,
  PixelWiseConditionalInstanceNorm).
- Utility layers like Flattening and ConstantLayer, as well as the ConvBlock
  for stacking multiple convolutional operations.

These layers can be combined and customized to build flexible neural network
architectures.
"""

import torch
import torch.nn as nn


class Dense(nn.Module):

    def __init__(self, in_feature, out_feature):
        super(Dense, self).__init__()

        self.dense = nn.Linear(in_feature, out_feature, bias=False)

    def __call__(self, x_dense):
        y_dense = self.dense(x_dense)

        return y_dense


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

        self.identity = nn.Identity()

    def __call__(self, x_identity):
        y_identity = self.identity(x_identity)

        return y_identity


class ConvLayer(nn.Module):

    def __init__(self, shape, nch, nfilters, pad=1, dim=2, **kwargs):
        super(ConvLayer, self).__init__()
        if kwargs.get('stride') is not None:
            stride = kwargs.get('stride')
        else:
            stride = 1
        if kwargs.get('bias') is not None:
            bias = kwargs.get('bias')
        else:
            bias = False
        if kwargs.get('groups') is not None:
            groups = kwargs.get('groups')
        else:
            groups = 1
        if kwargs.get('padding_mode') is not None:
            padding_mode = kwargs.get('padding_mode')
        else:
            padding_mode = 'zeros'

        if dim == 1:
            self.conv_layer = nn.Conv1d(nch,
                                        nfilters,
                                        shape,
                                        stride=stride,
                                        padding=pad,
                                        bias=bias,
                                        groups=groups,
                                        padding_mode=padding_mode)
        elif dim == 2:
            self.conv_layer = nn.Conv2d(nch,
                                        nfilters,
                                        shape,
                                        stride=stride,
                                        padding=pad,
                                        bias=bias,
                                        groups=groups,
                                        padding_mode=padding_mode)
        elif dim == 3:
            self.conv_layer = nn.Conv3d(nch,
                                        nfilters,
                                        shape,
                                        stride=stride,
                                        padding=pad,
                                        bias=bias,
                                        groups=groups,
                                        padding_mode=padding_mode)

    def __call__(self, x_conv):

        y_conv = self.conv_layer(x_conv)

        return y_conv


class TransposeConv2DLayer(nn.Module):

    def __init__(self, shape, nch, nout, stride, pad=1):
        super(TransposeConv2DLayer, self).__init__()

        self.tran_conv_layer = nn.ConvTranspose2d(nch,
                                                  nout,
                                                  shape,
                                                  stride=stride,
                                                  padding=pad,
                                                  output_padding=[1, 0],
                                                  bias=False)

    def __call__(self, x_tran_conv):
        y_tran_conv = self.tran_conv_layer(x_tran_conv)

        return y_tran_conv


class Activation(nn.Module):

    def __init__(self, **kwargs):
        super(Activation, self).__init__()

        if kwargs.get('act_type') is None:
            act_type = 'LeakyReLU'
            kwargs['neg_slope'] = 0.01
            kwargs['inplace'] = True
        else:
            act_type = kwargs['act_type']

        if kwargs.get('inplace') is None:
            kwargs['inplace'] = True

        if act_type == 'LeakyReLU':
            if kwargs.get('neg_slope') is None:
                kwargs['neg_slope'] = 0.01
            if kwargs.get('inplace') is None:
                kwargs['inplace'] = True
            neg_slope = kwargs['neg_slope']
            inplace_boolean = kwargs['inplace']
            self.act = nn.LeakyReLU(neg_slope, inplace=inplace_boolean)

        if act_type == 'ReLU':
            inplace_boolean = kwargs['inplace']
            self.act = nn.ReLU(inplace=inplace_boolean)

        if act_type == 'Softplus':
            self.act = torch.nn.Softplus()

        if act_type == 'Softmax':
            if kwargs.get('dim') is None:
                kwargs['dim'] = None
            dim = kwargs['dim']
            self.act = torch.nn.Softmax(dim=dim)

        if act_type == 'Sigmoid':
            self.act = torch.nn.Sigmoid()

        if act_type == 'Tanh':
            self.act = torch.nn.Tanh()

    def __call__(self, x_act):
        y_act = self.act(x_act)

        return y_act


class Dropout(nn.Module):

    def __init__(self, dim=2, **kwargs):
        super(Dropout, self).__init__()

        if kwargs.get('dropout') is None:
            kwargs['dropout'] = 1
        if kwargs.get('inplace') is None:
            kwargs['inplace'] = False
        dropout = kwargs['dropout']
        inplace_boolean = kwargs['inplace']
        if dim == 1:
            self.dropout = nn.Dropout(p=dropout, inplace=inplace_boolean)
        elif dim == 2:
            self.dropout = nn.Dropout2d(p=dropout, inplace=inplace_boolean)
        elif dim == 3:
            self.dropout = nn.Dropout3d(p=dropout, inplace=inplace_boolean)

    def __call__(self, x_drop):

        y_drop = self.dropout(x_drop)

        return y_drop


class Pooling(nn.Module):

    def __init__(self, dim=2, pooling_type='Max', **kwargs):
        super(Pooling, self).__init__()

        if kwargs.get('kernel_size') is None:
            kwargs['kernel_size'] = 2

        if pooling_type == 'Max':
            if dim == 1:
                self.pool = nn.MaxPool1d(kwargs['kernel_size'])
            elif dim == 2:
                self.pool = nn.MaxPool2d(kwargs['kernel_size'])
            elif dim == 3:
                self.pool = nn.MaxPool3d(kwargs['kernel_size'])

        if pooling_type == 'Avg':
            if dim == 1:
                self.pool = nn.AvgPool1d(kwargs['kernel_size'])
            elif dim == 2:
                self.pool = nn.AvgPool2d(kwargs['kernel_size'])
            elif dim == 3:
                self.pool = nn.AvgPool3d(kwargs['kernel_size'])

    def __call__(self, x_pool):

        y_pool = self.pool(x_pool)

        return y_pool


class Upsampling(nn.Module):

    def __init__(self, dim=2, **kwargs):
        super(Upsampling, self).__init__()

        if dim == 1:
            self.upsamp_type = 'linear'
        elif dim == 2:
            self.upsamp_type = 'bilinear'
        elif dim == 3:
            self.upsamp_type = 'trilinear'

    def __call__(self, x_usmpl, size):

        y_usmpl = nn.functional.interpolate(x_usmpl,
                                            size=size[2:],
                                            mode=self.upsamp_type,
                                            align_corners=False)

        return y_usmpl


class BatchNorm(nn.Module):

    def __init__(self, nch, dim=2, trainable=False):
        super(BatchNorm, self).__init__()

        if dim == 1:
            self.batch_norm = nn.BatchNorm1d(nch,
                                             affine=trainable,
                                             track_running_stats=False)
        elif dim == 2:
            self.batch_norm = nn.BatchNorm2d(nch,
                                             affine=trainable,
                                             track_running_stats=False)
        elif dim == 3:
            self.batch_norm = nn.BatchNorm3d(nch,
                                             affine=trainable,
                                             track_running_stats=False)

    def __call__(self, x_bn):
        y_bn = self.batch_norm(x_bn)
        return y_bn


class LayerNorm2D(nn.Module):

    def __init__(self, trainable=False):
        super(LayerNorm2D, self).__init__()
        self.trainable = trainable

    def __call__(self, x_ln):
        inp_shape = x_ln.shape[1:]
        if not hasattr(self, 'layer_norm'):
            self.layer_norm = nn.LayerNorm(inp_shape,
                                           elementwise_affine=self.trainable)
        y_ln = self.layer_norm(x_ln)
        return y_ln


class InstanceNorm(nn.Module):

    def __init__(self, num_features, dim=2, affine=False):
        super(InstanceNorm, self).__init__()
        if dim == 2:
            self.instance_norm = torch.nn.InstanceNorm2d(num_features,
                                                         affine=affine)
        elif dim == 3:
            self.instance_norm = torch.nn.InstanceNorm3d(num_features,
                                                         affine=affine)        

    def __call__(self, x_in):
        y_in = self.instance_norm(x_in)
        return y_in


class ConditionalInstance2DNorm(nn.Module):

    def __init__(self, num_channels, dim=2, eps=1e-5, affine=False):
        super(ConditionalInstance2DNorm, self).__init__()
        self.instance_norm = InstanceNorm(num_channels, dim=dim, affine=affine)

    def forward(self, input, condition):
        weight = condition[:, :, 0]
        bias = condition[:, :, 1]
        normalized = self.instance_norm(input)
        weight = weight.unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)
        transformed = weight * normalized + bias
        return transformed


class ConstantLayer(nn.Module):

    def __init__(self, constant):
        super(ConstantLayer, self).__init__()
        self.constant = constant

    def forward(self, input):
        output = self.constant + 0*input
        return output


class PixelWiseConditionalInstanceNorm(nn.Module):

    def __init__(self,
                 num_channels,
                 dim=2,
                 affine=False,
                 cond_disable=False,
                 res_connect=False):
        super(PixelWiseConditionalInstanceNorm, self).__init__()
        if cond_disable is False:
            self.instance_norm_wgt = InstanceNorm(num_channels,
                                                  dim=dim,
                                                  affine=affine)
        else:
            self.instance_norm_wgt = ConstantLayer(1)
        self.instance_norm_inp = InstanceNorm(num_channels,
                                              dim=dim,
                                              affine=affine)
        self.res_connect = res_connect

    def forward(self, xin, condition, mfact=1):

        weight = mfact*self.instance_norm_wgt(condition)
        bias = 0
        normalized = self.instance_norm_inp(xin)
        transformed = weight * normalized + bias
        # Residual Connection
        if self.res_connect:
            transformed = transformed + normalized
        xout = transformed

        return xout


class Flattening(nn.Module):

    def __init__(self, start_dim=1):
        super(Flattening, self).__init__()
        torch_ver = str(torch.__version__)
        torch_ver_splt = torch_ver.split('.')
        cond = ((torch_ver_splt[0] >= '1')
                and (torch_ver_splt[1] >= '1')
                and (torch_ver_splt[2] >= '6'))
        if cond:
            self.flatten = nn.Flatten(start_dim=start_dim)
        else:
            self.flatten = self.flatten_old_py(start_dim=start_dim)

    def flatten_old_py(self, start_dim=1):
        def flatten(flt_inp):
            out_flt = torch.flatten(flt_inp, start_dim=start_dim)
            return out_flt

        return flatten

    def __call__(self, x_flat):

        y_flat = self.flatten(x_flat)

        return y_flat


class ConvBlock(nn.Module):

    def __init__(self,
                 shape,
                 nch,
                 nfilters,
                 nconvs,
                 dim=2,
                 ncout=None,
                 pad=1,
                 bnorm=False,
                 res_connect=False,
                 **kwargs):
        super(ConvBlock, self).__init__()

        if not ncout:
            ncout = nfilters

        self.res_connect = res_connect
        if self.res_connect:
            self.res_conv = ConvLayer(dim=dim,
                                      shape=1,
                                      nch=nch,
                                      nfilters=ncout,
                                      pad=0,
                                      **kwargs)

        self.conv_block = []
        for _ in range(nconvs - 1):
            self.conv_block.append(ConvLayer(shape,
                                             nch,
                                             nfilters,
                                             dim=dim,
                                             pad=pad,
                                             **kwargs))
            self.conv_block.append(Activation(**kwargs))
            if bnorm is True:
                self.conv_block.append(BatchNorm(nfilters, dim=dim))
            nch = nfilters
        self.conv_block.append(ConvLayer(shape,
                                         nch,
                                         ncout,
                                         dim=dim,
                                         pad=pad,
                                         **kwargs))
        self.conv_block.append(Activation(**kwargs))
        if bnorm is True:
            self.conv_block.append(BatchNorm(ncout, dim=dim))
        self.conv_block = nn.Sequential(*self.conv_block)

    def __call__(self, xin):

        # Convolution block operation
        xconv = xin
        for i in range(len(self.conv_block)):
            xconv = self.conv_block[i](xconv)

        # Residual Connection
        if self.res_connect:
            xres = self.res_conv(xin)
            xconv = xconv + xres           
        xout = xconv

        return xout
