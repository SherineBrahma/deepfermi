import torch
from torch import nn
from network.layers import (ConvBlock,
                            ConvLayer,
                            Dropout,
                            Identity,
                            Pooling,
                            Upsampling)


class Unet(nn.Module):
    """
    A U-Net architecture with an encoder-decoder structure. It consists
    an encoder, which progressively reduces the spatial dimensions while
    extracting features, followed by a decoder that upsamples the feature
    maps. Skip connections are used between the encoder and decoder to
    preserve spatial information.

    Parameters:
        dim (int): The dimension of the input data (1, 2, or 3).
        ncin (int): The number of input channels.
        ncout (int): The number of output channels.
        nstage (int): The number of stages in the encoder-decoder architecture.
        nconv_stage (int): The number of convolutional layers in each stage.
        nfilters (int): The number of filters in the first layer.
        res_connect (bool): If True, adds residual connections.
        dropout (float): Dropout rate for regularization.
        bias (bool): If True, adds bias to convolution layers.
    """

    def __init__(self,
                 dim=2,
                 ncin=2,
                 ncout=2,
                 nstage=3,
                 nconv_stage=2,
                 nfilters=16,
                 res_connect=False,
                 dropout=0,
                 bias=False):
        super(Unet, self).__init__()

        # General Initializations
        dsamp_fact = 2
        nfilters_start = nfilters
        self.zshape = None
        self.res_connect = res_connect
        self.m_lambda = 1
        res_conv_block = False
        padding_mode = 'circular'

        # Encoder
        # Initialization
        self.enc = nn.ModuleList()
        # Input Layer
        self.conv_in = ConvLayer(dim=dim,
                                 shape=1,
                                 nch=ncin,
                                 nfilters=nfilters,
                                 pad=0,
                                 bias=bias,
                                 padding_mode=padding_mode)
        self.enc.append(self.conv_in)
        # Encoding Blocks
        nch = nfilters_start
        nfilters = nfilters_start
        for ns_count in range(nstage):
            self.enc.append(EncoderBlock(dim=dim,
                                         nch=nch,
                                         nfilters=nfilters,
                                         nconvs=nconv_stage,
                                         dropout=dropout,
                                         stage_indx=ns_count,
                                         bias=bias,
                                         res_connect=res_conv_block,
                                         padding_mode=padding_mode))
            nch = nfilters
            nfilters = dsamp_fact * nch

        # Decoder
        # Initialization
        self.dec = nn.ModuleList()
        # Decoding Block
        nch = nfilters // dsamp_fact
        nfilters = nch // dsamp_fact
        for ns_count in range(nstage - 1):
            self.dec.append(DecoderBlock(dim=dim,
                                         nch=nch,
                                         nfilters=nfilters,
                                         nconvs=nconv_stage,
                                         dropout=dropout,
                                         bias=bias,
                                         res_connect=res_conv_block,
                                         padding_mode=padding_mode))
            nch = nch // dsamp_fact
            nfilters = nch // dsamp_fact
        # Output Layer
        self.dec.append(ConvLayer(dim=dim,
                                  shape=1,
                                  nch=nch,
                                  nfilters=ncout,
                                  pad=0))

    def __call__(self, xin):

        # Encoding
        self.skip_out = []
        xenc = xin
        for i in range(0, len(self.enc)):
            if i != 0:
                self.skip_out.append(xenc)
            xenc = self.enc[i](xenc)
        nskip = len(self.skip_out)

        # Decoding
        j = 0
        xdec = xenc
        for i in range(len(self.dec) - 1):
            xdec = self.dec[i](xdec, self.skip_out[nskip - j - 1])
            j = j + 1            
        for i in range(j, len(self.dec)):
            xdec = self.dec[i](xdec)

        # Residual Connection
        if self.res_connect:
            xdec = xdec + xin
        xout = xdec

        return xout


class EncoderBlock(nn.Module):
    """
    A block of the encoder in the U-Net architecture.

    This block consists of a pooling layer, dropout, and a convolutional
    block to progressively reduce the spatial dimensions while learning
    features.

    Parameters:
        dim (int): The dimension of the input data (1, 2, or 3).
        nch (int): The number of input channels.
        nfilters (int): The number of filters in each convolutional layer.
        nconvs (int): The number of convolutional layers.
        dropout (float): Dropout rate for regularization.
        stage_indx (int): The index of the stage in the U-Net.
        bias (bool): If True, adds bias to convolution layers.
        res_connect (bool): If True, adds residual connections.
        padding_mode (str): The padding mode for convolutions (e.g., 'zeros').
    """

    def __init__(self,
                 dim=2,
                 nch=3,
                 nfilters=3,
                 nconvs=2,
                 dropout=0,
                 stage_indx=1,
                 bias=True,
                 res_connect=False,
                 padding_mode='zeros'):
        super(EncoderBlock, self).__init__()

        self.enc_blk = nn.ModuleList()
        if stage_indx == 0:
            self.enc_blk.append(Identity())
        else:
            self.enc_blk.append(Pooling(dim=dim, pooling_type="Max"))
        self.enc_blk.append(Dropout(dropout=dropout, dim=dim))
        self.enc_blk.append(ConvBlock(dim=dim,
                                      shape=3,
                                      nch=nch,
                                      nfilters=nfilters,
                                      nconvs=nconvs,
                                      bnorm=False,
                                      bias=bias,
                                      res_connect=res_connect,
                                      padding_mode=padding_mode))

    def __call__(self, xin):

        x1 = self.enc_blk[0](xin)
        x2 = self.enc_blk[1](x1)
        x3 = self.enc_blk[2](x2)
        xout = x3

        return xout


class DecoderBlock(nn.Module):
    """
    A block of the decoder in the U-Net architecture.

    This block consists of a dropout layer, upsampling layer, and a
    convolutional block to progressively increase the spatial dimensions of
    the feature maps.

    Parameters:
        dim (int): The dimension of the input data (1, 2, or 3).
        nch (int): The number of input channels.
        nfilters (int): The number of filters in each convolutional layer.
        nconvs (int): The number of convolutional layers.
        dropout (float): Dropout rate for regularization.
        bias (bool): If True, adds bias to convolution layers.
        res_connect (bool): If True, adds residual connections.
        padding_mode (str): The padding mode for convolutions (e.g., 'zeros').
    """

    def __init__(self,
                 dim=2,
                 nch=3,
                 nfilters=6,
                 nconvs=2,
                 dropout=0,
                 bias=True,
                 res_connect=False,
                 padding_mode='zeros'):
        super(DecoderBlock, self).__init__()

        self.dec_blk = nn.ModuleList()
        self.dec_blk.append(Dropout(dropout=dropout, dim=dim))
        self.dec_blk.append(Upsampling(dim=dim))
        self.dec_blk.append(ConvBlock(dim=dim,
                                      shape=3,
                                      nch=nch+(nch//2),
                                      nfilters=nfilters,
                                      nconvs=nconvs,
                                      bnorm=False,
                                      bias=bias,
                                      res_connect=res_connect,
                                      padding_mode=padding_mode))

    def __call__(self, xin, xskip):

        x1 = self.dec_blk[0](xin)
        x2 = self.dec_blk[1](x1, xskip.shape)
        x3 = self.dec_blk[2](torch.cat((x2, xskip), 1))
        xout = x3

        return xout
