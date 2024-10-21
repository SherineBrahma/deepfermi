from network.layers import *
import numpy as np
from utils import *

class cUnet(nn.Module):

    def __init__(self, dim = 2, ncin=2, ncout=2, nstage=3, nconv_stage=2, nfilters=16, ksize=3, res_connect=False, dropout=0, bias=False, cond_disable=False):
        super(cUnet, self).__init__()

        # General Initializations
        dsamp_fact = 2
        nfilters_start = nfilters
        self.zshape = None
        self.res_connect = res_connect
        self.m_lambda = 1
        res_conv_block = False
        padding_mode='circular'

        # Encoder
        # Initialization
        self.enc = nn.ModuleList()
        # Input Layer 
        self.conv_in = ConvLayer(dim=dim, shape=1, nch=ncin, nfilters=nfilters, pad=0, bias=bias, padding_mode=padding_mode)
        self.enc.append(self.conv_in)
        # Encoding Blocks
        nch = nfilters_start
        nfilters = nfilters_start
        for ns_count in range(nstage):
            self.enc.append(EncoderBlock(dim=dim, nch=nch, nfilters=nfilters, nconvs=nconv_stage, dropout=dropout, stage_indx=ns_count, bias=bias, res_connect = res_conv_block, padding_mode=padding_mode))
            nch = nfilters
            nfilters = dsamp_fact * nch
            
        # Conditioner
        # Initialization
        self.cond = nn.ModuleList()
        cond_nch = nch
        cond_nfilters = nch
        # Conditioning Block
        for ns_count in range(nstage - 1):
            self.cond.append(DecoderBlock(dim=dim, nch=cond_nch, nfilters=cond_nfilters, nconvs=nconv_stage, dropout=dropout, bias=bias, res_connect = res_conv_block, padding_mode=padding_mode))
            cond_nch = cond_nfilters
            cond_nfilters = cond_nch // dsamp_fact
            
        # Decoder
        # Initialization
        cdec_nfilters = nfilters // dsamp_fact
        cdec_nch = nch // dsamp_fact        
        self.cdec = nn.ModuleList()
        nch = cdec_nfilters
        nfilters = cdec_nch
        # Decoding Block
        for ns_count in range(nstage - 1):
            self.cdec.append(CondDecoderBlock(dim=dim, nch=nch, nfilters=nfilters, nconvs=nconv_stage, dropout=dropout, bias=bias, res_connect = res_conv_block, padding_mode=padding_mode, cond_disable=cond_disable))
            nch = nch // dsamp_fact
            nfilters = nch // dsamp_fact
        # Output Layer        
        self.cdec.append(ConvLayer(dim=dim, shape=1, nch=nch, nfilters=ncout, pad=0))

    def __call__(self, xin, xcond):            
        
        # Encoding
        self.skip_out = []
        xenc = xin
        for i in range(0, len(self.enc)):
            if i != 0:
                self.skip_out.append(xenc)
            xenc = self.enc[i](xenc)
        nskip = len(self.skip_out)
        
        # Conditioners
        j = 0
        self.cond_out = []
        for i in range(len(self.cond)):
            current_layer = self.cond[i]
            xcond = current_layer(xcond, self.skip_out[nskip - j - 1])
            self.cond_out.append(xcond)
            j = j + 1

        # Decoding
        j = 0
        xdec = xenc
        for i in range(len(self.cdec) - 1):
            xdec = self.cdec[i](xdec, self.skip_out[nskip - j - 1], self.cond_out[i], self.m_lambda)
            j = j + 1            
        for i in range(j, len(self.cdec)):
            xdec = self.cdec[i](xdec)

        # Residual Connection
        if self.res_connect:
            xdec = xdec + xin
        xout = xdec

        return xout

class EncoderBlock(nn.Module):

    def __init__(self, dim=2, nch=3, nfilters=3, nconvs=2, dropout=0, stage_indx=1, bias=True, res_connect=False, padding_mode='zeros'):
        super(EncoderBlock, self).__init__()

        self.enc_blk = nn.ModuleList()
        if stage_indx==0:
            self.enc_blk.append(Identity())
        else:
            self.enc_blk.append(Pooling(dim=dim, pooling_type="Max"))
        self.enc_blk.append(Dropout(dropout=dropout, dim=dim))
        self.enc_blk.append(ConvBlock(dim=dim, shape=3, nch=nch, nfilters=nfilters, nconvs=nconvs, bnorm=False, bias=bias, res_connect=res_connect, padding_mode=padding_mode))

    def __call__(self, xin):

        x1 = self.enc_blk[0](xin)
        x2 = self.enc_blk[1](x1)
        x3 = self.enc_blk[2](x2)
        xout = x3

        return xout
    
class DecoderBlock(nn.Module):

    def __init__(self, dim=2, nch=3, nfilters=6, nconvs=2, dropout=0, bias=True, res_connect=False, padding_mode='zeros'):
        super(DecoderBlock, self).__init__()

        self.dec_blk = nn.ModuleList()
        self.dec_blk.append(Dropout(dropout=dropout, dim=dim))
        self.dec_blk.append(ConvBlock(dim=dim, shape=3, nch=nch, nfilters=nfilters, nconvs=nconvs, bnorm=False, bias=bias, res_connect=res_connect, padding_mode=padding_mode))
        self.dec_blk.append(Upsampling(dim=dim))

    def __call__(self, xin, xskip):

        x1 = self.dec_blk[0](xin)
        x2 = self.dec_blk[1](x1)
        x3 = self.dec_blk[2](x2, xskip.shape)
        xout = x3

        return xout

class CondDecoderBlock(nn.Module):

    def __init__(self, dim=2, nch=3, nfilters=6, nconvs=2, dropout=0, bias=True, res_connect=False, padding_mode='zeros', cond_disable=False):
        super(CondDecoderBlock, self).__init__()

        self.cdec_blk = nn.ModuleList()
        self.cdec_blk.append(Upsampling(dim=dim))
        self.cdec_blk.append(ConvBlock(dim=dim, shape=3, nch=nch, nfilters=nfilters, nconvs=1, bias=bias, padding_mode=padding_mode))
        self.cdec_blk.append(Dropout(dropout=dropout, dim=dim))
        self.cdec_blk.append(PixelWiseConditionalInstanceNorm(nch, dim=dim, cond_disable=cond_disable, res_connect=True))
        self.cdec_blk.append(ConvBlock(dim=dim, shape=3, nch=nch, nfilters=nfilters, nconvs=nconvs, bnorm=False, bias=bias, res_connect=res_connect, padding_mode=padding_mode))

    def __call__(self, xin, xskip, xcond, m_lambda):

        x1 = self.cdec_blk[0](xin, xskip.shape)
        x2 = self.cdec_blk[1](x1)
        x3 = torch.cat((x2, xskip), 1)
        x4 = self.cdec_blk[2](x3)
        x5 = self.cdec_blk[3](x4, xcond, m_lambda)
        x6 = self.cdec_blk[4](x5)
        xout = x6

        return xout