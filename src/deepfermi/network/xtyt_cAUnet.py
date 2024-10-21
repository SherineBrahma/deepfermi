from network.layers import *
import numpy as np
from utils import *

class XTYT_cAUnet(nn.Module):

    def __init__(self, ncin=2, ncout=2, nstage=3, nconv_stage=2, nfilters=16, ksize=3, res_connect=False, dropout=0, bias=False):
        super(XTYT_cAUnet, self).__init__()
        
        # Accessing Attention coefficients: aenet.dec_net.cnn.cdec[1].cdec_blk[1].attention

        # General Initializations
        dim = 2
        dsamp_fact = 2
        nfilters_start = nfilters
        self.zshape = None
        self.res_connect = res_connect
        self.m_lambda = 1
        res_conv_block = True

        self.upsamp_shape = []

        # XTYT functions
        self.reshape_op_xyt2xt_yt = XYT2XT_YT()
        self.reshape_op_xt_yt2xyt = XT_YT2XYT()

        # Encoder
        # Initialization
        self.enc = nn.ModuleList()
        # Input Layer
        self.conv_in = ConvBlock(dim=dim, shape=1, nch=ncin, nfilters=nfilters, nconvs=1, pad=0, bias=bias)
        self.enc.append(self.conv_in)
        # Encoding Blocks
        nch = nfilters_start
        nfilters = nfilters_start
        for ns_count in range(nstage):
            self.enc.append(EncoderBlock(dim=dim, nch=nch, nfilters=nfilters, nconvs=nconv_stage, dropout=dropout, stage_indx=ns_count, bias=bias, res_connect = res_conv_block))
            nch = nfilters
            nfilters = dsamp_fact * nch
            
        # Conditioner
        # Initialization
        self.cond = nn.ModuleList()
        cond_nch = nch
        cond_nfilters = nch // dsamp_fact
        # Conditioning Block
        for ns_count in range(nstage - 1):
            self.cond.append(DecoderBlock(dim=dim, nch=cond_nch, nfilters=cond_nfilters, nconvs=nconv_stage, dropout=dropout, bias=bias, res_connect = res_conv_block))
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
            self.cdec.append(CondAGDecoderBlock(dim=dim, nch=nch, nfilters=nfilters, nconvs=nconv_stage, dropout=dropout, bias=bias, res_connect = res_conv_block))
            nch = nch // dsamp_fact
            nfilters = nch // dsamp_fact
        # Output Layer        
        self.cdec.append(ConvLayer(dim=dim, shape=1, nch=nch, nfilters=ncout, pad=0))

    def __call__(self, xin, xcond, type='xt'):
        
        # get the number of samples used; needed for re-assembling operation
        # x has the shape (mb,2,nx,ny,nt)
        mb = xin.shape[0]

        # Spatio-temporal slices
        xin_xt_yt = self.reshape_op_xyt2xt_yt(xin, type)              
        
        # Encoding
        self.skip_out = []
        xenc = xin_xt_yt
        for i in range(0, len(self.enc)):
            if i != 0:
                self.skip_out.append(xenc)
            xenc = self.enc[i](xenc)
        nskip = len(self.skip_out)
        
        # print(xenc.shape)
        # basic_imshow(xenc[50,0,:].cpu().detach(), '/data/brahma01/DCEPerfusion/Experiments/Debug/')
        
        # Conditioners
        j = 0
        self.cond_out = []
        for i in range(len(self.cond)):
            current_layer = self.cond[i]
            self.cond_out.append(xcond)
            xcond = current_layer(xcond, self.skip_out[nskip - j - 1])
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
            xdec = xdec + xin_xt_yt
            
        # Transforming to spatial domain
        xout = self.reshape_op_xt_yt2xyt(xdec, type, mb)

        return xout

class EncoderBlock(nn.Module):

    def __init__(self, dim=2, nch=3, nfilters=3, nconvs=2, dropout=0, stage_indx=1, bias=True, res_connect=False):
        super(EncoderBlock, self).__init__()

        self.enc_blk = nn.ModuleList()
        if stage_indx==0:
            self.enc_blk.append(Identity())
        else:
            self.enc_blk.append(Pooling(dim=dim, pooling_type="Max"))
        self.enc_blk.append(Dropout(dropout=dropout, dim=dim))
        self.enc_blk.append(ConvBlock(dim=dim, shape=3, nch=nch, nfilters=nfilters, nconvs=nconvs, bnorm=False, bias=bias, pad=1, res_connect=res_connect))

    def __call__(self, xin):

        x1 = self.enc_blk[0](xin)
        x2 = self.enc_blk[1](x1)
        x3 = self.enc_blk[2](x2)
        xout = x3

        return xout
    
class DecoderBlock(nn.Module):

    def __init__(self, dim=2, nch=3, nfilters=6, nconvs=2, dropout=0, bias=True, res_connect=False):
        super(DecoderBlock, self).__init__()

        self.dec_blk = nn.ModuleList()
        self.dec_blk.append(Dropout(dropout=dropout, dim=dim))
        self.dec_blk.append(ConvBlock(dim=dim, shape=3, nch=nch, nfilters=nfilters, nconvs=nconvs, bnorm=False, bias=bias, pad=1, res_connect=res_connect))
        self.dec_blk.append(Upsampling(dim=dim))

    def __call__(self, xin, xskip):

        x1 = self.dec_blk[0](xin)
        x2 = self.dec_blk[1](x1)
        x3 = self.dec_blk[2](x2, xskip.shape)
        xout = x3

        return xout
    
class CondAGDecoderBlock(nn.Module):

    def __init__(self, dim=2, nch=3, nfilters=6, nconvs=2, dropout=0, bias=True, res_connect=False):
        super(CondAGDecoderBlock, self).__init__()

        self.cdec_blk = nn.ModuleList()
        self.cdec_blk.append(PixelWiseConditionalInstance2DNorm(nch, cond_disable=True)) # DISABLED CONDITION
        self.cdec_blk.append(Attention(dim=dim, nch=nch, nfilters=nfilters, bias=bias))
        self.cdec_blk.append(Upsampling(dim=dim))
        self.cdec_blk.append(ConvBlock(dim=dim, shape=3, nch=nch, nfilters=nfilters, nconvs=1, bias=bias, pad=1))
        self.cdec_blk.append(Dropout(dropout=dropout, dim=dim))        
        self.cdec_blk.append(ConvBlock(dim=dim, shape=3, nch=nch, nfilters=nfilters, nconvs=nconvs, bnorm=False, bias=bias, pad=1, res_connect=res_connect))

    def __call__(self, xin, xskip, xcond, m_lambda):

        x1 = self.cdec_blk[0](xin, xcond, m_lambda)
        x2 = self.cdec_blk[1](xskip, x1)
        x3 = self.cdec_blk[2](x1, x2.shape)
        x4 = self.cdec_blk[3](x3)
        x5 = torch.cat((x4, x2), 1)
        x6 = self.cdec_blk[4](x5)
        xout = self.cdec_blk[5](x6)

        return xout
    
class Attention(nn.Module):
    """ 
	Attention module that learns to
    focus at the region of interest
	"""
    def __init__(self, dim=2, nch=3, nfilters=6, bias=False):
        super(Attention, self).__init__()

        self.ag_blk = nn.ModuleList()
        self.ag_blk.append(ConvLayer(dim=dim, shape=1, nch=nch, nfilters=nch, pad=0))
        self.ag_blk.append(ConvLayer(dim=dim, shape=1, nch=nfilters, nfilters=nch, pad=0))
        self.ag_blk.append(Pooling(dim=dim, pooling_type="Max"))
        self.ag_blk.append(Activation(act_type='LeakyReLU'))
        self.ag_blk.append(ConvLayer(dim=dim, shape=1, nch=nch, nfilters=1, pad=0))
        self.ag_blk.append(Activation(act_type='Sigmoid'))
        self.ag_blk.append(Upsampling(dim=dim))

    def __call__(self, xin, xgate):
        
        x1 = self.ag_blk[0](xgate)
        x2 = self.ag_blk[1](xin)
        x3 = self.ag_blk[2](x2)
        x4 = self.ag_blk[3](x1 + x3)
        x5 = self.ag_blk[4](x4)
        x6 = self.ag_blk[5](x5)
        self.attention = self.ag_blk[6](x6, xin.shape)
        xout = self.attention*xin

        return xout
    
class XYT2XT_YT(nn.Module):
	""" 
	Class needed for the reshaping operator:
	Given x with shape (mb,2,Nx,Ny,Nt), x is reshped to have
	either shape (mb*Nx,2,Ny,Nt) for the yt-domain or 
	the shape (mb*Ny,2,Nx,Nt) for the xt-domain
	"""
	
	def __init__(self):
		super(XYT2XT_YT, self).__init__()

	def forward(self, x, reshape_type):

		return xyt2xt_yt(x, reshape_type)

class XT_YT2XYT(nn.Module):
	""" 
	Class needed for the reassembling the cine MR image to its original shape:
	reverses the operation XYT2XT_YT,
	note that the mini-batch size is needed
	"""
	
	def __init__(self):
		super(XT_YT2XYT, self).__init__()

	def forward(self, x, reshape_type,mb):
		
		return xt_yt2xyt(x, reshape_type,mb)