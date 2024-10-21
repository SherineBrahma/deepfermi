from network.layers import *
import torch.nn.functional as F
import numpy as np

class EncNet(nn.Module):

    def __init__(self, dim = 2, ncin=2, nstage=3, nconv_stage=2, nfilters=16, ksize=3, res_connect=False, dropout=0, bias=False):
        super(EncNet, self).__init__()

        # General Initializations
        dsamp_fact = 2
        nfilters_start = nfilters
        padding_mode='circular'
        self.zshape = None
        self.res_connect = res_connect
        res_conv_block = True      

        # ENCODER
        # Initialization
        self.enc = nn.ModuleList()
        # Input Layer
        self.enc.append(ConvLayer(dim=dim, shape=1, nch=ncin, nfilters=nfilters, pad=0, padding_mode=padding_mode))
        # Encoding Blocks
        nch = nfilters_start
        nfilters = nfilters_start
        for ns_count in range(nstage):
            self.enc.append(EncoderBlock(dim=dim, nch=nch, nfilters=nfilters, nconvs=nconv_stage, dropout=dropout, stage_indx=ns_count, bias=bias, padding_mode=padding_mode, res_connect = res_conv_block))
            nch = nfilters
            nfilters = dsamp_fact * nch
        self.enc.append(ConvLayer(dim=dim, shape=3, nch=nch, nfilters=nch, pad=1, padding_mode=padding_mode))
        # self.enc.append(MLPConv3DLayer(shape=3, nch=nch, nfilter=nch, pad=1))

    def __call__(self, xin):

        # Encoding a.conv_layer.weight.numel()
        xenc = xin
        for i in range(len(self.enc)):
            xenc = self.enc[i](xenc)
            
        # Encoded latent space information            
        zout = xenc
            
        return zout

class EncoderBlock(nn.Module):

    def __init__(self, dim=2, nch=3, nfilters=3, nconvs=2, dropout=0, stage_indx=1, bias=False, padding_mode='zeros', res_connect=False):
        super(EncoderBlock, self).__init__()

        self.enc_blk = nn.ModuleList()
        if stage_indx==0:
            self.enc_blk.append(Identity())
        else:
            self.enc_blk.append(Pooling(dim=dim, pooling_type="Max"))
        self.enc_blk.append(Dropout(dropout=dropout, dim=dim))
        self.enc_blk.append(ConvBlock(dim=dim, shape=3, nch=nch, nfilters=nfilters, nconvs=nconvs, bnorm=False, bias=bias, pad=1, padding_mode=padding_mode, res_connect=res_connect))

    def __call__(self, xin):

        x1 = self.enc_blk[0](xin)
        x2 = self.enc_blk[1](x1)
        x3 = self.enc_blk[2](x2)
        xout = x3

        return xout
    
class MLPConv3DLayer(nn.Module):

    def __init__(self, shape=3, nch=64, nfilter=64, pad=1):
        super(MLPConv3DLayer, self).__init__()
        self.pad = (pad, pad, pad, pad, pad, pad)
        self.shape = (shape, shape, shape)
        knumel = torch.tensor(self.shape).prod()
        self.wgt_patch = torch.nn.init.xavier_uniform_(nn.Parameter(torch.empty(1,30,30,28,1, dtype=torch.float)))
        self.wgt_in = torch.nn.init.xavier_uniform_(nn.Parameter(torch.empty(nch,knumel,knumel, dtype=torch.float), requires_grad=True))
        self.LeakyReLU = nn.LeakyReLU()
        self.wgt_out = torch.nn.init.xavier_uniform_(nn.Parameter(torch.empty(nch*knumel,nfilter, dtype=torch.float), requires_grad=True))        
        
    def __call__(self, input):
        
        input_pad = F.pad(input, self.pad, mode='circular')
        patches = self.wgt_patch + input_pad.unfold(4, self.shape[2], 1).unfold(3, self.shape[1], 1).unfold(2, self.shape[0], 1).moveaxis(1,-1).flatten(start_dim=-4, end_dim=-1)
        hidden = self.LeakyReLU(patches.matmul(torch.block_diag(*self.wgt_in)))
        output = hidden.matmul(self.wgt_out).moveaxis(-1,1)

        return output
    