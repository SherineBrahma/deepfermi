import torch
import numpy as np
from network.layers import *

# def super_wrap(a):
#     print(a)
#     def wrap(func):
#         """
#         Wraps *func* with additional code.
#         """
#         # we define a wrapper function. This will execute all additional code
#         # before and after the "real" function.
#         def wrapped(*args, **kwargs):
#             print("before-call")
#             output = func(*args, **kwargs)
#             print("after-call")
#             return output
#         # Use "update_wrapper" to keep docstrings and other function metadata
#         # intact
#         update_wrapper(wrapped, func)

#         # We can now return the wrapped function
#         return wrapped
#     return wrap

# class super_wrap():
#     def __init__(self):
#         self.wrap=wrap

class WGAN_ADVNet(nn.Module):
    
    def __init__(self, img_dim, op_dim=2, ncin=2, nstage=3, nconv_stage=2, nfilters=16, dropout=0, bias=False): # , iDS_config=None
        super(WGAN_ADVNet, self).__init__()

        # General Initializations
        dsamp_fact = 2
        padding_mode='circular'        

        # ENCODER
        # Initialization
        self.enc = nn.ModuleList()        
        # Input Layer
        nfilters_start = nfilters
        self.conv_in = ConvLayer(dim=op_dim, shape=1, nch=ncin, nfilters=nfilters, pad=0, bias=bias, padding_mode=padding_mode)
        self.enc.append(self.conv_in)
        # Encoding Blocks
        nch = nfilters_start
        nfilters = nfilters_start
        for ns_count in range(nstage):
            self.enc.append(EncoderBlock(dim=op_dim, nch=nch, nfilters=nfilters, nconvs=nconv_stage, dropout=dropout, stage_indx=ns_count, bias=bias, padding_mode=padding_mode))
            nch = nfilters
            nfilters = nch//dsamp_fact

        # CLASSIFIER
        # Initialization
        self.cls = nn.ModuleList()
        # Input Layer
        flatten = Flattening()
        self.cls.append(flatten)
        # Classification Block
        if op_dim==2:
            cin = (nfilters_start// 2**(nstage-1)) * (img_dim[0]// 2**(nstage-1)) * (img_dim[1] // 2**(nstage-1))
        elif op_dim==3:
            cin = (nfilters_start// 2**(nstage-1)) * (img_dim[0]// 2**(nstage-1)) * (img_dim[1] // 2**(nstage-1)) * (img_dim[2] // 2**(nstage-1))
        cout = cin // dsamp_fact
        for ns_count in range(nstage):
            self.cls.append(Dense(cin, cout))
            self.cls.append(Activation(act_type="ReLU"))
            cin = cout
            cout = np.maximum(cin // dsamp_fact, 1)
        # Output Layer
        self.dense_out = Dense(cin, 1)
        self.cls.append(self.dense_out)
        
        # # Interface data-structure
        # # Input
        # if iDS_config.in_ds is not None:
        #     self.iDS_in = iDS_config.in_ds
        # else:
        #     self.iDS_in = Identity()
        
    def __call__(self, xin):      

        # Encoding
        xenc = xin # self.iDS_in(xin)
        for i in range(len(self.enc)):
            xenc = self.enc[i](xenc)        

        # Classification
        xcls = xenc
        for i in range(len(self.cls)):
            xcls = self.cls[i](xcls)
        
        # Output
        xadv = xcls

        return xadv

class EncoderBlock(nn.Module):

    def __init__(self, dim=2, nch=3, nfilters=3, nconvs=2, dropout=0, stage_indx=1, bias=True, padding_mode='zeros'):
        super(EncoderBlock, self).__init__()

        self.enc_blk = nn.ModuleList()
        if stage_indx==0:
            self.enc_blk.append(Identity())
        else:
            self.enc_blk.append(Pooling(dim=dim, pooling_type="Max"))
        self.enc_blk.append(Dropout(dropout=dropout, dim=dim))
        self.enc_blk.append(ConvBlock(dim=dim, shape=3, nch=nch, nfilters=nfilters, nconvs=nconvs, bnorm=False, bias=bias, padding_mode=padding_mode))

    def __call__(self, xin):

        x1 = self.enc_blk[0](xin)
        x2 = self.enc_blk[1](x1)
        x3 = self.enc_blk[2](x2)
        xout = x3

        return xout