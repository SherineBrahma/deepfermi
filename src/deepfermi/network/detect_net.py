import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from fermi import *
from network.layers import *
from utils import *
from torchvision import transforms
from network.layers import *
from torch.autograd.functional import hvp

class DetectNet(nn.Module):
    
    def __init__(self, cnn, eval=False):
        
        """
		A CNN which consists of alternating blocks of CNNs and CGs; 
		- the forward model is the NUFFT operator;
		- CNN_M estimates the mean. It can be any CNN suitable for processing a a 2D cine MR image.
		- CNN_AV estimates the aleatoric variance. It can be any CNN suitable for processing a a 2D cine MR image.
		
		Parameters:
			
			- EncObj: 	- the econding operator with the forward E , adjoint E^h and composite operator H:= E^H \circ E
			- CNN: 	 	- the CNN-block to be used
			- nu: 	 	- the number of alternations between CG- and CNN modules
			- npcg: 	- te number of  CG iterations in the CG-module;
			- mode: 	- defines whether we a re in the pretraining (only CNN) or inthe fine-tuning mode (CNN+CG) ;
			- use_precon: - defines whether to pre-condition the problem by using the density compensation function
			- lambda_reg - the regularization parameter; if None, it is learned during training;
								
		"""        
        
        super(DetectNet, self).__init__()
  
        self.cnn = cnn
        self.sigmoid = torch.nn.Sigmoid()
        self.eval = eval
  
    def forward(self, xin):
        
        # Applying cnn
        xdetect = self.cnn(xin)
        
        # Classification layer
        xout = self.sigmoid(xdetect)
        
        if self.eval==True:
            xout = (xout>0.2).float()
        
        return xout
    
class MLPConv1DBlock(nn.Module):

    def __init__(self, shape, nch, nfilters, nconvs, ncout=None, pad=1, bnorm=False, res_connect=False, **kwargs):
        super(MLPConv1DBlock, self).__init__()

        if not ncout:
            ncout = nfilters
            
        self.res_connect = res_connect
        if self.res_connect:
            self.res_conv = ConvLayer(dim=1, shape=1, nch=nch, nfilters=ncout, pad=0, **kwargs)
        self.mlp_conv3D_block = []
        for i in range(nconvs - 1):
            self.mlp_conv3D_block.append(MLPConv1DLayer(shape=shape, nch=nch, nfilter=nfilters, pad=pad))
            if bnorm == True:
                self.mlp_conv3D_block.append(BatchNorm(nfilters, dim=3))
            nch = nfilters
        self.mlp_conv3D_block.append(MLPConv1DLayer(shape=shape, nch=nch, nfilter=ncout, pad=pad))
        if bnorm == True:
            self.mlp_conv3D_block.append(BatchNorm(ncout, dim=3))
        self.mlp_conv3D_block = nn.Sequential(*self.mlp_conv3D_block)

    def __call__(self, xin):
        
        # Convolution block operation
        xconv = xin
        for i in range(len(self.mlp_conv3D_block)):
            xconv = self.mlp_conv3D_block[i](xconv)            
        # Residual Connection
        if self.res_connect:
            xres = self.res_conv(xin)
            xconv = xconv + xres            
        xout = xconv

        return xout
    
class MLPConv1DLayer(nn.Module):

    def __init__(self, shape=3, nch=64, nfilter=64, pad=1, padding_mode='circular'):
        super(MLPConv1DLayer, self).__init__()
        self.pad = (pad, pad)
        self.padding_mode = padding_mode
        self.shape = (shape,)
        knumel = torch.tensor(self.shape).prod()
        self.LeakyReLU = nn.LeakyReLU()
        self.wgt_in = torch.nn.init.kaiming_uniform_(nn.Parameter(torch.empty(nch,knumel,knumel, dtype=torch.float), requires_grad=True))
        self.wgt_h1 = torch.nn.init.kaiming_uniform_(nn.Parameter(torch.empty(nch*knumel, nch*4, dtype=torch.float), requires_grad=True))
        self.wgt_out = torch.nn.init.kaiming_uniform_(nn.Parameter(torch.empty(nch*4,nfilter, dtype=torch.float), requires_grad=True))    
        
    def __call__(self, input):
        
        input_pad = F.pad(input, self.pad, mode=self.padding_mode)
        patches = input_pad.unfold(2, self.shape[0], 1).moveaxis(1,-1).flatten(start_dim=-2, end_dim=-1)
        h1 = self.LeakyReLU(patches.matmul(torch.block_diag(*self.wgt_in)))
        h2 = self.LeakyReLU(h1.matmul(self.wgt_h1)) 
        output = h2.matmul(self.wgt_out).moveaxis(-1,1)

        return output