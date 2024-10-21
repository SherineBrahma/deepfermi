import torch
import numpy as np
from network.layers import *
from utils import *
    

class Translate_iDS(nn.Module):
    
    def __init__(self, dim=3, op_dim=2, in_ds=None, out_ds=None, slice_dir=None):
        super(Translate_iDS, self).__init__()
        
        # 2D spatio-temporal 3D input data processing
        if dim==3 and op_dim==2 and in_ds=='2D' and (slice_dir=='xt' or slice_dir=='yt' or slice_dir=='xtyt'):
            self.in_ds = xyt2xt_yt

        # 2D spatial 3D input data processing
        if dim==3 and op_dim==2 and in_ds=='2D' and slice_dir=='xy':
            self.in_ds = xyt2xy
            
        # 2D spatio-temporal 3D output data processing
        if dim==3 and op_dim==2 and out_ds=='2D' and (slice_dir=='xt' or slice_dir=='yt' or slice_dir=='xtyt'):
            self.out_ds = xt_yt2xyt
            
        # 2D spatial 3D output data processing
        if dim==3 and op_dim==2 and out_ds=='2D' and slice_dir=='xy':
            self.out_ds = xy2xyt
    
    
   