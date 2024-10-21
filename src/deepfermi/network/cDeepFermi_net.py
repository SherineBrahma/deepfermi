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

class cDeepFermi(nn.Module):
    
    def __init__(self, cnn, time, osamp=1, nu=1, max_iter_lbfgs=100, max_eval_lbfgs=100, mode='pre_training', learn_lambda=True):
        
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
  
        assert mode in ['pre_training','fine_tuning','testing'], \
			"mode has to be one of 'pre_training', 'fine_tuning' or 'testing"
        
        
        super(cDeepFermi, self).__init__()
  
        self.cnn = cnn
        self.nu = nu 
        self.mode = mode
          
		# Learned regularizer
        beta=1.
        lambda_init = np.log(np.exp(beta)-1.)/beta
        self.lambda_reg = nn.Parameter(torch.tensor(5.35123*lambda_init, dtype=torch.float), requires_grad=bool(learn_lambda))
        
        # Parameter estimation module
        self.synth_nn = SynthNet(dim = 3, ncin=24, nfilters=24, ncout=3, nstage=3, nconv_stage=2, bias=False, groups=3)
        
        # Fermi specific
        S = 10        
        self.register_buffer('S_op', expand_dim(torch.tensor([1,1/S,S]), f_dim_pad=1, b_dim_pad=2))
        self.register_buffer('SH_op', expand_dim(torch.tensor([1,S,1/S]), b_dim_pad=2))
        self.register_buffer('time', time/S)        
        self.register_buffer('osamp', torch.tensor(osamp))
        self.register_buffer('max_iter_lbfgs', torch.tensor(max_iter_lbfgs))
        self.register_buffer('max_eval_lbfgs', torch.tensor(max_eval_lbfgs))
        self.dc_module = FermiDataConsLBFGS.apply
  
    def forward(self, xin, zin, seg, aif=None, ctc=None, indx_dc=None):
        
        self.lbfgs_iter = 0
        
        if self.mode =='pre_training':			
			
   			# Apply neural networks
            ein = xin
            xcnn = self.cnn(ein, zin)
            eta_nn = self.synth_nn(xcnn)
            eta = eta_nn
   
            return eta
                			
        elif self.mode in['fine_tuning', 'testing']:

            assert aif!=None and ctc!=None, "Arterial input function and concentration time curve required for ensuring data-consistency!"            
            
            # # Denoising ctc
            # ctc = svd_approx(ctc, k=4)
						
            for _ in range(self.nu):
                
				# Apply neural networks
                ein = xin
                xcnn = self.cnn(ein, zin)
                eta_nn = self.synth_nn(xcnn)
                
                a = 1
                b = 1
                
				# Apply data-consistency layer
                eta_nn = self.S_op * seg.unsqueeze(1) * eta_nn                
                eta_pi, self.lbfgs_iter = self.dc_module(ctc, aif, self.time, seg, self.osamp, indx_dc, self.lambda_reg, eta_nn, self.max_iter_lbfgs, self.max_eval_lbfgs)            
                
            eta = self.SH_op * eta_pi
            # print('Total lbfgs iteration: ' + str(self.lbfgs_iter))
            # print('lambda_reg value: ' + str(self.lambda_reg))

            return eta

class SynthNet(nn.Module):

    def __init__(self, dim=2, ncin=2, nfilters=2, ncout=2, nstage=3, nconv_stage=2, bias=False, groups=1):
        super(SynthNet, self).__init__()
        
        # General Initializations
        dsamp_fact = 2
        if dim==2:
            pool_kshape = (1, 2)
            img_dim_out = (None, 1)
        elif dim==3:
            pool_kshape = (1,1,2)
            img_dim_out = (None, None, 1)
        
        # Constructing network
        self.synth_net = nn.ModuleList()
        nch = ncin
        ncout_layer = nfilters // dsamp_fact        
        for _ in range(nstage):
            self.synth_net.append(ConvBlock(dim=dim, shape=3, nch=nch, nfilters=nfilters, ncout=ncout_layer, nconvs=nconv_stage, pad=1, bias=bias, groups=groups, res_connect=True))
            self.synth_net.append(Pooling(dim=dim, kernel_size=pool_kshape, pooling_type="Max"))
            nch = ncout_layer
            nfilters = ncout_layer
            ncout_layer = nfilters // dsamp_fact     
        self.synth_net.append(ConvLayer(dim=dim, shape=1, nch=nch, nfilters=ncout, pad=0, groups=groups))
        if dim==2:
            self.synth_net.append(nn.AdaptiveAvgPool2d(img_dim_out))
        elif dim==3:
            self.synth_net.append(nn.AdaptiveAvgPool3d(img_dim_out))
        

    def __call__(self, xin):
        
        # Data processing
        xconv = xin
        for i in range(len(self.synth_net)):
            xconv = self.synth_net[i](xconv)
        
        # Output
        xout = xconv.squeeze(-1)
        	
        return xout
    
class FermiDataConsLBFGS(torch.autograd.Function):
    """Both forward and backward are static methods."""

    @staticmethod
    def forward(ctx, ctc, aif, time, seg, osamp, indx_dc, lambda_reg, eta_nn, max_iter, max_eval):
        
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        
        # Segmenting curves            
        aif_seg = aif[seg==1]
        ctc_seg = ctc[seg==1]
        
        # Compensating offset in the time curves
        oTp = 5
        aif_seg = F.relu(aif_seg-aif_seg[...,0:oTp].mean(-1, keepdim=True))
        ctc_seg = F.relu(ctc_seg-ctc_seg[...,0:oTp].mean(-1, keepdim=True))
        
        # Oversampling curves (Linear)
        aif_osamp = interp_linear_1D(aif_seg, size=osamp*aif_seg.shape[-1])
        ctc_osamp = interp_linear_1D(ctc_seg, size=osamp*ctc_seg.shape[-1])
        time_osamp = interp_linear_1D(time.unsqueeze(0), size=osamp*time.shape[-1])[0]
        
        # Initializing data-consistency objective
        eta_prior = eta_nn.detach().clone()
        lambda_reg = lambda_reg.detach().clone()
        F_Op = FermiDConsObj(eta_prior, ctc_osamp, aif_osamp, time_osamp, seg, osamp, indx_dc, lambda_reg)  
        
        # Defining Closure        
        def closure():                        
            # Start optimization
            lbfgs.zero_grad()
            loss = F_Op(eta_pi)            
            loss.backward()                              
            return loss
        
        # LBFGS setup and execution     
        eta_pi =  eta_prior.detach().clone()
        eta_pi.requires_grad = True
        lbfgs = optim.LBFGS([eta_pi], lr=1 , history_size=100, max_iter=max_iter, max_eval=max_eval, line_search_fn="strong_wolfe")
        lbfgs.step(closure)
        n_iter = lbfgs.n_iter
        
        # Saving objects required for backward    
        ctx._the_function = F_Op
        ctx.save_for_backward(eta_pi, lambda_reg, eta_prior)
        
        return eta_pi, n_iter

    @staticmethod
    def backward(ctx, grad_out, _ ):
        
        # Loading objects saved during forward
        eta_pi, lambda_reg, eta_prior = ctx.saved_tensors
        F_Op = ctx._the_function          
        g = FermiDConsConjGrad(F_Op, eta_pi, grad_out, grad_out, niter=4)
        
        # Computing gradients to be backpropagated        
        grad_ctc = grad_aif = grad_time = grad_seg = grad_osamp = grad_indx_dc = grad_max_iter = grad_max_eval = None
        C_nn = (eta_prior**2).sum(dim=(0,2,3), keepdim=True)
        grad_eta_prior = g * (2/C_nn)  * F.softplus(lambda_reg)
        grad_lambda_reg = -(g * (2/C_nn) * torch.sigmoid(lambda_reg) * (eta_pi-eta_prior)).sum()
        
        return grad_ctc, grad_aif, grad_time, grad_seg, grad_osamp, grad_indx_dc, grad_lambda_reg, grad_eta_prior, grad_max_iter, grad_max_eval
    
    
class FermiDConsObj(nn.Module):

    def __init__(self, eta_prior, ctc_osamp, aif_osamp, time_osamp, seg, osamp, indx_dc, lambda_reg):
        super(FermiDConsObj, self).__init__()
        
        # General Initializations
        self.eta_prior = eta_prior
        self.ctc_osamp = ctc_osamp
        self.aif_osamp = aif_osamp
        self.time_osamp = time_osamp
        self.seg = seg
        self.osamp = osamp
        self.indx_dc = indx_dc
        self.lambda_reg = lambda_reg
            
    def __call__(self, eta_pi):
        
        # Calculating and segmenting fermi impulse response
        fermi_ir = fermi_ir_func(eta_pi, self.time_osamp)
        fermi_ir = fermi_ir[self.seg==1]
        
        # Convolution operation            
        if self.indx_dc is None:
            ctc_est = convolve(self.aif_osamp, fermi_ir)/self.osamp  
            ctc_dc = self.ctc_osamp
        else:
            ctc_est = convolve(self.aif_osamp, fermi_ir)[..., self.indx_dc*self.osamp]/self.osamp
            ctc_dc = self.ctc_osamp[..., self.indx_dc*self.osamp]
            
        # Constructing objective                
        C_nn = (self.eta_prior**2).sum(dim=(0,2,3), keepdim=True)
        C_dc = (ctc_dc**2).sum()
        objective = ((ctc_dc - ctc_est)**2/C_dc).sum() + F.softplus(self.lambda_reg) * (((eta_pi - self.eta_prior)**2)/C_nn).sum()
        
        return objective
    
def FermiDConsConjGrad(F_Op, eta_pi, g, grad_in, niter=10):
    
    # g is the starting value, grad_in the rhs;
    _, r = hvp(F_Op, eta_pi, v=g)
    r = grad_in-r

    #initialize p
    p = r.clone()

    #old squared norm of residual
    sqnorm_r_old = torch.bmm(r.flatten(start_dim=1).unsqueeze(1),r.flatten(start_dim=1).unsqueeze(-1))
     
    for _ in range(niter):

        #calculate Hp;
        _, d = hvp(F_Op, eta_pi, v=p)

        #calculate step size alpha;
        inner_p_d = torch.bmm(p.flatten(start_dim=1).unsqueeze(1),  d.flatten(start_dim=1).unsqueeze(-1))
        alpha = expand_dim(sqnorm_r_old / inner_p_d, b_dim_pad=1)

        #perform step and calculate new residual;
        g = g + alpha*p
        r = r - alpha*d
        
        #new residual norm
        sqnorm_r_new = torch.bmm(r.flatten(start_dim=1).unsqueeze(1),r.flatten(start_dim=1).unsqueeze(-1))
        # print('||res_||_2^2 = {}'.format(sqnorm_r_new))        
        #calculate beta and update the norm;
        beta = expand_dim(sqnorm_r_new / sqnorm_r_old, b_dim_pad=1)
        sqnorm_r_old = sqnorm_r_new

        p = r + beta*p

    return g


