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

class DeepFermi(nn.Module):
    
    def __init__(self, cnn, osamp=1, nu=1, max_iter_lbfgs=100, max_eval_lbfgs=100, mode='pre_training', learn_lambda=True, od_enable=False):
        
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
        
        
        super(DeepFermi, self).__init__()
  
        self.cnn = cnn
        self.nu = nu 
        self.mode = mode
        self.inorm_stage_1 = InstanceNorm(2, dim=3, affine=True)
        self.inorm_stage_2 = InstanceNorm(36, dim=3, affine=True)
        
        # self.inorm_stage_1 = InstanceNorm(6, dim=3, affine=True)
        
		# Learned regularizer
        beta = 1.
        lambda_init = np.log(np.exp(beta)-1.)/beta
        self.lambda_reg = nn.Parameter(torch.tensor(5.35123*lambda_init, dtype=torch.float), requires_grad=bool(learn_lambda))
        
        # Parameter estimation module
        self.synth_nn = SynthNet(dim=3, ncin=36, nfilters=48, ncout=3, nstage=3, nconv_stage=2, bias=False, groups=3)
        
        # Fermi specific
        S = 10
        self.register_buffer('S', torch.tensor(S))
        self.register_buffer('S_op', expand_dim(torch.tensor([1,1/S,S]), f_dim_pad=1, b_dim_pad=2))
        self.register_buffer('SH_op', expand_dim(torch.tensor([1,S,1/S]), b_dim_pad=2))
        self.register_buffer('osamp', torch.tensor(osamp))
        self.register_buffer('max_iter_lbfgs', torch.tensor(max_iter_lbfgs))
        self.register_buffer('max_eval_lbfgs', torch.tensor(max_eval_lbfgs))
        self.od_enable = od_enable
        self.dc_module = FermiDataConsLBFGS.apply
  
    def forward(self, xin, seg, aif=None, ctc=None, time=None, indx_dc=[]):
        
        assert xin.shape[0] == 1, "Only mb=1 supported"
        
        self.lbfgs_iter = 0        
        time = (time-time[:,0])/self.S
        time_2D_dyn = (time.unsqueeze(1).unsqueeze(1) * torch.ones(ctc.shape, device=ctc.device))
        aif_2D_dyn = (aif.unsqueeze(1).unsqueeze(1) * torch.ones(ctc.shape, device=ctc.device))
        
        if self.mode =='pre_training':
            
            # Apply neural networks                
            seg_in = seg.unsqueeze(1).unsqueeze(-1).repeat(1,1,1,1, xin.shape[-1])
            xin = self.inorm_stage_1(torch.cat((xin, seg_in),1))
            xcnn = self.cnn(xin)
            aif_cnn = aif_2D_dyn.unsqueeze(1).repeat(1, xcnn.shape[1],1,1,1)
            time_cnn = time_2D_dyn.unsqueeze(1).repeat(1, xcnn.shape[1],1,1,1)
            xcnn = self.inorm_stage_2(torch.cat((aif_cnn, xcnn, time_cnn),1)) # Change later
            eta_nn = self.synth_nn(xcnn)
            eta = eta_nn
   
            return eta
                			
        elif self.mode in['fine_tuning', 'testing']:

            assert aif!=None and ctc!=None, "Arterial input function and concentration time curve required for ensuring data-consistency!"
						
            for _ in range(self.nu):
                
                # Apply neural networks                
                seg_in = seg.unsqueeze(1).unsqueeze(-1).repeat(1,1,1,1, xin.shape[-1])
                xin = self.inorm_stage_1(torch.cat((xin, seg_in),1))
                xcnn = self.cnn(xin)
                aif_cnn = aif_2D_dyn.unsqueeze(1).repeat(1, xcnn.shape[1],1,1,1)
                time_cnn = time_2D_dyn.unsqueeze(1).repeat(1, xcnn.shape[1],1,1,1)
                xcnn = self.inorm_stage_2(torch.cat((aif_cnn, xcnn, time_cnn),1)) # Change later
                eta_nn = self.synth_nn(xcnn)
                
				# Apply data-consistency layer
                eta_nn = self.S_op * eta_nn # if self.mode == 'fine_tuning' else self.S_op * seg.unsqueeze(1) * eta_nn          
                eta_pi, self.lbfgs_iter = self.dc_module(ctc, aif_2D_dyn, time, seg, self.osamp, indx_dc, self.lambda_reg, eta_nn, self.max_iter_lbfgs, self.max_eval_lbfgs, self.od_enable)            
                # eta_pi = eta_nn
                
            eta = self.SH_op * eta_pi

            return eta

class SynthNet(nn.Module):

    def __init__(self, dim=2, ncin=2, nfilters=2, ncout=2, nstage=3, nconv_stage=2, bias=False, groups=1):
        super(SynthNet, self).__init__()
        
        # General Initializations
        dsamp_fact = 2
        padding_mode = 'circular'
        if dim==1:
            pool_kshape = (2)
            img_dim_out = (5)
            shape = (3)
            pad = (1)
        elif dim==2:
            pool_kshape = (1, 2)
            img_dim_out = (None, 5)
            shape = (1, 3)
            pad = (0, 1)
        elif dim==3:
            pool_kshape = (1,1,2)
            img_dim_out = (None, None, 1)
            shape = (1, 1, 3)
            pad = (0, 0, 1)
        
        # Constructing network
        self.synth_net = nn.ModuleList()
        # self.synth_net.append(ConvLayer(dim=dim, shape=1, nch=ncin, nfilters=nfilters, pad=0, bias=bias, padding_mode=padding_mode))    
        self.synth_net.append(ConvBlock(dim=dim, shape=shape, nch=ncin, nfilters=nfilters, nconvs=nconv_stage, pad=pad, bias=bias, res_connect=True))    
        nch = nfilters
        ncout_layer = nfilters // dsamp_fact 
        for _ in range(nstage):
            self.synth_net.append(ConvBlock(dim=dim, shape=shape, nch=nch, nfilters=nfilters, ncout=ncout_layer, nconvs=nconv_stage, pad=pad, bias=bias, groups=groups, res_connect=True))
            self.synth_net.append(Pooling(dim=dim, kernel_size=pool_kshape, pooling_type="Max"))
            nch = ncout_layer
            nfilters = ncout_layer
            ncout_layer = nfilters // dsamp_fact
        
        if dim==1:
            self.synth_net.append(nn.AdaptiveAvgPool1d(img_dim_out))
        elif dim==2:
            self.synth_net.append(nn.AdaptiveAvgPool2d(img_dim_out))
        elif dim==3:
            self.synth_net.append(nn.AdaptiveAvgPool3d(img_dim_out))
            
        self.synth_net.append(ConvBlock(dim=dim, shape=shape, nch=nch, nfilters=ncout, ncout=ncout, nconvs=nconv_stage, pad=pad, bias=bias, groups=groups, res_connect=True))
        self.synth_net.append(ConvLayer(dim=dim, shape=1, nch=ncout, nfilters=ncout, pad=0, bias=bias, groups=groups))

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
    def forward(ctx, ctc, aif, time, seg, osamp, indx_dc, lambda_reg, eta_nn, max_iter, max_eval, od_enable):
        
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        
        # Segmenting curves            
        aif_seg = aif#[seg==1]
        ctc_seg = ctc#[seg==1]
        
        # Compensating offset in the time curves
        oTp = 5
        # aif_seg = F.relu(aif_seg-aif_seg[...,0:oTp].mean(-1, keepdim=True))
        # ctc_seg = F.relu(ctc_seg-ctc_seg[...,0:oTp].mean(-1, keepdim=True))
        aif_seg = aif_seg-aif_seg[...,0:oTp].mean(-1, keepdim=True)
        ctc_seg = ctc_seg-ctc_seg[...,0:oTp].mean(-1, keepdim=True)
        
        # Oversampling curves (Linear)
        aif_osamp = interp_linear_1D(aif_seg, size=osamp*aif_seg.shape[-1])
        ctc_osamp = interp_linear_1D(ctc_seg, size=osamp*ctc_seg.shape[-1])
        time_osamp = interp_linear_1D(time, size=osamp*time.shape[-1])
        
        # Initializing data-consistency objective
        eta_prior = eta_nn.detach().clone()
        lambda_reg = lambda_reg.detach().clone()
        F_Op = FermiDConsObj(eta_prior, ctc_osamp, aif_osamp, time_osamp, seg, osamp, indx_dc, lambda_reg)
                
        # Defining Closure        
        def closure():        
            # Start optimization
            lbfgs.zero_grad()
            F_Op.n_iter = lbfgs.n_iter
            loss = F_Op(eta_pi)
            loss.backward()
            return loss
        
        # Initializing physics-informed perfusion parameters
        eta_pi = eta_prior.detach().clone()
        eta_pi.requires_grad = True
        
        # LBFGS setup
        lbfgs = optim.LBFGS([eta_pi], lr=1 , history_size=10, max_iter=max_iter, max_eval=max_eval, line_search_fn="strong_wolfe")
        
        # LBFGS Execution        
        indx_total = torch.arange(time.shape[-1])        
        prev_mask_od = torch.ones(indx_total.__len__())
        od_iter = 0
        od_max_iter = 5
        terminate = False
        while terminate == False:
        
            # LBFGS optimzation steps
            lbfgs.step(closure)                    
            F_Op.indx_dc_od, mask_od = ModZ_OD(F_Op.zmod, F_Op.indx_dc)
            # Termination condition
            terminate = (mask_od-prev_mask_od).norm().item()==0 or od_iter>od_max_iter or od_enable!=True
            prev_mask_od = mask_od.clone()
            # Increment outlier detection iteration
            od_iter+=1
            
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
        g = FermiDConsConjGrad(F_Op, eta_pi, grad_out, grad_out, niter=10)
        
        # Computing gradients to be backpropagated        
        grad_ctc = grad_aif = grad_time = grad_seg = grad_osamp = grad_indx_dc = grad_max_iter = grad_max_eval = grad_od_enable = None
        grad_eta_prior = 2 * g * F.softplus(lambda_reg) 
        grad_lambda_reg = -(2 * g * torch.sigmoid(lambda_reg) * (eta_pi-eta_prior)).sum()
        
        # map = grad_eta_prior[0,0,...]
        # matplotlib.use('TkAgg')
        # plt.figure()
        # plt.imshow((map).detach().cpu())  
        # plt.show()
        
        # map = grad_eta_prior[0,1,...]
        # matplotlib.use('TkAgg')
        # plt.figure()
        # plt.imshow((map).detach().cpu())  
        # plt.show()  
        
        # map = grad_eta_prior[0,2,...]
        # matplotlib.use('TkAgg')
        # plt.figure()
        # plt.imshow((map).detach().cpu())  
        # plt.show()  
        
        return grad_ctc, grad_aif, grad_time, grad_seg, grad_osamp, grad_indx_dc, grad_lambda_reg, grad_eta_prior, grad_max_iter, grad_max_eval, grad_od_enable

class FermiDConsObj(nn.Module):

    def __init__(self, eta_prior, ctc_osamp, aif_osamp, time_osamp, seg, osamp, indx_dc, lambda_reg):
        super(FermiDConsObj, self).__init__()
        
        # General Initializations
        self.eta_prior = eta_prior
        self.ctc_osamp = ctc_osamp
        self.aif_osamp = aif_osamp
        self.time_osamp = time_osamp
        self.seg = seg
        self.osamp = osamp.item()
        self.indx_dc_od = self.indx_dc = torch.arange(ctc_osamp[...,::osamp].shape[-1]) if indx_dc == [] else indx_dc
        self.lambda_reg = lambda_reg
        self.prev_iter = 0
        self.n_iter = 0
            
    def __call__(self, eta_pi):
        
        # Negative shift allowed
        neg_shift = 2*self.osamp
        
        # Calculating and segmenting fermi impulse response
        fermi_ir = fermi_ir_func(eta_pi, self.time_osamp.squeeze(), C=500, neg_shift=neg_shift)
        fermi_ir = fermi_ir #[self.seg==1]        
        
        # Convolution operation         
        ctc_est = convolve(self.aif_osamp, fermi_ir, neg_shift=neg_shift)[...,::self.osamp]/self.osamp
        ctc_dc = self.ctc_osamp[...,::self.osamp]
        
        # Modified z-score calculation
        tframes_err = torch.norm((ctc_dc-ctc_est)[self.seg==1], dim=0).detach()
        self.zmod = (0.6745*(tframes_err - tframes_err.median()))/(tframes_err-tframes_err.median()).abs().median()
        
        # Rejecting outliers                
        ctc_est = ctc_est[..., self.indx_dc_od]
        ctc_dc = ctc_dc[..., self.indx_dc_od]
        
        # Constructing objective
        C_dc = (ctc_dc**2).sum()
        objective = ((ctc_dc - ctc_est)**2/C_dc).sum() + F.softplus(self.lambda_reg)*(((eta_pi - self.eta_prior)**2)).sum() + (F.relu(-eta_pi)**2).sum() 
        
        return objective

def FermiDConsConjGrad(F_Op, eta_pi, g, grad_in, niter=5):
    
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
        inner_p_d = torch.bmm(p.flatten(start_dim=1).unsqueeze(1), d.flatten(start_dim=1).unsqueeze(-1))
        alpha = expand_dim(sqnorm_r_old / inner_p_d, b_dim_pad=1)

        #perform step and calculate new residual;
        g = g + alpha*p
        r = r - alpha*d
        
        # new residual norm
        sqnorm_r_new = torch.bmm(r.flatten(start_dim=1).unsqueeze(1),r.flatten(start_dim=1).unsqueeze(-1))
        # print('||res_||_2^2 = {}'.format(sqnorm_r_new))        
        # calculate beta and update the norm;
        beta = expand_dim(sqnorm_r_new / sqnorm_r_old, b_dim_pad=1)
        sqnorm_r_old = sqnorm_r_new

        p = r + beta*p
        
        # map = g[0,2,...]
        # matplotlib.use('TkAgg')
        # plt.figure()
        # plt.imshow((map).detach().cpu())
        # plt.title("Flow")
        # plt.show()
        
    if torch.any(torch.isnan(g))==True:
        print('NaN detected: Skipping optimization layer backpropagation')
        g = torch.zeros(g.shape, device=g.device)
        
    return g

class PixelwiseFeedforwardLayer(nn.Module):
    def __init__(self, nch, node_in, node_out):
        super(PixelwiseFeedforwardLayer, self).__init__()
        
        # Initialization
        self.node_out = node_out
        
        # Constructing network
        self.pff_net = nn.ModuleList()
        for _ in range(nch):
            self.pff_net.append(nn.Linear(node_in, node_out))

    def forward(self, xin):
        
        # Extracting shape
        nb, nch, nx, ny, nt = xin.shape
        
        # Apply pixel-wise feed-forward layer
        xout = torch.zeros((nb, nch, nx, ny, self.node_out), device=xin.device)
        for i in range(nch):
            a = 1
            xpff = xin[:,i,...].view(nb * nx * ny, nt)
            xpff = self.pff_net[i](xpff)
            xout[:,i,...] = xpff.view(nb, nx, ny, -1)
        
        return xout