import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
from network.Unet import Unet
# from network.detect_net import DetectNet
import dataio
from termcolor import colored

# Fermi impusle response
def translate_ir_func(delay, t, t0=0, C=1000, neg_shift=0):
    t = t-t0
    t = torch.cat((-torch.flip(t, [0])[-neg_shift-1:-1], t))    
    delayed_dirac = t-delay
    C = C/torch.pi
    translate_ir = np.sqrt(C/np.pi)*torch.exp(-C*(delayed_dirac)**2)    
    translate_ir = translate_ir/translate_ir.max()
    return translate_ir

def ModZ_OD(score, indx, thres = 3.5):
    # Detecting outliers 
    indx_outlier = (score > thres).nonzero().squeeze()
    if indx_outlier.numel() != 0:
        indx_od = torch.tensor([i.item() for i in indx if i not in indx_outlier])
    else:
        indx_od = indx
    # Generating mask
    mask_od = torch.ones(score.__len__())
    mask_od[indx_outlier] = 0
    return indx_od, mask_od

# Fermi impusle response
def fermi_ir_func(eta, t, t0=0, C=100, neg_shift=0, p=0.9999):    
    one = torch.ones(eta[:, 0, ...].shape, device=eta.device)
    t = t-t0
    t = torch.cat((-torch.flip(t, [0])[-neg_shift-1:-1], t))
    t_len = t.shape[0]
    t = t * torch.repeat_interleave(one.unsqueeze(-1), t_len, dim=-1)
    flow_rate = eta[:, 0, ...].unsqueeze(-1)
    delay = eta[:, 1, ...].unsqueeze(-1) # - ((2/C) * np.log(p/(1-p)))
    decay_rate = eta[:, 2, ...].unsqueeze(-1)
    delay_fermi = t-delay 
    delayed_heavy = t-delay
    heavy_side = torch.sigmoid(C*(delayed_heavy))
    output = flow_rate*(1/(torch.exp((delay_fermi)*decay_rate)+1)) * heavy_side
    return output

# Convolutional operator
def convolve(input, im_res, neg_shift=0):
     
    # output =  im_res ⨂ input 
    out_size = input.shape[-1] 
    # Pad input
    input_pad = F.pad(input,((0, neg_shift+out_size)))
    im_res_pad = F.pad(im_res,((0, out_size)))    
    # Apply FFT
    input_fft = torch.fft.rfft(input_pad, dim=-1)
    im_res_fft = torch.fft.rfft(im_res_pad, dim=-1)    
    # Convolution in fourier-domain
    output = im_res_fft*input_fft
    # Apply IFFT
    output = torch.fft.irfft(output, dim=-1)[..., neg_shift:neg_shift+out_size]
    
    return output

# Convolutional operator
def convolve_direct(input, im_res, neg_shift=0, seg=None):   
    # output =  im_res ⨂ input
    t_len = input.shape[-1]
    output = torch.zeros(input.shape, dtype=input.dtype, device=input.device)
    input = F.pad(input,((0, neg_shift)))
    im_res_flip = torch.flip(im_res, [-1])
    for t_indx in range(t_len):
        output[..., t_indx] = torch.sum(im_res_flip[..., -(t_indx+1+neg_shift):] * input[...,:(t_indx+1+neg_shift)], dim=-1)
    
    return output

# LBFGS solver for Fermi parameters
class FermiLBFGSSolver(nn.Module):
    
    def __init__(self, osamp=1, od_enable=False):        
        
        super(FermiLBFGSSolver, self).__init__()
        
        # Initializations
        S = 10
        self.register_buffer('S', torch.tensor(S))  
        self.register_buffer('S_op', expand_dim(torch.tensor([1,1/S,S]), f_dim_pad=1, b_dim_pad=2))
        self.register_buffer('SH_op', expand_dim(torch.tensor([1,S,1/S]), b_dim_pad=2))
        self.osamp = osamp
        self.od_enable = od_enable
  
    def forward(self, eta_init, ctc, aif, seg, time, indx_lbfgs=[]):
        
        # General initialization
        time = (time-time[0])/self.S
        time_osamp = interp_linear_1D(time.unsqueeze(0), size=self.osamp*time.shape[-1])[0]
        
        # Segmenting curves            
        aif_seg = aif#[seg==1]
        ctc_seg = ctc#[seg==1]
        
        # Compensating offset in the time curves
        oTp = 5
        aif_seg = aif_seg-aif_seg[...,0:oTp].mean(-1, keepdim=True)
        ctc_seg =  ctc_seg-ctc_seg[...,0:oTp].mean(-1, keepdim=True)
        
        # Oversampling curves (Linear)
        aif_osamp = interp_linear_1D(aif_seg, size=self.osamp*aif_seg.shape[-1])
        
        # lbfgs optimizer initialization
        global indx_lbfgs_od
        indx_lbfgs_od = indx_lbfgs = torch.arange(time.shape[-1]) if indx_lbfgs == [] else indx_lbfgs
        eta_lbfgs = self.S_op * eta_init
        eta_lbfgs.requires_grad = True
        # lbfgs = optim.LBFGS([eta_lbfgs], lr=1 , history_size=10, max_eval=1, max_iter=1, line_search_fn="strong_wolfe")
        lbfgs = optim.LBFGS([eta_lbfgs], lr=1 , history_size=10, max_eval=1000, max_iter=1000, line_search_fn="strong_wolfe")
        
        global zmod
        zmod = 0
            
        def closure():
            # Initializations
            global ctc_est_db  
            global fermi_ir_db
            global indx_lbfgs_od
            global zmod
            
            # Start optimization
            lbfgs.zero_grad()
            neg_shift = 2*self.osamp
            fermi_ir = fermi_ir_func(eta_lbfgs, time_osamp, C=500, neg_shift=neg_shift)
            # Segmenting fermi impulse response
            fermi_ir = fermi_ir#[seg==1]
            # Convolution
            ctc_est = convolve(aif_osamp, fermi_ir, neg_shift=neg_shift)[...,::self.osamp]/self.osamp
            
            # Modified z-score calculation
            tframes_err = torch.norm((ctc_seg-ctc_est)[seg==1], dim=0)
            zmod = (0.6745*(tframes_err - tframes_err.median()))/(tframes_err-tframes_err.median()).abs().median()
            
            # Saving vectors for debugging
            ctc_est_db = ctc_est.clone()
            fermi_ir_db = fermi_ir.clone()
                    
            ctc_est = ctc_est[..., indx_lbfgs_od]
            ctc_lbfgs = ctc_seg[..., indx_lbfgs_od]
                
            # Loss function
            C_mse = torch.sum(ctc_lbfgs**2)
            objective = torch.sum(((ctc_lbfgs - ctc_est))**2)/C_mse + (F.relu(-eta_lbfgs)**2).sum()
            objective.backward(retain_graph=True)
            return objective
        
        # LBFGS Execution    
        prev_mask_od = mask_od = torch.ones(indx_lbfgs.__len__())
        od_iter = 0
        od_max_iter = 3
        terminate = False
        while terminate == False:
            # LBFGS optimzation steps
            lbfgs.step(closure)
            if self.od_enable==True:
                indx_lbfgs_od, mask_od = ModZ_OD(zmod, indx_lbfgs)
                # Termination condition
                terminate = (mask_od-prev_mask_od).norm().item()==0 or od_iter>od_max_iter or self.od_enable!=True
                prev_mask_od = mask_od.clone()
                # Increment outlier detection iteration
                od_iter+=1
            else:
                terminate = True
        
        eta = self.SH_op * eta_lbfgs

        return eta, mask_od

def shift_aif(aif, mbolus, time, osamp):
    
    # Estimating delay for the aif
    # Initialization
    device = 'cuda'
    S = 10
    neg_shift = 20
    time_t0 = time[0]/S
    time = time/S
    time_osamp = interp_linear_1D(time.unsqueeze(0), size=osamp*time.shape[-1])[0]
    aifPreCorrection = aif
    
    # Compensating offset in the time curves
    oTp = 5
    mbolus = mbolus-mbolus[..., :oTp].mean(-1, keepdim=True)
    Comp_C = aifPreCorrection.max()/mbolus.max()     
    mbolus = Comp_C * mbolus
    
    # Oversampling curves (Linear)
    aifPreCorrection_osamp = interp_linear_1D(aifPreCorrection, size=osamp*aifPreCorrection.shape[-1])
    mbolus_osamp = interp_linear_1D(mbolus, size=osamp*mbolus.shape[-1])
    
    # For loss mask
    global index_array
    index_array = torch.arange(mbolus_osamp.shape[-1], device=device)
    
    # lbfgs optimizer initialization
    avg_time_step_size = (time-torch.roll(time, shifts=(1), dims=(0)))[1:].mean()
    global delay_init
    delay_init = (mbolus.argmax()-aif.argmax())*avg_time_step_size + 0.001
    delay_lbfgs = 1/S * torch.tensor(delay_init, device=device)
    delay_lbfgs.requires_grad = True 
    lbfgs = optim.LBFGS([delay_lbfgs], lr=1 , history_size=10, max_eval=100, max_iter=100, line_search_fn="strong_wolfe")
    
    def closure():
        # Initializations
        global aif_est
        global shift_ir
        global delay_init
        
        # Start optimization
        lbfgs.zero_grad()
        shift_ir = translate_ir_func(delay_lbfgs , time_osamp, time_t0, C=500, neg_shift=neg_shift*osamp)        
        
        # Fail safe mechanism incase delay gets undefined
        if torch.any(torch.isnan(delay_lbfgs))==False:
            shift_ir = translate_ir_func(delay_lbfgs , time_osamp, time_t0, C=500, neg_shift=neg_shift*osamp)
            shift_ir = shift_ir.squeeze(0).squeeze(0)
            aif_est = convolve(aifPreCorrection_osamp, shift_ir, neg_shift=neg_shift*osamp)
        else:
            print('NaN detected: Setting initial delay shift value and exiting.')
            shift_ir = translate_ir_func(delay_init , time_osamp, time_t0, C=500, neg_shift=neg_shift*osamp)
            shift_ir = shift_ir.squeeze(0).squeeze(0)
            aif_est = convolve(aifPreCorrection_osamp, shift_ir, neg_shift=neg_shift*osamp)
            return 0.0
        comp_factor = aifPreCorrection_osamp.max()/aif_est.detach().max()
        aif_est = comp_factor * aif_est
        
        # Calculating masks
        C = 0.01
        m_peak_index = mbolus_osamp.argmax()
        a_peak_index = aif_est.argmax()
        m_mask = torch.sigmoid(-(C*((index_array)-m_peak_index))).clone()
        a_mask = torch.sigmoid(-(C*((index_array)-a_peak_index))).clone()
        
        # Loss function
        m = m_mask * mbolus_osamp
        a = a_mask * aif_est
        objective = torch.sum(((m - a))**2)/(m**2).sum() + (F.relu((m[:,0:m_peak_index] - a[:,0:m_peak_index]))**2).sum()
        objective.backward(retain_graph=True)
    
        return objective
    
    lbfgs.step(closure)
    
    # Correcting AIF delay 
    del_delay = delay_lbfgs if torch.any(torch.isnan(delay_lbfgs))==False else delay_init
    shift_offset = 0
    shifts = int(torch.round(del_delay/avg_time_step_size)) + shift_offset
    aifCorrected = torch.roll(aifPreCorrection[0,:], shifts=(shifts), dims=(0))
        
    # Exclude rolled over values
    if shifts>=0:
        aifCorrected[:shifts]=0
    else:        
        aifCorrected[shifts:]=0
    
    return aifCorrected    
    
    # matplotlib.use('TkAgg')
    # plt.figure()
    # plt.title("Concentration Time Curves")
    # plt.plot(mbolus.squeeze(0).cpu(), label="mbolus", linewidth=1, color="blue", linestyle="solid")
    # plt.plot(aifPreCorrection.squeeze(0).cpu(), label="aif pre-shift", linewidth=1, color="black", linestyle="dashed")
    # plt.plot(aifCorrected.squeeze(0).cpu(), label="aif shifted", linewidth=1, color="red", linestyle="dashed")
    # plt.legend(loc="upper right")   
    # plt.show()
    
    # matplotlib.use('TkAgg') # DELETE
    # plt.figure()
    # plt.title("Concentration Time Curves")
    # plt.plot(a.squeeze(0).detach().cpu(), label="mbolus", linewidth=1, color="blue", linestyle="solid")
    # plt.plot(m.squeeze(0).detach().cpu(), label="mbolus real", linewidth=1, color="red", linestyle="solid")
    # plt.plot((m-a).abs().squeeze(0).detach().cpu(), label="error", linewidth=1, color="green", linestyle="solid")
    # plt.plot(((m*a)/torch.norm(m)).squeeze(0).detach().cpu(), label="overlap", linewidth=1, color="purple", linestyle="solid")
    # plt.legend(loc="upper right")   
    # plt.show()
    
# Masking for self-supervised learning
def ssup_mask(im_sig, mask_indx, seg, mbolus):
    
    # Generating mask
    mask =  torch.ones((mask_indx.shape[0], 1, 1, 1), device=im_sig.device)
    
    # Masking
    nb, nx, ny, nt = im_sig.shape
    im_sig_indx = np.arange(0, nb)
    # scaling = 3 + 1*(torch.rand((1, 1, 1, nt), device=im_sig.device) - 0.5)
    # ones = torch.ones((1, 1, 1, nt), device=im_sig.device)
    # sign = torch.pow(-ones,torch.bernoulli(0.1 * ones))
    
    # Main bolus framewise-perturbation
    mbolus_2d_dyn = seg.unsqueeze(-1) * mbolus.unsqueeze(1).unsqueeze(1)
    perturb_mask = torch.zeros((nb, 1, 1, nt), device=im_sig.device)
    mbolus_probability = 0.3
    perturb_mask[im_sig_indx,...,mask_indx] = torch.bernoulli( mbolus_probability * torch.ones((mask_indx.shape[0], im_sig.shape[0], 1, 1), device=im_sig.device))
    perturbation = (perturb_mask * ((1-seg.unsqueeze(-1)) * im_sig + mbolus_2d_dyn)) + ((1-perturb_mask) * (im_sig + torch.normal(0, 0.1 * im_sig.std(-1)).unsqueeze(-1)))

    # # Framewise-perturbation
    # mean = torch.zeros((nb, 1, 1, nt), device=im_sig.device)
    # std = (im_sig*seg.unsqueeze(-1)).std(dim=(1,2))
    # perturbation = torch.normal(mean, std).repeat(1,nx,ny,1) + im_sig[im_sig_indx, ..., mask_indx]
    
    # # Pixelwise-perturbation
    # mean = torch.zeros((nb, nx, ny, nt), device=im_sig.device)
    # std = (im_sig*seg.unsqueeze(-1)).std(dim=(1,2,3)).repeat(1,nx,ny,nt)
    # perturbation = torch.normal(mean, std) # .repeat(nb,nx,ny,1) + im_sig[im_sig_indx, ..., mask_indx]
    
    # Perturbation addition
    im_sig[im_sig_indx, ..., mask_indx] = (mask * (perturbation[im_sig_indx, ..., mask_indx])) + ((1-mask) * im_sig[im_sig_indx, ..., mask_indx])
    
    return im_sig

def anomaly_inject(im_sig, max_anon=5):
    
    # Calculate standard-deviation
    std = torch.ones(im_sig.shape, device=im_sig.device) * im_sig.std(-1, keepdim=True)
    # Anomaly injection
    outlier_scaling = 3 + (5*torch.rand((max_anon, im_sig.shape[0], 1, 1), device=im_sig.device))
    outlier_probability = 0.3
    outlier_mask = torch.tensor(torch.bernoulli(outlier_probability * torch.ones((max_anon, im_sig.shape[0], 1, 1))), device=im_sig.device)
    outlier_tindex = torch.randint(0, im_sig.shape[-1], (max_anon, im_sig.shape[0]))
    im_sig_index = torch.arange(0, im_sig.shape[0])
    im_sig[im_sig_index, ..., outlier_tindex] = (outlier_mask * (im_sig[im_sig_index, ..., outlier_tindex] + outlier_scaling * std[im_sig_index, ..., outlier_tindex])) + ((1-outlier_mask) * im_sig[im_sig_index, ..., outlier_tindex])
    
    # Anomaly location
    a_inj = torch.ones(im_sig.shape, device=im_sig.device)
    a_inj[im_sig_index, ..., outlier_tindex] = (1-outlier_mask)
    
    return im_sig, a_inj

def outlier_fill(signal, mask_od, wlen):
    
    # Exclude outliers and linearly interpolate using immediate neighbour
    nb, _, _, _ = signal.shape
    for i in range(nb):
        
        # Extracting immediate non-outlier neighbour        
        mask_od_i = mask_od[i][:wlen[i]]        
        od_indx = (mask_od_i==0).nonzero()
        signal_indx = (mask_od_i==1).nonzero()
        # Handling usecase when signal end is an outlier
        if signal_indx[-1] != wlen[i]-1:
            signal[i,...,wlen[i]-1] = signal[i,...,signal_indx[-1].item()]
            mask_od_i = torch.cat((mask_od_i, torch.tensor([1])))
            signal_indx = (mask_od_i==1).nonzero()                       
        for j in range(od_indx.numel()):
            od_indx_j = od_indx[j].item()
            if od_indx_j != 0 and od_indx_j != wlen[i]-1:
                right = signal_indx[signal_indx>od_indx_j][0]
                left = signal_indx[signal_indx<od_indx_j][-1]
                signal[i,...,od_indx_j] = ((signal[i,...,right]-signal[i,...,left])/(right-left))*(od_indx_j-left) + signal[i,...,left]
    
    return signal