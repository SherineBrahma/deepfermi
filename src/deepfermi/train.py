import torch
from fermi import *
from utils import *

def unet_train(im_sig, aif, ctc, seg, time, wlen, indx_nn, indx_dc, mbolus, unet, unet_optmzr, fermi_params, od_enable=False):    
    # Extracting parameters    
    S = fermi_params['S']
    S_op = fermi_params['S_op']
    osamp = fermi_params['osamp']
    
    # Windowing
    im_sig = im_sig[...,0:wlen]
    aif = aif[...,0:wlen]
    mbolus = mbolus[...,0:wlen]
    ctc = ctc[...,0:wlen]
    time = time[...,0:wlen]

    # Zeroing out Gradients
    unet.zero_grad()
    
    # Activating training mode
    unet.train()

    # Compute only adversarial gradients and not autoencoder decoder gradients
    for p in unet.parameters():
        p.requires_grad = True

    # Estimate perfusion parameters
    # im_sig, _ = anomaly_inject(im_sig)        
    im_sig = ssup_mask(im_sig, indx_nn[:,np.newaxis], seg, mbolus)
    eta_gen = unet(im_sig.unsqueeze(1), seg, aif=aif, ctc=ctc, time=time, indx_dc=indx_dc)
    
    # Segmenting curves
    aif_2D_dyn = aif.unsqueeze(1).unsqueeze(1) * torch.ones(ctc.shape, device=ctc.device)
    aif_seg = aif_2D_dyn
    ctc_seg = ctc
    
    # Compensating offset in the time curves
    oTp = 5
    aif = aif_seg-aif_seg[...,0:oTp].mean(-1, keepdim=True)
    ctc = ctc_seg-ctc_seg[...,0:oTp].mean(-1, keepdim=True)
    
    # Oversampling curves (Linear)
    time = (time-time[:,0])
    time_osamp = interp_linear_1D(time, size=osamp*time.shape[-1])/S
    aif_osamp = interp_linear_1D(aif, size=osamp*aif.shape[-1])
    
    # Estimating concentration curves
    neg_shift = 2*osamp
    fermi_ir = fermi_ir_func(S_op * eta_gen, time_osamp.squeeze(), C=500, neg_shift=neg_shift)
    ctc_est = convolve(aif_osamp, fermi_ir, neg_shift=neg_shift)[..., ::osamp]/osamp
    
    # Modified Z-score outlier rejection
    if od_enable==True:
        temporal_frames_error = torch.norm((ctc-ctc_est)[seg==1], dim=0).detach()
        zmod = (0.6745*(temporal_frames_error - temporal_frames_error.median()))/(temporal_frames_error-temporal_frames_error.median()).abs().median()
        indx_nn_od, mask_od = ModZ_OD(zmod, indx_nn)
        # For monitoring
        print((mask_od==0).nonzero().squeeze())
    else:           
        indx_nn_od = indx_nn
                    
    ctc_est = ctc_est[..., indx_nn_od]  
    ctc = ctc[..., indx_nn_od]
    
    # Self-supervised loss                     
    C_ssup = torch.sum(ctc)**2
    ssup_loss = torch.sum((ctc - ctc_est)**2)/C_ssup # + (F.relu(-eta_gen)**2).sum() # Relu is added for later experiment
    
    # Updating the network
    ssup_loss.backward()
    
    unet_optmzr.step()

    return ssup_loss

def unet_pretrain(im_sig, aif, ctc, seg, time, wlen, indx_nn, eta_pretrain, mbolus, unet, unet_optmzr, iter):    
    # Windowing
    im_sig = im_sig[...,0:wlen]
    aif = aif[...,0:wlen]
    mbolus = mbolus[...,0:wlen]
    ctc = ctc[...,0:wlen]
    time = time[...,0:wlen]        

    # Zeroing out Gradients
    unet.zero_grad()
    
    # Activating training mode
    unet.train()

    # Compute only adversarial gradients and not autoencoder decoder gradients
    for p in unet.parameters():
        p.requires_grad = True
    
    # Train Unet
    im_sig = ssup_mask(im_sig, indx_nn[:,np.newaxis], seg, mbolus)
    eta_gen = unet(im_sig.unsqueeze(1), seg, aif=aif, ctc=ctc, time=time)
    
    # mse loss function
    # C = 1
    C = (eta_pretrain**2).sum(dim=(0,2,3), keepdim=True)
    # if iter<0:#20000:      
    #     C = 1
    # else:
    #     C = (eta_pretrain**2).sum(dim=(0,2,3), keepdim=True)
    loss = (((eta_pretrain - eta_gen)**2)/C).sum() # + (F.relu(-eta_gen)**2).sum() 
    # loss = ((eta_pretrain - eta_gen)**2).sum()
    
    # Updating the network
    loss.backward()
    unet_optmzr.step()
    
    # matplotlib.use('TkAgg')
    # plt.figure()
    # plt.title("Concentration Time Curves")
    # plt.plot(mbolus.squeeze(0).cpu(), label="mbolus", linewidth=1, color="blue", linestyle="solid")
    # plt.plot(aif.squeeze(0).cpu(), label="aif", linewidth=1, color="red", linestyle="solid")
    # plt.legend(loc="upper right")   
    # plt.show()
    
    # map = ctc[0,...].mean(-1)
    # matplotlib.use('TkAgg')
    # plt.figure()
    # plt.imshow((map).detach().cpu())  
    # plt.show()
    
    unet(im_sig.unsqueeze(1), seg, aif=aif, ctc=ctc, time=time)

    return loss

def unet_eval(im_sig, aif, ctc, seg, time, wlen, unet, fermi_params):    
    # Extracting parameters    
    S = fermi_params['S']
    S_op = fermi_params['S_op']
    osamp = fermi_params['osamp']
        
    # Windowing
    im_sig = im_sig[...,0:wlen]
    aif = aif[...,0:wlen]
    ctc = ctc[...,0:wlen]
    time = time[...,0:wlen]
    
    # Estimate perfusion parameters
    eta_gen = unet(im_sig.unsqueeze(1), seg, aif=aif, ctc=ctc, time=time)
    
    # Segmenting curves
    aif_2D_dyn = aif.unsqueeze(1).unsqueeze(1) * torch.ones(ctc.shape, device=ctc.device)
    aif_seg = aif_2D_dyn[seg==1]
    ctc_seg = ctc[seg==1]
    
    # Compensating offset in the time curves
    oTp = 5
    aif = aif_seg-aif_seg[...,0:oTp].mean(-1, keepdim=True)
    ctc = ctc_seg-ctc_seg[...,0:oTp].mean(-1, keepdim=True)
    
    # Oversampling curves (Linear)
    time = (time-time[:,0])
    time_osamp = interp_linear_1D(time, size=osamp*time.shape[-1])/S
    aif_osamp = interp_linear_1D(aif, size=osamp*aif_seg.shape[-1])
    
    # Self-supervised loss
    neg_shift = 2*osamp
    fermi_ir = fermi_ir_func(S_op * eta_gen, time_osamp.squeeze(), C=500, neg_shift=neg_shift)
    fermi_ir = fermi_ir[seg==1]
    ctc_est = convolve(aif_osamp, fermi_ir, neg_shift=neg_shift)[..., ::osamp]/osamp
    C_ssup = torch.sum(ctc**2)
    ssup_loss = torch.sum((ctc - ctc_est)**2)/C_ssup

    return ssup_loss