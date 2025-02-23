import numpy as np
import torch
from fermi import ModZ_OD, convolve, fermi_ir_func, ssup_mask
from utils import interp_linear_1D


def unet_train(im_sig,
               aif,
               ctc,
               seg,
               time,
               wlen,
               indx_nn,
               indx_dc,
               mbolus,
               unet,
               unet_optmzr,
               fermi_params,
               od_enable=False):
    """
    Train DeepFermi using self-supervised learning to estimate perfusion
    parameters based on input signal images, arterial input function (AIF),
    concentration-time curves (CTC), and time steps. It includes windowing,
    optional modified Z-score outlier rejection training to train on
    "online cleaned" data.

    Parameters:
        im_sig (torch.Tensor): The input signal image (e.g., MRI images).
        aif (torch.Tensor): The arterial input function (AIF).
        ctc (torch.Tensor): The concentration-time curve (CTC).
        seg (torch.Tensor): The segmentation mask for myocardium.
        time (torch.Tensor): The time points for the concentration curves.
        wlen (int): The length of the first pass of the main bolus window.
        indx_nn (torch.Tensor): Indices for the neural network training.
        indx_dc (torch.Tensor): Indices for data consistency.
        mbolus (torch.Tensor): The bolus image used for masking.
        unet (nn.Module): The U-Net model.
        unet_optmzr (torch.optim.Optimizer): The optimizer for U-Net.
        fermi_params (dict): The parameters for Fermi model, including 'S',
                            'S_op', and 'osamp'.
        od_enable (bool, optional): Flag to enable outlier detection. Defaults
                                    to False.

    Returns:
        torch.Tensor: The self-supervised loss value.

    """
    # Extracting parameters    
    S = fermi_params['S']
    S_op = fermi_params['S_op']
    osamp = fermi_params['osamp']

    # Windowing based on the first-pass bolus window length 
    im_sig = im_sig[..., 0:wlen]
    aif = aif[..., 0:wlen]
    mbolus = mbolus[..., 0:wlen]
    ctc = ctc[..., 0:wlen]
    time = time[..., 0:wlen]

    # General initialization
    unet.zero_grad()
    unet.train()
    for p in unet.parameters():
        p.requires_grad = True

    # Estimate perfusion parameters
    # im_sig, _ = anomaly_inject(im_sig)        
    im_sig = ssup_mask(im_sig, indx_nn[:, np.newaxis], seg, mbolus)
    eta_gen = unet(im_sig.unsqueeze(1),
                   seg,
                   aif=aif,
                   ctc=ctc,
                   time=time,
                   indx_dc=indx_dc)

    # Segmenting curves
    aif_2D_dyn = (aif.unsqueeze(1).unsqueeze(1)
                  * torch.ones(ctc.shape, device=ctc.device))
    aif_seg = aif_2D_dyn
    ctc_seg = ctc

    # Compensating offset in the time curves
    oTp = 5
    aif = aif_seg-aif_seg[..., 0:oTp].mean(-1, keepdim=True)
    ctc = ctc_seg-ctc_seg[..., 0:oTp].mean(-1, keepdim=True)

    # Oversampling curves (Linear)
    time = (time-time[:, 0])
    time_osamp = interp_linear_1D(time, size=osamp * time.shape[-1])/S
    aif_osamp = interp_linear_1D(aif, size=osamp * aif.shape[-1])

    # Estimating concentration curves
    neg_shift = 2*osamp
    fermi_ir = fermi_ir_func(S_op * eta_gen,
                             time_osamp.squeeze(),
                             C=500,
                             neg_shift=neg_shift)
    ctc_est = convolve(aif_osamp,
                       fermi_ir,
                       neg_shift=neg_shift)[..., ::osamp]/osamp

    # Modified Z-score outlier rejection
    if od_enable is True:
        temporal_frames_error = torch.norm(
            (ctc-ctc_est)[seg == 1], dim=0
            ).detach()
        deviation = temporal_frames_error - temporal_frames_error.median()
        zmod = (
            (0.6745*deviation) / deviation.abs().median()
            )
        indx_nn_od, _ = ModZ_OD(zmod, indx_nn)
    else:
        indx_nn_od = indx_nn

    ctc_est = ctc_est[..., indx_nn_od]
    ctc = ctc[..., indx_nn_od]

    # Self-supervised loss
    C_ssup = torch.sum(ctc)**2
    ssup_loss = torch.sum((ctc - ctc_est)**2)/C_ssup

    # Updating the network
    ssup_loss.backward()
    unet_optmzr.step()

    return ssup_loss


def unet_pretrain(im_sig,
                  aif,
                  ctc,
                  seg,
                  time,
                  wlen,
                  indx_nn,
                  eta_pretrain,
                  mbolus,
                  unet,
                  unet_optmzr,
                  iter):
    """
    Pre-train exclusively the deep-learning prior of DeepFermi without
    engaging the dataconsistency layer before employing the model-based
    self-supervised learning. This is achieved by minimizing the MSE
    between the predicted perfusion parameters (eta_gen) and
    the ground truth labels (eta_pretrain), the latter of which is
    obtained through traditional Fermi deconvolution approach using LBFGS.
    The pre-training is performed to speed-up the training process and enhance
    training stability.

    Parameters:
        im_sig (torch.Tensor): The input signal image (e.g., MRI images).
        aif (torch.Tensor): The arterial input function (AIF).
        ctc (torch.Tensor): The concentration-time curve (CTC).
        seg (torch.Tensor): The segmentation mask for myocardium.
        time (torch.Tensor): The time points for the concentration curves.
        wlen (int): The length of the first pass of the main bolus window.
        indx_nn (torch.Tensor): Indices for the neural network training.
        eta_pretrain (torch.Tensor): The ground truth perfusion parameters
                                     labels.
        mbolus (torch.Tensor): The bolus image used for masking.
        unet (nn.Module): The U-Net model.
        unet_optmzr (torch.optim.Optimizer): The optimizer for U-Net.
        iter (int): The number of iterations to run.

    Returns:
        torch.Tensor: The loss value for the pre-training step.

    """

    # Windowing
    im_sig = im_sig[..., 0:wlen]
    aif = aif[..., 0:wlen]
    mbolus = mbolus[..., 0:wlen]
    ctc = ctc[..., 0:wlen]
    time = time[..., 0:wlen]

    # General initialization
    unet.zero_grad()
    unet.train()
    for p in unet.parameters():
        p.requires_grad = True

    # Train Unet
    im_sig = ssup_mask(im_sig, indx_nn[:, np.newaxis], seg, mbolus)
    eta_gen = unet(im_sig.unsqueeze(1), seg, aif=aif, ctc=ctc, time=time)

    # mse loss function
    C = (eta_pretrain**2).sum(dim=(0, 2, 3), keepdim=True)
    loss = (((eta_pretrain - eta_gen)**2)/C).sum()

    # Updating the network
    loss.backward()
    unet_optmzr.step()

    return loss


def unet_eval(im_sig,
              aif,
              ctc,
              seg,
              time,
              wlen,
              unet,
              fermi_params):
    """
    Evaluate Deepfermi on given data by indirectly calculating the error
    between the measured concentration-time curve (CTC) and the estimated
    concentration-time curve based on the DeepFermi estimated perfusion
    parameters. This is because the direct estimation error cannot be
    calculated since the ground truth perfusion parameters are not
    accessible in real scenarios.

    Parameters:
        im_sig (torch.Tensor): The input signal image (e.g., DCE-MR Image).
        aif (torch.Tensor): The arterial input function (AIF).
        ctc (torch.Tensor): The concentration-time curve (CTC).
        seg (torch.Tensor): The segmentation mask for myocardium.
        time (torch.Tensor): The time points for the concentration curves.
        wlen (int): The length of the first pass of the main bolus window.
        unet (nn.Module): The U-Net model.
        fermi_params (dict): The parameters for the Fermi model, including 'S',
                             'S_op', and 'osamp'.

    Returns:
        torch.Tensor: The self-supervised loss value for the evaluation.

    """

    # Extracting parameters
    S = fermi_params['S']
    S_op = fermi_params['S_op']
    osamp = fermi_params['osamp']

    # Windowing
    im_sig = im_sig[..., 0:wlen]
    aif = aif[..., 0:wlen]
    ctc = ctc[..., 0:wlen]
    time = time[..., 0:wlen]

    # Estimate perfusion parameters
    eta_gen = unet(im_sig.unsqueeze(1), seg, aif=aif, ctc=ctc, time=time)

    # Segmenting curves
    aif_2D_dyn = (aif.unsqueeze(1).unsqueeze(1)
                  * torch.ones(ctc.shape, device=ctc.device))
    aif_seg = aif_2D_dyn[seg == 1]
    ctc_seg = ctc[seg == 1]

    # Compensating offset in the time curves
    oTp = 5
    aif = aif_seg-aif_seg[..., 0:oTp].mean(-1, keepdim=True)
    ctc = ctc_seg-ctc_seg[..., 0:oTp].mean(-1, keepdim=True)

    # Oversampling curves (Linear)
    time = (time-time[:, 0])
    time_osamp = interp_linear_1D(time, size=osamp*time.shape[-1])/S
    aif_osamp = interp_linear_1D(aif, size=osamp*aif_seg.shape[-1])

    # Self-supervised loss
    neg_shift = 2*osamp
    fermi_ir = fermi_ir_func(S_op * eta_gen, time_osamp.squeeze(),
                             C=500,
                             neg_shift=neg_shift)
    fermi_ir = fermi_ir[seg == 1]
    ctc_est = convolve(aif_osamp,
                       fermi_ir,
                       neg_shift=neg_shift)[..., ::osamp]/osamp
    C_ssup = torch.sum(ctc**2)
    ssup_loss = torch.sum((ctc - ctc_est)**2)/C_ssup

    return ssup_loss