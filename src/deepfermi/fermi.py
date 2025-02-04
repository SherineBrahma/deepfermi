
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from utils import interp_linear_1D


def translate_ir_func(delay, t, t0=0, C=1000, neg_shift=0):
    """
    Approximates a delta impulse response function using a Gaussian function,
    translated by a delay parameter.
    """

    t = t-t0
    t = torch.cat((-torch.flip(t, [0])[-neg_shift-1:-1], t))    
    delayed_dirac = t-delay
    C = C/torch.pi
    translate_ir = np.sqrt(C/np.pi)*torch.exp(-C*(delayed_dirac)**2)    
    translate_ir = translate_ir/translate_ir.max()

    return translate_ir


def ModZ_OD(score, indx, thres=3.5):
    """
    Detects outliers based on a modified z-score threshold and generates a
    mask for outlier detection.
    """

    # Detecting outliers
    indx_outlier = (score > thres).nonzero().squeeze()
    if indx_outlier.numel() != 0:
        indx_od = torch.tensor(
            [i.item() for i in indx if i not in indx_outlier]
            )
    else:
        indx_od = indx

    # Generating mask
    mask_od = torch.ones(score.__len__())
    mask_od[indx_outlier] = 0

    return indx_od, mask_od


def fermi_ir_func(eta, t, t0=0, C=100, neg_shift=0, p=0.9999):
    """
    Calculates the Fermi impulse response function based on the given
    perfusion parameters. The negative shift, specified also specified
    in the convolution function, is used to adjust the zero time reference.
    """

    one = torch.ones(eta[:, 0, ...].shape, device=eta.device)
    t = t-t0
    t = torch.cat((-torch.flip(t, [0])[-neg_shift-1:-1], t))
    t_len = t.shape[0]
    t = t * torch.repeat_interleave(one.unsqueeze(-1), t_len, dim=-1)
    flow_rate = eta[:, 0, ...].unsqueeze(-1)
    delay = eta[:, 1, ...].unsqueeze(-1)
    decay_rate = eta[:, 2, ...].unsqueeze(-1)
    delay_fermi = t-delay 
    delayed_heavy = t-delay
    heavy_side = torch.sigmoid(C*(delayed_heavy))
    output = flow_rate*(1/(torch.exp((delay_fermi)*decay_rate)+1)) * heavy_side

    return output


def convolve(xin, im_res, neg_shift=0):
    """
    Performs convolution of the input with the impulse response in the
    Fourier domain.

    Negative shift padding is applied to handle non-causal time translations
    of the impulse response. This flexibility is useful during optimization,
    as it prevents the start of the Fermi function from moving out of the
    frame, which would otherwise misestimate the delay parameters and,
    consequently, other parameters.
    """

    # pylint: disable=not-callable
    # output =  im_res ⨂ xin
    out_size = xin.shape[-1]
    # Pad input
    xin_pad = F.pad(xin, ((0, neg_shift + out_size)))
    im_res_pad = F.pad(im_res, ((0, out_size)))
    # Apply FFT
    xin_fft = torch.fft.rfft(xin_pad, dim=-1)
    im_res_fft = torch.fft.rfft(im_res_pad, dim=-1)
    # Convolution in fourier-domain
    output = im_res_fft*xin_fft
    # Apply IFFT
    output = torch.fft.irfft(output, dim=-1)[..., neg_shift:neg_shift+out_size]

    return output


def convolve_direct(xin, im_res, neg_shift=0):
    """
    Performs convolution of the input with the impulse response in the
    time domain.

    Negative shift padding is applied to handle non-causal time translations
    of the impulse response. This flexibility is useful during optimization,
    as it prevents the start of the Fermi function from moving out of the
    frame, which would otherwise misestimate the delay parameters and,
    consequently, other parameters.
    """

    # pylint: disable=not-callable
    # # output =  im_res ⨂ input
    t_len = xin.shape[-1]
    output = torch.zeros(xin.shape, dtype=xin.dtype, device=xin.device)
    xin = F.pad(xin, ((0, neg_shift)))
    im_res_flip = torch.flip(im_res, [-1])
    for t_indx in range(t_len):
        output[..., t_indx] = torch.sum(
            im_res_flip[..., -(t_indx+1+neg_shift):]
            * xin[..., :(t_indx+1 + neg_shift)],
            dim=-1
            )

    return output


class FermiLBFGSSolver(nn.Module):
    """
    Solves the Fermi model using the LBFGS optimizer to fit the perfusion
    parameters given a set of concentration-time curves (CTC),
    arterial input functions (AIF), and segmentation. It applies oversampling
    to the input curves and uses the modified z-score for outlier detection
    during optimization.
    """
    
    def __init__(self, osamp=1, od_enable=False):

        super(FermiLBFGSSolver, self).__init__()

        # Initializations
        S = 10
        S_op = rearrange(torch.tensor([1, 1/S, S]),
                         'np -> 1 np 1 1')
        SH_op = rearrange(torch.tensor([1, S, 1/S]),
                          'np -> np 1 1')
        self.register_buffer('S', torch.tensor(S))
        self.register_buffer('S_op', S_op)
        self.register_buffer('SH_op', SH_op)
        self.osamp = osamp
        self.od_enable = od_enable

    def forward(self, eta_init, ctc, aif, seg, time, indx_lbfgs=[]):

        # General initialization
        time = (time-time[0])/self.S
        time_osamp = interp_linear_1D(
            time.unsqueeze(0), size=self.osamp*time.shape[-1]
            )[0]

        # Compensating offset in the time curves
        oTp = 5
        aif = aif-aif[..., 0:oTp].mean(-1, keepdim=True)
        ctc = ctc-ctc[..., 0:oTp].mean(-1, keepdim=True)

        # Oversampling curves (Linear)
        aif_osamp = interp_linear_1D(aif, size=self.osamp*aif.shape[-1])

        # lbfgs optimizer initialization
        global indx_lbfgs_od
        indx_lbfgs_od = indx_lbfgs = torch.arange(
            time.shape[-1]
            ) if indx_lbfgs == [] else indx_lbfgs
        eta_lbfgs = self.S_op * eta_init
        eta_lbfgs.requires_grad = True
        lbfgs = optim.LBFGS([eta_lbfgs],
                            lr=1.0,
                            history_size=10,
                            max_eval=1000,
                            max_iter=1000,
                            line_search_fn="strong_wolfe")

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
            fermi_ir = fermi_ir_func(eta_lbfgs,
                                     time_osamp,
                                     C=500,
                                     neg_shift=neg_shift)

            # Convolution
            ctc_est = convolve(
                aif_osamp,
                fermi_ir,
                neg_shift=neg_shift
                )[..., ::self.osamp]/self.osamp

            # Modified z-score calculation
            tframes_err = torch.norm((ctc-ctc_est)[seg == 1], dim=0)
            zmod = (
                0.6745*(tframes_err - tframes_err.median())
                )/(tframes_err - tframes_err.median()).abs().median()

            # Saving vectors for debugging
            ctc_est_db = ctc_est.clone()
            fermi_ir_db = fermi_ir.clone()

            # Loss function
            ctc_est = ctc_est[..., indx_lbfgs_od]
            ctc_lbfgs = ctc[..., indx_lbfgs_od]
            C_mse = torch.sum(ctc_lbfgs**2)
            objective = (
                torch.sum(((ctc_lbfgs - ctc_est))**2)/C_mse
                + (F.relu(-eta_lbfgs)**2).sum())
            objective.backward(retain_graph=True)

            return objective

        # LBFGS Execution
        prev_mask_od = mask_od = torch.ones(indx_lbfgs.__len__())
        od_iter = 0
        od_max_iter = 3
        terminate = False
        while terminate is False:
            # LBFGS optimzation steps
            lbfgs.step(closure)
            if self.od_enable is True:
                indx_lbfgs_od, mask_od = ModZ_OD(zmod, indx_lbfgs)
                # Termination condition
                terminate = ((mask_od-prev_mask_od).norm().item() == 0
                             or od_iter > od_max_iter
                             or self.od_enable is not True)
                prev_mask_od = mask_od.clone()
                # Increment outlier detection iteration
                od_iter += 1
            else:
                terminate = True

        eta = self.SH_op * eta_lbfgs

        return eta, mask_od


def shift_aif(aif, mbolus, time, osamp):
    """
    Shifts the arterial input function (AIF) to align with the bolus arrival
    time. This function estimates the time delay between the AIF and the
    bolus by optimizing the delay parameter using the L-BFGS method. The AIF
    is then shifted accordingly, and the resulting corrected AIF is returned.
    The function applies linear interpolation and oversampling to improve the
    precision of the time series.
    """

    # Estimating delay for the aif
    # Initialization
    device = 'cuda'
    S = 10
    neg_shift = 20
    time_t0 = time[0]/S
    time = time/S
    time_osamp = interp_linear_1D(
        time.unsqueeze(0), size=osamp*time.shape[-1]
        )[0]
    aifPreCorrection = aif

    # Compensating offset in the time curves
    oTp = 5
    mbolus = mbolus-mbolus[..., :oTp].mean(-1, keepdim=True)
    Comp_C = aifPreCorrection.max()/mbolus.max()
    mbolus = Comp_C * mbolus

    # Oversampling curves (Linear)
    aifPreCorrection_osamp = interp_linear_1D(
        aifPreCorrection,
        size=osamp * aifPreCorrection.shape[-1]
        )
    mbolus_osamp = interp_linear_1D(
        mbolus,
        size=osamp*mbolus.shape[-1]
        )

    # For loss mask
    global index_array
    index_array = torch.arange(mbolus_osamp.shape[-1], device=device)

    # lbfgs optimizer initialization
    avg_time_step_size = (
        time-torch.roll(time, shifts=(1), dims=(0))
        )[1:].mean()
    global delay_init
    delay_init = (mbolus.argmax()-aif.argmax())*avg_time_step_size + 0.001
    delay_lbfgs = 1/S * torch.tensor(delay_init, device=device)
    delay_lbfgs.requires_grad = True 
    lbfgs = optim.LBFGS([delay_lbfgs],
                        lr=1,
                        history_size=10,
                        max_eval=100,
                        max_iter=100,
                        line_search_fn="strong_wolfe")

    def closure():
        # Initializations
        global aif_est
        global shift_ir
        global delay_init

        # Start optimization
        lbfgs.zero_grad()
        shift_ir = translate_ir_func(delay_lbfgs,
                                     time_osamp,
                                     time_t0,
                                     C=500,
                                     neg_shift=neg_shift*osamp)

        # Fail safe mechanism incase delay gets undefined
        if not torch.any(torch.isnan(delay_lbfgs)):
            shift_ir = translate_ir_func(delay_lbfgs,
                                         time_osamp,
                                         time_t0,
                                         C=500,
                                         neg_shift=neg_shift*osamp)
            shift_ir = shift_ir.squeeze(0).squeeze(0)
            aif_est = convolve(aifPreCorrection_osamp,
                               shift_ir,
                               neg_shift=neg_shift * osamp)
        else:
            print(
                'NaN detected: Setting initial delay shift value and exiting.'
                )
            shift_ir = translate_ir_func(delay_init,
                                         time_osamp,
                                         time_t0,
                                         C=500,
                                         neg_shift=neg_shift * osamp)
            shift_ir = shift_ir.squeeze(0).squeeze(0)
            aif_est = convolve(aifPreCorrection_osamp,
                               shift_ir,
                               neg_shift=neg_shift * osamp)
            print(delay_lbfgs)
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
        objective = (
            torch.sum(((m - a))**2)/(m**2).sum()
            + (F.relu((m[:, 0:m_peak_index] - a[:, 0:m_peak_index]))**2).sum()
            )
        objective.backward(retain_graph=True)

        return objective

    lbfgs.step(closure)

    # Correcting AIF delay
    del_delay = (
        delay_lbfgs
        if not torch.any(torch.isnan(delay_lbfgs))
        else delay_init
        )
    shift_offset = 0
    shifts = int(torch.round(del_delay/avg_time_step_size)) + shift_offset
    aifCorrected = torch.roll(aifPreCorrection[0, :],
                              shifts=(shifts), dims=0)

    # Exclude rolled over values
    if shifts >= 0:
        aifCorrected[:shifts] = 0
    else:
        aifCorrected[shifts:] = 0

    return aifCorrected


def ssup_mask(im_sig, mask_indx, seg, mbolus):
    """
    Applies a self-supervised learning mask to the input signal by perturbing
    the values, modeled as the spillage of the main bolus signal into the
    segmented myocardial regions of the signal intensity image.
    """

    # Generating mask
    mask = torch.ones((mask_indx.shape[0], 1, 1, 1), device=im_sig.device)

    # Masking
    nb, _, _, nt = im_sig.shape
    im_sig_indx = np.arange(0, nb)

    # Main bolus framewise-perturbation
    mbolus_2d_dyn = seg.unsqueeze(-1) * mbolus.unsqueeze(1).unsqueeze(1)
    perturb_mask = torch.zeros((nb, 1, 1, nt), device=im_sig.device)
    mbolus_probability = 0.3
    perturb_mask[im_sig_indx, ..., mask_indx] = torch.bernoulli(
        mbolus_probability
        * torch.ones((mask_indx.shape[0], im_sig.shape[0], 1, 1),
                     device=im_sig.device)
                     )
    perturbation = (
        (perturb_mask * ((1-seg.unsqueeze(-1)) * im_sig + mbolus_2d_dyn))
        + ((1-perturb_mask) * (
            im_sig + torch.normal(0, 0.1 * im_sig.std(-1)).unsqueeze(-1)
            ))
        )
    im_sig[im_sig_indx, ..., mask_indx] = (
        (mask * (perturbation[im_sig_indx, ..., mask_indx]))
        + ((1-mask) * im_sig[im_sig_indx, ..., mask_indx])
        )

    return im_sig
