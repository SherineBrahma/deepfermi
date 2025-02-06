import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from fermi import ModZ_OD, convolve, fermi_ir_func
from network.layers import ConvBlock, ConvLayer, InstanceNorm, Pooling
from torch.autograd.functional import hvp
from utils import interp_linear_1D


class DeepFermi(nn.Module):
    """
    DeepFermi is a model-based neural network module utilizing the fermi model
    to ensure data consistency for estimating perfusion parameters.

    Attributes:

        mode (str): The operation mode of the network, which can be
                    'pre_training', 'fine_tuning', or 'testing'.
        inorm_stage_1 (InstanceNorm): Instance normalization before the cnn.
        cnn (nn.Module): Convolutional neural network for feature extraction.
        inorm_stage_2 (InstanceNorm): Instance normalization after the feature
                                      extraction stage and before the parameter
                                      estimation module.
        synth_nn (SynthNet): Network module for parameter estimation.
        lambda_reg (nn.Parameter): Regularizer parameter in data-consistency
                                   module.
        S, S_op, SH_op: For scaling and descaling perfusion parameters, helpful
                        in pre-conditioning the non-linear problem.
        od_enable (bool): Flag indicating whether outlier detection is enabled.
        dc_module (FermiDataConsLBFGS): The data consistency module 
                                        during model-based (fine-tuning)
                                        training and testing phases.
    """

    def __init__(self,
                 cnn,
                 osamp=1,
                 max_iter_lbfgs=100,
                 max_eval_lbfgs=100,
                 mode='pre_training',
                 learn_lambda=True,
                 od_enable=False):
        super(DeepFermi, self).__init__()

        # Mode validation
        msg = (
            "mode has to be one of 'pre_training', 'fine_tuning' or 'testing"
            )
        assert mode in ['pre_training', 'fine_tuning', 'testing'], msg
        self.mode = mode

        # Neural network module
        # Initial feature extraction stage
        self.inorm_stage_1 = InstanceNorm(2, dim=3, affine=True)
        self.cnn = cnn
        # Parameter estimation stage
        self.inorm_stage_2 = InstanceNorm(36, dim=3, affine=True)
        self.synth_nn = SynthNet(dim=3,
                                 ncin=36,
                                 nfilters=48,
                                 ncout=3,
                                 nstage=3,
                                 nconv_stage=2,
                                 bias=False,
                                 groups=3)

        # Data-consistency module
        # Learned regularizer
        beta = 1.
        lambda_init = np.log(np.exp(beta)-1.)/beta
        self.lambda_reg = nn.Parameter(
            torch.tensor(
                5.35123*lambda_init,
                dtype=torch.float
                ),
            requires_grad=bool(learn_lambda)
            )
        # Fermi deconvolution configurations
        S = 10
        S_op = rearrange(torch.tensor([1, 1/S, S]),
                         'np -> 1 np 1 1')
        SH_op = rearrange(torch.tensor([1, S, 1/S]),
                          'np -> np 1 1')
        self.register_buffer('S', torch.tensor(S))
        self.register_buffer('S_op', S_op)
        self.register_buffer('SH_op', SH_op)
        self.register_buffer('osamp', torch.tensor(osamp))
        self.register_buffer('max_iter_lbfgs', torch.tensor(max_iter_lbfgs))
        self.register_buffer('max_eval_lbfgs', torch.tensor(max_eval_lbfgs))
        self.od_enable = od_enable
        self.dc_module = FermiDataConsLBFGS.apply

    def forward(self, xin, seg, aif=None, ctc=None, time=None, indx_dc=[]):

        assert xin.shape[0] == 1, "Only mb=1 supported"

        # Pre-processing to ensure input compatibility with parameter
        # estimation stage
        time = (time-time[:, 0])/self.S
        time_2D_dyn = (
            time.unsqueeze(1).unsqueeze(1)
            * torch.ones(ctc.shape, device=ctc.device)
            )
        aif_2D_dyn = (
            aif.unsqueeze(1).unsqueeze(1)
            * torch.ones(ctc.shape, device=ctc.device)
            )
        # Initializing LBFGS iterations
        self.lbfgs_iter = 0

        if self.mode == 'pre_training':

            # Apply neural networks
            seg_in = seg.unsqueeze(1).unsqueeze(-1).repeat(1,
                                                           1,
                                                           1,
                                                           1,
                                                           xin.shape[-1])
            xin = self.inorm_stage_1(
                torch.cat((xin, seg_in), 1)
                )
            xcnn = self.cnn(xin)
            aif_cnn = aif_2D_dyn.unsqueeze(1).repeat(1,
                                                     xcnn.shape[1],
                                                     1,
                                                     1,
                                                     1)
            time_cnn = time_2D_dyn.unsqueeze(1).repeat(1,
                                                       xcnn.shape[1],
                                                       1,
                                                       1,
                                                       1)
            xcnn = self.inorm_stage_2(
                torch.cat((aif_cnn, xcnn, time_cnn), 1)
                )
            eta_nn = self.synth_nn(xcnn)
            eta = eta_nn

            return eta

        elif self.mode in ['fine_tuning', 'testing']:

            # Pre-liminary check
            msg = (
                "Arterial input function and concentration time curve required"
                " for ensuring data-consistency!"
                )
            assert aif is not None and ctc is not None, msg

            # Apply neural networks
            seg_in = seg.unsqueeze(1).unsqueeze(-1).repeat(1,
                                                           1,
                                                           1,
                                                           1,
                                                           xin.shape[-1])
            xin = self.inorm_stage_1(
                torch.cat((xin, seg_in), 1)
                )
            xcnn = self.cnn(xin)
            aif_cnn = aif_2D_dyn.unsqueeze(1).repeat(1,
                                                     xcnn.shape[1],
                                                     1,
                                                     1,
                                                     1)
            time_cnn = time_2D_dyn.unsqueeze(1).repeat(1,
                                                       xcnn.shape[1],
                                                       1,
                                                       1,
                                                       1)
            xcnn = self.inorm_stage_2(
                torch.cat((aif_cnn, xcnn, time_cnn), 1)
                )
            eta_nn = self.synth_nn(xcnn)

            # Apply data-consistency layer
            eta_nn = self.S_op * eta_nn
            eta_pi, self.lbfgs_iter = self.dc_module(ctc, 
                                                     aif_2D_dyn,
                                                     time,
                                                     seg,
                                                     self.osamp,
                                                     indx_dc,
                                                     self.lambda_reg,
                                                     eta_nn,
                                                     self.max_iter_lbfgs,
                                                     self.max_eval_lbfgs,
                                                     self.od_enable)
            eta = self.SH_op * eta_pi

            return eta


class SynthNet(nn.Module):
    """
    SynthNet is a network module used for estimating perfusion parameters from
    features extracted by the initial CNN stage. This process is facilitated by
    convolutional blocks, pooling layers along the time axis, and finally an
    adaptive average-pooling operation to ultimately obtain 2D parameter maps
    from 3D input.
    """

    def __init__(self,
                 dim=2,
                 ncin=2,
                 nfilters=2,
                 ncout=2,
                 nstage=3,
                 nconv_stage=2,
                 bias=False,
                 groups=1):
        super(SynthNet, self).__init__()

        # General Initializations
        dsamp_fact = 2
        assert dim in [1, 2, 3], "Only dim values of 1, 2, or 3 allowed."
        if dim == 1:
            pool_kshape = (2)
            img_dim_out = (5)
            shape = (3)
            pad = (1)
        elif dim == 2:
            pool_kshape = (1, 2)
            img_dim_out = (None, 5)
            shape = (1, 3)
            pad = (0, 1)
        elif dim == 3:
            pool_kshape = (1, 1, 2)
            img_dim_out = (None, None, 1)
            shape = (1, 1, 3)
            pad = (0, 0, 1)
        else:
            pool_kshape = None
            shape = None
            pad = None

        # Constructing network
        self.synth_net = nn.ModuleList()
        self.synth_net.append(ConvBlock(dim=dim,
                                        shape=shape,
                                        nch=ncin,
                                        nfilters=nfilters,
                                        nconvs=nconv_stage,
                                        pad=pad,
                                        bias=bias,
                                        res_connect=True))
        nch = nfilters
        ncout_layer = nfilters // dsamp_fact
        for _ in range(nstage):
            self.synth_net.append(ConvBlock(dim=dim,
                                            shape=shape,
                                            nch=nch,
                                            nfilters=nfilters,
                                            ncout=ncout_layer,
                                            nconvs=nconv_stage,
                                            pad=pad, bias=bias,
                                            groups=groups,
                                            res_connect=True))
            self.synth_net.append(Pooling(dim=dim,
                                          kernel_size=pool_kshape,
                                          pooling_type="Max"))
            nch = ncout_layer
            nfilters = ncout_layer
            ncout_layer = nfilters // dsamp_fact

        if dim == 1:
            self.synth_net.append(nn.AdaptiveAvgPool1d(img_dim_out))
        elif dim == 2:
            self.synth_net.append(nn.AdaptiveAvgPool2d(img_dim_out))
        elif dim == 3:
            self.synth_net.append(nn.AdaptiveAvgPool3d(img_dim_out))

        self.synth_net.append(ConvBlock(dim=dim,
                                        shape=shape,
                                        nch=nch,
                                        nfilters=ncout,
                                        ncout=ncout,
                                        nconvs=nconv_stage,
                                        pad=pad,
                                        bias=bias,
                                        groups=groups,
                                        res_connect=True))
        self.synth_net.append(ConvLayer(dim=dim,
                                        shape=1,
                                        nch=ncout,
                                        nfilters=ncout,
                                        pad=0,
                                        bias=bias,
                                        groups=groups))

    def __call__(self, xin):

        # Apply layer modules
        xconv = xin
        for i in range(len(self.synth_net)):
            xconv = self.synth_net[i](xconv)

        # Output
        xout = xconv.squeeze(-1)

        return xout


class FermiDataConsLBFGS(torch.autograd.Function):
    """
    FermiDataConsLBFGS implements both the forward and backward passes for
    fermi-based data consistency optimization using LBFGS.

    The forward pass applies a physics-informed optimization process for
    perfusion parameters using concentration-time curves (CTC), arterial
    input function (AIF), and time information, while the backward pass
    computes the gradients for backpropagation during training.
    """

    @staticmethod
    def forward(ctx,
                ctc,
                aif,
                time,
                seg,
                osamp,
                indx_dc,
                lambda_reg,
                eta_nn,
                max_iter,
                max_eval,
                od_enable):

        # Compensating offset in the time curves
        oTp = 5
        aif = aif-aif[..., 0:oTp].mean(-1, keepdim=True)
        ctc = ctc-ctc[..., 0:oTp].mean(-1, keepdim=True)

        # Oversampling curves (Linear)
        aif_osamp = interp_linear_1D(aif, size=osamp*aif.shape[-1])
        ctc_osamp = interp_linear_1D(ctc, size=osamp*ctc.shape[-1])
        time_osamp = interp_linear_1D(time, size=osamp*time.shape[-1])

        # Initializing data-consistency objective
        eta_prior = eta_nn.detach().clone()
        lambda_reg = lambda_reg.detach().clone()
        F_Op = FermiDConsObj(eta_prior,
                             ctc_osamp,
                             aif_osamp,
                             time_osamp,
                             seg,
                             osamp,
                             indx_dc,
                             lambda_reg)

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
        lbfgs = optim.LBFGS([eta_pi],
                            lr=1,
                            history_size=10,
                            max_iter=max_iter,
                            max_eval=max_eval,
                            line_search_fn="strong_wolfe")

        # LBFGS Execution
        indx_total = torch.arange(time.shape[-1])
        prev_mask_od = torch.ones(indx_total.__len__())
        od_iter = 0
        od_max_iter = 5
        terminate = False
        while terminate is False:

            # LBFGS optimzation steps
            lbfgs.step(closure)

            # Outlier detection step
            F_Op.indx_dc_od, mask_od = ModZ_OD(F_Op.zmod, F_Op.indx_dc)
            terminate = (
                (mask_od-prev_mask_od).norm().item() == 0
                or od_iter > od_max_iter
                or od_enable is not True
                )
            prev_mask_od = mask_od.clone()
            od_iter += 1

        n_iter = lbfgs.n_iter
        # Saving objects required for backward
        ctx._the_function = F_Op
        ctx.save_for_backward(eta_pi, lambda_reg, eta_prior)

        return eta_pi, n_iter

    @staticmethod
    def backward(ctx, grad_out, _):

        # Loading objects saved during forward
        eta_pi, lambda_reg, eta_prior = ctx.saved_tensors
        F_Op = ctx._the_function          
        g = FermiDConsConjGrad(F_Op, eta_pi, grad_out, grad_out, niter=10)

        # Initializing the gradients
        grad_ctc = None
        grad_aif = None
        grad_time = None
        grad_seg = None
        grad_osamp = None
        grad_indx_dc = None
        grad_max_iter = None
        grad_max_eval = None
        grad_od_enable = None

        # Computing gradients to be backpropagated
        # pylint: disable=not-callable
        C_nn = (eta_prior**2).sum(dim=(0, 2, 3), keepdim=True)
        grad_eta_prior = g * (2/C_nn) * F.softplus(lambda_reg)
        grad_lambda_reg = -(
            g * (2/C_nn)
            * torch.sigmoid(lambda_reg)
            * (eta_pi-eta_prior)
            ).sum()

        return (grad_ctc,
                grad_aif,
                grad_time,
                grad_seg,
                grad_osamp,
                grad_indx_dc,
                grad_lambda_reg,
                grad_eta_prior,
                grad_max_iter,
                grad_max_eval,
                grad_od_enable)


class FermiDConsObj(nn.Module):
    """
    FermiDConsObj computes the objective function used in the data
    consistency module for optimizing perfusion parameters.
    """

    def __init__(self,
                 eta_prior,
                 ctc_osamp,
                 aif_osamp,
                 time_osamp,
                 seg, osamp,
                 indx_dc,
                 lambda_reg):
        super(FermiDConsObj, self).__init__()

        # General Initializations
        self.eta_prior = eta_prior
        self.ctc_osamp = ctc_osamp
        self.aif_osamp = aif_osamp
        self.time_osamp = time_osamp
        self.seg = seg
        self.osamp = osamp.item()
        self.indx_dc_od = self.indx_dc = (
            torch.arange(ctc_osamp[..., ::osamp].shape[-1])
            if indx_dc == [] else indx_dc
            )
        self.lambda_reg = lambda_reg
        self.prev_iter = 0
        self.n_iter = 0

    def __call__(self, eta_pi):

        # Negative shift allowed
        neg_shift = 2*self.osamp

        # Calculating and segmenting fermi impulse response
        fermi_ir = fermi_ir_func(eta_pi,
                                 self.time_osamp.squeeze(),
                                 C=500,
                                 neg_shift=neg_shift)

        # Convolution operation
        ctc_est = convolve(self.aif_osamp,
                           fermi_ir,
                           neg_shift=neg_shift)[..., ::self.osamp]/self.osamp
        ctc_dc = self.ctc_osamp[..., ::self.osamp]

        # Modified z-score calculation
        tframes_err = torch.norm(
            (ctc_dc-ctc_est)[self.seg == 1], dim=0
            ).detach()
        self.zmod = (
            (0.6745*(tframes_err - tframes_err.median()))
            / (tframes_err-tframes_err.median()).abs().median()
            )

        # Rejecting outliers
        ctc_est = ctc_est[..., self.indx_dc_od]
        ctc_dc = ctc_dc[..., self.indx_dc_od]

        # Constructing objective
        # pylint: disable=not-callable
        C_nn = (self.eta_prior**2).sum(dim=(0, 2, 3), keepdim=True)
        C_dc = (ctc_dc**2).sum()
        objective = (((ctc_dc - ctc_est)**2/C_dc).sum()
                     + (F.softplus(self.lambda_reg)
                        * (((eta_pi - self.eta_prior)**2)/C_nn).sum())
                     + (F.relu(-eta_pi)**2).sum())

        return objective


def FermiDConsConjGrad(F_Op, eta_pi, g, grad_in, niter=5):
    """
    Performs conjugate gradient optimization during the backward pass of
    calculating the Fermi data-consistency objective.
    """

    # g is the starting value, grad_in the rhs;
    _, r = hvp(F_Op, eta_pi, v=g)
    r = grad_in-r

    # Initialize p
    p = r.clone()

    # Old squared norm of residual
    sqnorm_r_old = torch.bmm(
        r.flatten(start_dim=1).unsqueeze(1),
        r.flatten(start_dim=1).unsqueeze(-1)
        )

    for _ in range(niter):

        # Calculate Hp;
        _, d = hvp(F_Op, eta_pi, v=p)

        # Calculate step size alpha;
        inner_p_d = torch.bmm(
            p.flatten(start_dim=1).unsqueeze(1),
            d.flatten(start_dim=1).unsqueeze(-1)
            )
        alpha = (sqnorm_r_old / inner_p_d).unsqueeze(-1)

        # Perform step and calculate new residual;
        g = g + alpha*p
        r = r - alpha*d

        # new residual norm
        sqnorm_r_new = torch.bmm(
            r.flatten(start_dim=1).unsqueeze(1),
            r.flatten(start_dim=1).unsqueeze(-1)
            )
        # calculate beta and update the norm;
        beta = (sqnorm_r_new / sqnorm_r_old).unsqueeze(-1)
        sqnorm_r_old = sqnorm_r_new
        p = r + beta*p

    if torch.any(torch.isnan(g)) is True:
        print('NaN detected: Skipping optimization layer backpropagation')
        g = torch.zeros(g.shape, device=g.device)

    return g
