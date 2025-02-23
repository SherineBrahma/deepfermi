from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import get_subplot


@dataclass
class Tracker:
    """
    For trackings and storing training and validation metrics during the
    progression of the training process. It includes losses, LBFGS iterations,
    and lambda regularization values. It also provides methods to  update and
    save training progress, visualize plots, and store model parameters.
    """

    # General
    save_path: str
    itr: List[int] | List[None] = field(default_factory=lambda: [])

    # Training
    ssup_loss_train: List[float] | List[None] = field(
        default_factory=lambda: []
        )
    avg_lbfgs_iter_train: List[float] | List[None] = field(
        default_factory=lambda: []
        )

    # Validation
    ssup_loss_val: List[float] | List[None] = field(
        default_factory=lambda: []
        )
    avg_lbfgs_iter_val: List[float] | List[None] = field(
        default_factory=lambda: []
        )

    # Lambda Regularization
    lambda_reg: List[float] | List[None] = field(
        default_factory=lambda: []
        )

    # Past plot number tracker
    mno: int = 0

    def __post_init__(self):

        # Make directories if does not exist to store results
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        unet_folder = Path.joinpath(Path(self.save_path), 'model/unet')
        Path(unet_folder).mkdir(parents=True, exist_ok=True)
    
    def update_and_save(self, itr,
                        ssup_loss_train,
                        avg_lbfgs_iter_train,
                        ssup_loss_val,
                        avg_lbfgs_iter_val,
                        unet):

        # Append values
        self.itr.append(itr)
        self.ssup_loss_train.append(ssup_loss_train.item())
        self.avg_lbfgs_iter_train.append(avg_lbfgs_iter_train.item())
        self.ssup_loss_val.append(ssup_loss_val.item())
        self.avg_lbfgs_iter_val.append(avg_lbfgs_iter_val.item())
        self.lambda_reg.append(unet.lambda_reg.item())

        # Save vectors       
        np.save(Path.joinpath(Path(self.save_path), 'it_vect'),
                np.array(self.itr))
        np.save(Path.joinpath(Path(self.save_path), 'ssup_loss_train'),
                np.array(self.ssup_loss_train))
        np.save(Path.joinpath(Path(self.save_path), 'avg_lbfgs_iter_train'),
                np.array(self.avg_lbfgs_iter_train))
        np.save(Path.joinpath(Path(self.save_path), 'ssup_loss_val'),
                np.array(self.ssup_loss_val))
        np.save(Path.joinpath(Path(self.save_path), 'avg_lbfgs_iter_val'),
                np.array(self.avg_lbfgs_iter_val))
        np.save(Path.joinpath(Path(self.save_path), 'lambda_reg'),
                np.array(self.lambda_reg))

        # Save models
        # unet
        unet_name = 'unet_iter_' + str(itr)
        save_path = Path.joinpath(Path(self.save_path), 'model/unet')
        torch.save(unet.state_dict(),
                   Path.joinpath(save_path, unet_name))
        torch.save(unet.state_dict(),
                   Path.joinpath(Path(self.save_path), 'unet'))
        
    def save_ssup_plot(self):
        
        # Maximum limit for y-axis
        max_lim = np.maximum(np.array(self.ssup_loss_train).max(),
                             np.array(self.ssup_loss_val).max())
        if max_lim >= 1.0:
            max_lim = 1.0
        else:
            max_lim = None
        # Plotting SSUP Curves
        fig = plt.figure()
        fig.suptitle('Self-supervised Loss curves')
        plt.plot(np.array(self.itr),
                 np.array(self.ssup_loss_train),
                 label="training",
                 linewidth=1)
        plt.plot(np.array(self.itr),
                 np.array(self.ssup_loss_val),
                 '--', label="validation",
                 linewidth=1)
        plt.xlabel('Iterations')
        plt.ylabel('Self-supervised Loss')
        plt.legend(loc="upper right")    
        plt.ylim(bottom=0, top=max_lim)
        save_path = Path.joinpath(Path(self.save_path),
                                  'ssup_loss_curves_fig.png')
        fig.savefig(save_path, dpi=500)
        plt.close()
        
    def save_lambda_reg_plot(self):        
        # Plotting Lambda Regularization Curves
        fig = plt.figure()
        fig.suptitle('Lambda Regularization')
        plt.plot(np.array(self.itr),
                 np.array(self.lambda_reg),
                 label="lambda_reg",
                 linewidth=1)
        plt.xlabel('Iterations')
        plt.ylabel('Lambda')
        plt.legend(loc="upper right")
        save_path = Path.joinpath(Path(self.save_path),
                                  'lambda_reg_fig.png')
        fig.savefig(save_path, dpi=500)
        plt.close()
        
    def save_avg_lbfgs_iter_plot(self):        
        # Plotting SSUP Curves
        fig = plt.figure()
        fig.suptitle('Average LBFGS iterations')
        plt.plot(np.array(self.itr),
                 np.array(self.avg_lbfgs_iter_train),
                 label="training",
                 linewidth=1)
        plt.plot(np.array(self.itr),
                 np.array(self.avg_lbfgs_iter_val),
                 '--',
                 label="validation",
                 linewidth=1)
        plt.xlabel('Iterations')
        plt.ylabel('Average iterations')
        plt.legend(loc="upper right")
        # plt.ylim(bottom=0, top=1.0)
        save_path = Path.joinpath(Path(self.save_path),
                                  'avg_lbfgs_iter_curves_fig.png')
        fig.savefig(save_path, dpi=500)
        plt.close()

    def save_samp_plot(self, dataset, unet):        
        # Randomly selecting indices        
        slc_indx = random.randint(0, dataset.__len__()-1)        
        # Slice selection
        im_sig = dataset.im_sig[slc_indx:slc_indx+1, :].cuda()
        aif = dataset.aif[slc_indx:slc_indx+1, :].cuda()
        ctc = dataset.ctc[slc_indx:slc_indx+1, :].cuda()
        seg = dataset.seg[slc_indx:slc_indx+1, :].cuda()
        time = dataset.time[slc_indx:slc_indx+1, :].cuda()
        wlen = dataset.wlen[slc_indx:slc_indx+1].cuda()
        eta_pretrain = dataset.eta_pretrain[slc_indx:slc_indx+1, :].cuda()
        
        # Pre-processing input
        aif = aif[..., 0:wlen]
        im_sig = im_sig[..., 0:wlen].unsqueeze(1)
        ctc = ctc[..., 0:wlen]
        time = time[..., 0:wlen]

        # Estimate eta
        eta_gen = unet(im_sig, seg, aif=aif, ctc=ctc, time=time)

        # Discarding eta background values
        seg_bkg = seg.unsqueeze(1).repeat(1, 3, 1, 1)
        eta_gen[seg_bkg == 0] = 0

        # Parameter maps visualization
        # LBFGS eta solution from svd approximated ctc
        eta_pretrain[seg_bkg == 0] = 0
        figsize = (8, 6)
        plot_list = [60 * eta_pretrain[0, 0],
                     eta_pretrain[0, 1],
                     eta_pretrain[0, 2],
                     60 * eta_gen[0, 0],
                     eta_gen[0, 1],
                     eta_gen[0, 2]]
        title_list = ["F SVD", "Tau SVD", "k SVD", "F EST", "Tau EST", "k EST"]
        sig_digit = 1
        flow_max = round(
            60 * (eta_pretrain[0, 0][seg[0] == 1].mean()
                  + 3*eta_pretrain[0, 0][seg[0] == 1].std()).cpu().numpy(),
            sig_digit)
        delay_max = round(
            1 * (eta_pretrain[0, 1][seg[0] == 1].mean()
                 + 3*eta_pretrain[0, 1][seg[0] == 1].std()).cpu().numpy(),
            sig_digit)
        decay_max = round(
            1 * (eta_pretrain[0, 2][seg[0] == 1].mean()
                 + 3*eta_pretrain[0, 2][seg[0] == 1].std()).cpu().numpy(),
            sig_digit)
        range_list = [(0, flow_max),
                      (0, delay_max),
                      (0, decay_max),
                      (0, flow_max),
                      (0, delay_max),
                      (0, decay_max)] 
        cmap_list = ['viridis',
                     'viridis',
                     'viridis',
                     'viridis',
                     'viridis',
                     'viridis']
        plot_list = [plot.detach().cpu() for plot in plot_list]
        pmaps_subplot = get_subplot(3,
                                    plot_list,
                                    title_list,
                                    range_list,
                                    cmap_list,
                                    figsize=figsize,
                                    suptitle='Perfusion Maps')
        subplot_name = str(self.mno % 5) + '_pmaps.png'
        pmaps_subplot.savefig(
            Path.joinpath(Path(self.save_path), subplot_name),
            dpi=500
            )

        # Error visualization
        # LBFGS eta solution from svd approximated ctc
        eta_gen_error = torch.sqrt((eta_gen-eta_pretrain)**2)
        figsize = (8, 6)
        plot_list = [60 * eta_gen_error[0, 0],
                     eta_gen_error[0, 1],
                     eta_gen_error[0, 2]]
        title_list = ["ΔF EST w.r.t SVD",
                      "ΔTau EST w.r.t SVD",
                      "Δk EST w.r.t SVD"]
        rscaleF = 0.1
        rscaleTau = 0.4
        rscalek = 0.1
        range_list = [(0, rscaleF*1.5),
                      (0, rscaleTau*1.5),
                      (0, rscalek*0.04)]
        cmap_list = ['inferno', 'inferno', 'inferno']
        plot_list = [plot.detach().cpu() for plot in plot_list]
        emaps_subplot = get_subplot(3,
                                    plot_list,
                                    title_list,
                                    range_list,
                                    cmap_list,
                                    figsize=figsize,
                                    suptitle='Error Maps')
        subplot_name = str(self.mno % 5) + '_emaps.png'
        emaps_subplot.savefig(
            Path.joinpath(Path(self.save_path), subplot_name),
            dpi=500
            )
        self.mno += 1

    @staticmethod        
    def get_subplot(ncol,
                    plot_list,
                    title_list,
                    range_list,
                    cmap_list,
                    figsize=None,
                    suptitle='Sub-Plot'):

        subplot = plt.figure(figsize=figsize)
        nplots = plot_list.__len__()
        nrows = np.ceil(nplots / ncol).astype(int)
        subplot.suptitle(suptitle)
        gs = subplot.add_gridspec(nrows, ncol)
        for plt_count in range(nplots):
            plot_img = plot_list[plt_count]
            i = plt_count % ncol
            j = np.floor(plt_count / ncol).astype(int)
            axs = subplot.add_subplot(gs[j, i])
            im = axs.imshow(plot_img, cmap=cmap_list[plt_count])
            axs.set_title(title_list[plt_count])
            axs.axis('off')
            l_lim = range_list[plt_count][0]
            u_lim = range_list[plt_count][1]    
            im.set_clim(l_lim, u_lim)
            divider = make_axes_locatable(axs)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        plt.close()

        return subplot
