import random
import shutil
from pathlib import Path
from typing import Tuple

import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gamma
from tqdm import tqdm


def assign_2D_qpar_val(seg,
                       seg_indx,
                       map,
                       inter_sb_var,
                       intra_sb_var) -> torch.Tensor:
    """
    Assigns values to a 2D map based on segment index with added variations.

    Args:
        seg (Tensor): 2D segmentation map indicating regions.
        seg_indx (int): Segment index for the specific region to update.
        map (Tensor): The parameter map to which the values are assigned.
        inter_sb_var (list): [mean, std, min_clip, max_clip] for inter-slice
                             variations.
        intra_sb_var (list): [mean, std, min_clip, max_clip] for intra-slice
                             variations.

    Returns:
        Tensor: Updated parameter map with values assigned to the specified
                segment.
    """

    qpar_val = torch.clip(
        torch.normal(inter_sb_var[0],
                     inter_sb_var[1]),
        inter_sb_var[2],
        inter_sb_var[3]
        )
    seg_xdim = seg.shape[0]
    seg_ydim = seg.shape[1]
    for y in range(seg_ydim):
        for x in range(seg_xdim):
            if seg[y][x] == seg_indx:
                map[y][x] = qpar_val + torch.clip(
                    torch.normal(intra_sb_var[0],
                                 intra_sb_var[1]),
                    intra_sb_var[2],
                    intra_sb_var[3]
                    )
    return map


def fermi_ir_func(t, eta) -> torch.Tensor:
    """
    Calculates the Fermi impulse response function for given time and
    perfusion parameters.

    Args:
        t (Tensor): Time points.
        eta (Tensor): Perfusion parameters including flow rate, delay, and
                      decay rate.

    Returns:
        Tensor: The computed Fermi impulse response.
    """

    one = torch.ones(eta[0].shape, device=eta.device)
    t_len = t.shape[0]    
    t = t * one.unsqueeze(-1).repeat(1, 1, t_len)
    flow_rate = eta[0].unsqueeze(-1)
    delay = eta[1].unsqueeze(-1)
    decay_rate = eta[2].unsqueeze(-1)
    with torch.no_grad():
        unit_step = torch.heaviside(t-delay,
                                    torch.tensor(0.5,
                                                 dtype=delay.dtype,
                                                 device=delay.device)
                                    )
    output = flow_rate*(1/(torch.exp((t-delay)*decay_rate)+1)) * unit_step
    return output


def aif_func(t, gpar) -> torch.Tensor:
    """
    Computes the Arterial Input Function (AIF) based on gamma variate function
    that is widely used to model the AIF in dynamic contrast-enhanced MRI.

    Args:
        t (Tensor): Time points.
        gpar (Tensor): Gamma variate function parameters including alpha, beta,
                       and delay.

    Returns:
        Tensor: The computed arterial input function.
    """

    xdim = gpar[0].shape[0]
    ydim = gpar[0].shape[1]
    one = torch.ones(gpar[0].shape, device=gpar.device)
    t_len = t.shape[0]
    t_one = t * one.unsqueeze(-1).repeat(1, 1, t_len)
    t = t.cpu()
    aplha_gamma = gpar[0].cpu()
    beta_gamma = gpar[1].cpu()
    delay_gamma = gpar[2].cpu()
    aif = torch.zeros(t_one.shape)
    for y in range(ydim):
        for x in range(xdim):
            cond = (aplha_gamma[y][x] != 0
                    and delay_gamma[y][x] != 0
                    and beta_gamma[y][x])
            if cond != 0:
                aif[y][x] = 0.06 * torch.tensor(
                    gamma.pdf(t, aplha_gamma[y][x],
                              delay_gamma[y][x],
                              beta_gamma[y][x])
                    )
    aif = aif.cuda()

    return aif


def convolve(input, im_res) -> torch.Tensor:
    """
    Applies 1D convolution between an input signal and an impulse response.

    Args:
        input (Tensor): The input signal to be convolved.
        im_res (Tensor): The impulse response to convolve with the input
                         signal.

    Returns:
        Tensor: The result of the convolution.
    """

    # output =  im_res ⨂ input
    _, _, t_len = im_res.shape
    im_res_flip = torch.flip(im_res, [2])
    output = torch.zeros(input.shape,
                         dtype=input.dtype,
                         device=input.device)
    for t_indx in range(t_len):
        output[..., t_indx] = torch.sum(
            im_res_flip[..., -(t_indx+1):] * input[..., :(t_indx+1)],
            dim=2
            )

    return output


def induce_outliers(ctc,
                    im_sig,
                    seg,
                    bpool,
                    max_outliers,
                    device='cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulates motion artifacts by randomly shifting the blood pool signal at
    random time positions to overlap with the segmented myocardial regions.

    Args:
        ctc (Tensor): The concentration-time curve tensor.
        im_sig (Tensor): The signal intensity image tensor.
        seg (Tensor): The segmentation map indicating myocardial regions.
        bpool (Tensor): The blood pool signal.
        param_range (dict): Dictionary containing parameters like max_outliers
                            for outlier introduction.
        device (str): The device to use for tensor operations, defaults
                      to 'cuda'.

    Returns:
        Tensor, Tensor: Updated concentration-time curve (ctc) and
                        signal intensity image (im_sig) with induced outliers.
    """

    # Inducing motion-outliers
    mask_indx = torch.randint(0,
                              im_sig.shape[-1],
                              (max_outliers,))

    # Bounding box area
    bbox = torch.zeros(seg.shape, device=device)
    bbox[seg == 1] = 1
    bbox[seg == 71] = 1
    bbox_x = bbox.sum(1)
    bbox_y = bbox.sum(0)
    bbox = (bbox_x.unsqueeze(1)) * (bbox_y.unsqueeze(0))
    bbox[bbox != 0] = 1
    bbox_xlen, bbox_ylen = bbox.sum(0).max(), bbox.sum(1).max()

    # Translating affected frames
    # pylint: disable=not-callable
    myo_seg = torch.zeros(seg.shape, device=device)
    myo_seg[seg == 1] = 1
    myo_seg[seg == 71] = 1
    nx, ny, _ = bpool.shape
    for frame in mask_indx:
        xshift = int(0.5 * torch.rand(1, device=device) * bbox_xlen)
        yshift = int(0.5 * torch.rand(1, device=device) * bbox_ylen)
        outlier_frame = F.pad(
            bpool[..., frame],
            (nx // 2, nx // 2, ny // 2, ny // 2)
            )
        outlier_frame = torch.roll(
            outlier_frame,
            shifts=(xshift, yshift),
            dims=(0, 1)
            )
        outlier_frame = outlier_frame[nx//2:nx//2 + nx, 
                                      ny//2:ny//2 + ny]
        outlier_frame = myo_seg * outlier_frame
        ctc[..., frame][outlier_frame != 0] = outlier_frame[
            outlier_frame != 0
            ]
        im_sig[..., frame][outlier_frame != 0] = outlier_frame[
            outlier_frame != 0
            ]

    return ctc, im_sig


def generate_gif(frames, save_path,
                 gif_name='Untitled.gif',
                 cmap='gray',
                 clim=[0, 1]) -> None:
    """
    Generates a GIF from a series of image frames and saves it to the
    specified path.

    Args:
        frames (np.array): The frames to be used to generate the GIF.
        save_path (str): The directory where the GIF will be saved.
        gif_name (str): The name of the GIF file.
        cmap (str): The color map to use for the GIF.
        clim (list): The color limits for the GIF.

    Returns:
        None
    """

    frames = np.asarray(frames)
    gif_dir = Path.joinpath(save_path, 'gif_cache')
    Path(gif_dir).mkdir(parents=True, exist_ok=True)
    print('generating gif...')
    for f_count in tqdm(range(frames.shape[2])):
        f_current = frames[..., f_count]
        fig = plt.figure()
        im = plt.imshow(f_current, cmap=cmap)
        plt.axis('off')
        im.set_clim(clim[0], clim[1])
        plt.colorbar(im)
        frame_name = str(f_count)
        frame_name = str.zfill(
            frame_name, int(np.floor(np.log10(frames.shape[2])+1))
            )
        fig.savefig(Path.joinpath(gif_dir, frame_name), dpi=100)
        plt.close()
    filenames = list(sorted(Path(gif_dir).glob('*.png*')))
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(Path.joinpath(save_path, gif_name), images)
    shutil.rmtree(gif_dir)
    print('generating gif complete...')


def visualize_dataset(data_dic, save_path) -> None:
    """
    Visualizes a random sample from the dataset by plotting parameter maps and
    saving them as images and GIFs.

    Args:
        data_dic (dict): The dataset dictionary containing signal,
                         segmentation, and perfusion parameter data.
        save_path (str): The path where visualized images and GIFs will be
                         saved.

    Returns:
        None
    """

    rand_indx = random.randint(0, (data_dic.keys().__len__() - 1))
    eta = data_dic[list(data_dic.keys())[rand_indx]]['eta']
    im_sig = data_dic[list(data_dic.keys())[rand_indx]]['im_sig']
    ctc = data_dic[list(data_dic.keys())[rand_indx]]['ctc']

    # Perfusion Parameters Maps
    plot_list = [60 * eta[0], eta[1], eta[2]]
    title_list = ["F GND", "Tau GND", "k GND"]
    range_list = [(0, 4), (0, 3), (0, 0.2)]
    cmap_list = ['viridis', 'viridis', 'viridis']
    no_of_plots = plot_list.__len__()
    no_of_col = 3
    no_of_rows = np.ceil(no_of_plots / no_of_col).astype(int)
    sub_plot_fig = plt.figure(figsize=(14, 6))
    sub_plot_fig.suptitle('Posterior Sampling')
    gs = sub_plot_fig.add_gridspec(no_of_rows, no_of_col)
    for plt_count in range(no_of_plots):
        plot_img = plot_list[plt_count]
        i = plt_count % no_of_col
        j = np.floor(plt_count / no_of_col).astype(int)
        axs = sub_plot_fig.add_subplot(gs[j, i])
        im = axs.imshow(plot_img, cmap=cmap_list[plt_count])
        axs.set_title(title_list[plt_count])
        axs.axis('off')
        l_lim = range_list[plt_count][0]
        u_lim = range_list[plt_count][1]    
        im.set_clim(l_lim, u_lim)
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    sub_plot_fig.savefig(Path.joinpath(save_path, 'PParam_Maps.png'), dpi=500)
    plt.close()

    # GIF
    generate_gif(ctc,
                 save_path,
                 gif_name='ctc.gif',
                 cmap='gray',
                 clim=[0, 0.8*ctc.max()])
    generate_gif(im_sig,
                 save_path,
                 gif_name='im_sig.gif',
                 cmap='gray',
                 clim=[0, 0.8*im_sig.max()])


def construct_dictionary(file_path,
                         param_range,
                         dtype=torch.float64,
                         device='cuda') -> dict:
    """
    Constructs a dataset dictionary from an HDF5 file by synthesizing data for
    myocardial perfusion and arterial input.

    Args:
        file_path (str): Path to the HDF5 file containing data.
        param_range (dict): Dictionary of parameter ranges for generating
                            simulated data.
        dtype (torch.dtype): The data type for the tensors, defaults to
                             torch.float64.
        device (str): The device to use for tensor operations, defaults to
                      'cuda'.

    Returns:
        dict: A dictionary containing the synthesized data (e.g., signal,
              segmentation, perfusion parameters).
    """

    # Time points
    time = torch.arange(param_range["time"]["start"],
                        param_range["time"]["end"],
                        param_range["time"]["step_size"],
                        dtype=dtype,
                        device=device)
    # Construc dictionary
    data_dic = {}
    with h5py.File(file_path, "r") as f:

        # Synthesizing data
        for sb in tqdm(range(list(f.keys()).__len__())):

            # # Extracting segmentation
            # # Segmentation Indentifiers
            # 1: Myocardium Left Ventrical
            # 71: Lesion Myocardium Left Ventrical
            # 5: AIF
            pid = list(f.keys())[sb]
            data_dic[pid] = {}
            seg = torch.tensor(f[pid]['segmentation'][()], device=device)

            # Creating place holder for perfusion parameters
            flow_rate_sim = torch.zeros(seg.shape, dtype=dtype, device=device)
            delay_sim = torch.zeros(seg.shape, dtype=dtype, device=device)
            decay_rate_sim = torch.zeros(seg.shape, dtype=dtype, device=device)

            # Myocardium healthy region
            flow_rate_sim = assign_2D_qpar_val(
                seg,
                1,
                flow_rate_sim,
                param_range["myo_healthy"]["flow"][0],
                param_range["myo_healthy"]["flow"][1]
                )
            delay_sim = assign_2D_qpar_val(
                seg,
                1,
                delay_sim,
                param_range["myo_healthy"]["delay"][0],
                param_range["myo_healthy"]["delay"][1]
                )
            decay_rate_sim = assign_2D_qpar_val(
                seg,
                1,
                decay_rate_sim,
                param_range["myo_healthy"]["decay"][0],
                param_range["myo_healthy"]["decay"][1])

            # Myocardium ischemic region
            flow_rate_sim = assign_2D_qpar_val(
                seg,
                71,
                flow_rate_sim,
                param_range["myo_ischemic"]["flow"][0],
                param_range["myo_ischemic"]["flow"][1]
                )
            delay_sim = assign_2D_qpar_val(
                seg,
                71,
                delay_sim,
                param_range["myo_ischemic"]["delay"][0],
                param_range["myo_ischemic"]["delay"][1]
                )
            decay_rate_sim = assign_2D_qpar_val(
                seg,
                71,
                decay_rate_sim,
                param_range["myo_ischemic"]["decay"][0],
                param_range["myo_ischemic"]["decay"][1]
                )

            # Calculating Fermi impulse response
            cat_dim = 0
            eta = torch.cat(
                (flow_rate_sim.unsqueeze(0),
                 delay_sim.unsqueeze(0),
                 decay_rate_sim.unsqueeze(0)),
                cat_dim
                )
            fermi_ir = fermi_ir_func(time, eta)

            # Arterial Input Function
            aif = torch.zeros(seg.shape, dtype=dtype, device=device)
            aplha_gamma = torch.zeros(seg.shape, dtype=dtype, device=device)
            beta_gamma = torch.zeros(seg.shape, dtype=dtype, device=device)
            delay_gamma = torch.zeros(seg.shape, dtype=dtype, device=device)
            # Assigning aif values [mean, variance, min_clip, max_clip]
            # Aplha gamma
            aplha_gamma = assign_2D_qpar_val(
                seg,
                5,
                aplha_gamma,
                param_range["aif"]["alpha"][0],
                param_range["aif"]["alpha"][1]
                )
            beta_gamma = assign_2D_qpar_val(
                seg,
                5,
                beta_gamma,
                param_range["aif"]["beta"][0],
                param_range["aif"]["beta"][1]
                )
            delay_gamma = assign_2D_qpar_val(
                seg,
                5,
                delay_gamma,
                param_range["aif"]["delay"][0],
                param_range["aif"]["delay"][1]
                )
            # AIF in blood pool Left Ventrical
            cat_dim = 0
            gpar = torch.cat(
                (aplha_gamma.unsqueeze(0),
                 beta_gamma.unsqueeze(0),
                 delay_gamma.unsqueeze(0)),
                cat_dim
                )
            aif = aif_func(time, gpar)

            # Average AIF within the blood pool
            bp_seg = seg.clone()
            bp_seg[bp_seg != 5] = 0
            bp_seg[bp_seg == 5] = 1
            aif_avg = torch.mean(aif[bp_seg == 1, ...], 0).clone()
            aif_2D = (aif_avg.unsqueeze(0).unsqueeze(0)
                      * torch.ones(fermi_ir.shape, device=device))

            # Concentration curve
            ctc = convolve(aif_2D, fermi_ir)

            # Contrast enhanced signal
            im_sig = ctc + aif

            # Inducing motion-outliers
            max_outliers = param_range["max_outliers"]
            ctc, im_sig = induce_outliers(ctc,
                                          im_sig,
                                          seg,
                                          aif,
                                          max_outliers,
                                          device)

            # Affected ctc
            myo_seg = torch.zeros(seg.shape, device=device)
            myo_seg[seg == 1] = 1
            myo_seg[seg == 71] = 1

            # Creating list
            data_dic[pid]["im_sig"] = im_sig.cpu().numpy()
            data_dic[pid]["seg"] = myo_seg.cpu().numpy()
            data_dic[pid]["ctc"] = ctc.cpu().numpy()
            data_dic[pid]["aif"] = aif_avg.cpu().numpy()
            data_dic[pid]["time"] = time.cpu().numpy()
            data_dic[pid]["wlen"] = param_range["wlen"]
            data_dic[pid]["eta"] = eta.cpu().numpy()

        return data_dic


def main(param_range,
         load_file_path,
         save_file_dir,
         save_file_name,
         device='cuda',
         dtype=torch.float64) -> None:
    """
    Main function to load a XCAT phantom for synthesizing perfusion dataset
    and saving it. The function also visualizes a random sample from the
    dataset.

    Args:
        param_range (dict): Dictionary of parameter ranges for generating the
                            dataset.
        load_file_path (str): Path to the input XCAT phantom data file.
        save_file_dir (str): Directory to save the generated dataset.
        save_file_name (str): The name of the file to save the dataset.
        device (str): The device for tensor operations, defaults to 'cuda'.
        dtype (torch.dtype): The data type for tensors, defaults to
                             torch.float64.

    Returns:
        None
    """

    # Load and save paths
    deepfermi_dir = Path(__file__).resolve().parent.parent.parent
    load_path = deepfermi_dir / load_file_path
    save_dir = deepfermi_dir / save_file_dir

    # Construction of simulated dataset
    data_dic = construct_dictionary(load_path, param_range, dtype, device)
    np.savez(Path.joinpath(save_dir, save_file_name), **data_dic)

    # Visualize a random sample
    visualize_dataset(data_dic, save_dir)


if __name__ == "__main__":

    # Configuration for loading and saving paths, computation device,
    # and tensor data type.
    load_file_path = 'data/XCAT_phantom/quiero_cardiac_mrf_sim.h5'
    save_file_dir = 'data'
    save_file_name = "dce_perfusion_data"
    device = 'cuda'
    dtype = torch.float64

    # Time settings for myocardial perfusion measurement in seconds.
    # Here, wlen is the window length for the first pass of main bolus.
    time_step = 0.94
    time_start = 0
    time_end = 106
    wlen = 100

    # Motion artifacts are simulated by introducing outliers at randomly
    # selected positions within the segmented myocardium. The max_outliers
    # parameter specifies the maximum number of such outlier positions to
    # be introduced into the dataset.
    max_outliers = 5

    # Perfusion parameters are modeled to account for both inter-slice and
    # intra-patient slice variations. These parameters are defined for both
    # the healthy and ischemic myocardial regions, with flow, delay, and decay,
    # each specified by mean, standard deviation (std), minimum clip, and
    # maximum clip values. The format is as follows:
    # (inter-slice, intra-patient slice) x (mean, std, min_clip, max_clip)

    # Perfusion Parameter Simulation
    # Myocardium healthy region
    myo_healthy_flow = np.array([[3/60, 0.005, 0, 5/60],
                                 [0, 0.005, 0, 5/60]])
    myo_healthy_delay = np.array([[2, 0.1, 0, 3],
                                  [0, 0.1, 0, 3]])
    myo_healthy_decay = np.array([[0.1, 0.01, 0, 0.5],
                                  [0, 0.01, 0, 0.5]])
    # Myocardium ischemic region
    myo_ischemic_flow = np.array([[1/60, 0.005, 0, 4/60],
                                  [0, 0.005, 0, 4/60]])
    myo_ischemic_delay = np.array([[3, 0.1, 0, 5],
                                   [0, 0.1, 0, 5]])
    myo_ischemic_decay = np.array([[0.05, 0.01, 0, 0.1],
                                   [0, 0.01, 0, 0.1]])

    # The arterial input function (AIF) is also modeled
    # with alpha, beta, and delay parameters, each specified by the same range
    # parameters. The format is the same as for the perfusion parameters.
    aif_alpha = np.array([[3, 0.1, 2, 5],
                          [0, 0.1, 2, 5]])
    aif_beta = np.array([[1.1, 0.1, 1, 2],
                         [0, 0.05, 1, 2]])
    aif_delay = 10 * time_step * np.array([[1, 0.5, 0.8, 2],
                                           [0, 0.1, 0.8, 2]])

    # Constructing the parameter range dictionary
    # Parameter range settings
    param_range = {}
    # Time (in seconds)
    param_range["time"] = {}
    param_range["time"]["step_size"] = time_step
    param_range["time"]["start"] = time_start
    param_range["time"]["end"] = time_end
    # Main-bolus window length
    param_range["wlen"] = wlen
    # Main-bolus window length
    param_range["max_outliers"] = max_outliers
    # Myocardium healthy region
    param_range["myo_healthy"] = {}
    param_range["myo_healthy"]["flow"] = torch.tensor(myo_healthy_flow,
                                                      dtype=dtype,
                                                      device=device)
    param_range["myo_healthy"]["delay"] = torch.tensor(myo_healthy_delay,
                                                       dtype=dtype,
                                                       device=device)
    param_range["myo_healthy"]["decay"] = torch.tensor(myo_healthy_decay,
                                                       dtype=dtype,
                                                       device=device)
    # Myocardium ischemic region
    param_range["myo_ischemic"] = {}
    param_range["myo_ischemic"]["flow"] = torch.tensor(myo_ischemic_flow,
                                                       dtype=dtype,
                                                       device=device)
    param_range["myo_ischemic"]["delay"] = torch.tensor(myo_ischemic_delay,
                                                        dtype=dtype,
                                                        device=device)
    param_range["myo_ischemic"]["decay"] = torch.tensor(myo_ischemic_decay,
                                                        dtype=dtype,
                                                        device=device)
    # Arterial input function
    param_range["aif"] = {}
    param_range["aif"]["alpha"] = torch.tensor(aif_alpha,
                                               dtype=dtype,
                                               device=device)
    param_range["aif"]["beta"] = torch.tensor(aif_beta,
                                              dtype=dtype,
                                              device=device)
    param_range["aif"]["delay"] = torch.tensor(aif_delay,
                                               dtype=dtype,
                                               device=device)

    # Generate the dataset
    main(param_range,
         load_file_path,
         save_file_dir,
         save_file_name,
         device,
         dtype)
