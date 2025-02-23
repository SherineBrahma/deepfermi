from pathlib import Path

import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image


def get_subplot(ncol, plot_list, title_list, range_list, cmap_list, figsize=None, suptitle='Sub-Plot'):
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
        u_lim = round(range_list[plt_count][1], 3)
        im.set_clim(l_lim, u_lim)
        divider = make_axes_locatable(axs)
        cax = divider.append_axes('bottom', size='5%', pad=0.05)
        bin = 4
        tick_array = np.round(np.linspace(l_lim, u_lim, bin), 3)
        cb = plt.colorbar(im, cax=cax, ticks=tick_array, orientation='horizontal')
        cb.ax.set_xticklabels(tick_array)
    plt.close()

    return subplot


def main(pdf_name_list, img_sub_dir_list, read_path, save_path) -> None:
    # General
    experiments_directory = str(Path(__file__).resolve().parent.parent.parent.parent / 'experiments')
    save_path = experiments_directory + '/' + save_path
    for i in range(read_path.__len__()):
        read_path[i] = experiments_directory + '/' + read_path[i]
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Initializing files for recording
    open(Path.joinpath(Path(save_path), 'eval_metrics.txt'), 'w').close()

    # Generating plot
    for i, path in enumerate(read_path):
        # Loading arrays
        pid = np.load(Path.joinpath(Path(path), 'pid.npy'))
        im_sig = np.load(Path.joinpath(Path(path), 'im_sig.npy'))
        seg = np.load(Path.joinpath(Path(path), 'seg.npy'))
        eta_lbfgs = np.load(Path.joinpath(Path(path), 'eta_lbfgs' + '.npy'))
        eta_net = np.load(Path.joinpath(Path(path), 'eta_net' + '.npy'))
        eta_gnd = np.load(Path.joinpath(Path(path), 'eta_gnd' + '.npy'))

        # Remove background
        eta_lbfgs[np.repeat(seg[:, None], 3, axis=1) == 0] = 0
        eta_net[np.repeat(seg[:, None], 3, axis=1) == 0] = 0
        eta_gnd[np.repeat(seg[:, None], 3, axis=1) == 0] = 0

        # Calculate Normalized Absolute Error
        # Absolute Error
        eta_lbfgs_error = eta_lbfgs - eta_gnd
        eta_net_error = eta_net - eta_gnd
        # Normalization
        eta_lbfgs_NAE = np.abs(eta_lbfgs_error).sum((0, 2, 3)) / np.abs(eta_gnd).sum((0, 2, 3))
        eta_net_NAE = np.abs(eta_net_error).sum((0, 2, 3)) / np.abs(eta_gnd).sum((0, 2, 3))

        with open(Path.joinpath(Path(save_path), 'eval_metrics.txt'), 'a') as file:
            file.write(pdf_name_list[i] + '\n')
            file.write('=' * pdf_name_list[i].__len__() + '\n')
            file.write('%s %f\n' % ('Flow NAE LBFGS: ', eta_lbfgs_NAE[0]))
            file.write('%s %f\n' % ('Delay NAE LBFGS: ', eta_lbfgs_NAE[1]))
            file.write('%s %f\n' % ('Decay NAE LBFGS: ', eta_lbfgs_NAE[2]))
            file.write('%s %f\n' % ('Flow NAE DeepFermi: ', eta_net_NAE[0]))
            file.write('%s %f\n' % ('Delay NAE DeepFermi: ', eta_net_NAE[1]))
            file.write('%s %f\n' % ('Decay NAE DeepFermi: ', eta_net_NAE[2]))
            file.write('\n')


if __name__ == '__main__':
    # Folders to be read
    pdf_name_list = ['outlier_retained', 'outlier_removed']
    img_sub_dir_list = ['eta_outlier_removed', 'eta_outlier_retained']
    read_path = ['01_02_test_deepfermi_pretrained_outlier_retained', '01_03_test_deepfermi_pretrained_outlier_removed']
    save_path = '01_04_test_deepfermi_pretrained'
    main(pdf_name_list, img_sub_dir_list, read_path, save_path)
