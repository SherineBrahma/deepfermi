import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable


def secs2time(seconds):
    """
    Function for printing seconds in the format weeks, days, hours, 
    minutes.
    """

    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    w, d = divmod(d, 7)

    time_string = '{:d} w : {:02d} d : {:02d} h : {:02d} m'.format(np.int(w),
                                                                   np.int(d),
                                                                   np.int(h),
                                                                   np.int(m))

    return time_string


def get_subplot(ncol,
                plot_list,
                title_list,
                range_list,
                cmap_list,
                figsize=None,
                suptitle='Sub-Plot'):
    """
    Creates a multi-panel subplot figure with a specified number of columns
    and custom titles, color maps, and value ranges.

    Parameters:
        ncol (int): The number of columns in the subplot grid.
        plot_list (list of arrays): A list of images or data arrays to be
                                    displayed in the subplots.
        title_list (list of str): A list of titles for each subplot.
        range_list (list of tuples): A list of tuples specifying the lower and
                                     upper color limits for each subplot.
        cmap_list (list of str): A list of color maps to be used for each
                                 subplot.
        figsize (tuple, optional): The size of the figure. If not provided,
                                   the default size will be used.
        suptitle (str, optional): The title for the entire figure. Default is
                                  'Sub-Plot'.

    Returns:
        matplotlib.figure.Figure: The generated subplot figure.

    """

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


def interp_linear_1D(y, size=None):
    """
    Performs 1D linear interpolation on an input tensor to a specified size.

    Parameters:
        y (torch.Tensor): The input tensor to be interpolated. The
                          interpolation occurs along the last dimension.
        size (int): The desired output size for the last dimension of the
                    tensor.

    Returns:
        torch.Tensor: The interpolated tensor with the specified size in the
                      last dimension.

    Raises:
        AssertionError: If `size` is None.

    """

    assert size is not None, "Enter output size"
    # Calculating slope and intercept
    nt = y.shape[-1]
    o = size/nt
    x = torch.linspace(0, 1, nt, device=y.device)
    m = (y[..., 1:] - y[..., :-1]) / (x[1:][None, :] - x[:-1][None, :])
    c = (-m*x[:-1][None, :]) + y[..., :-1]
    i_dx = torch.arange(0, nt-1)
    dx = (x[i_dx + 1] - x[i_dx])/o
    # Linear interpolation
    DX = torch.kron(
        dx.unsqueeze(-1),
        torch.diag(torch.arange(0, o, device=y.device))
        )
    X = (
        torch.kron(x[i_dx].unsqueeze(-1),
                   torch.eye(int(o), device=y.device)) + DX
        ).sum(-1)
    C = c.repeat_interleave(int(o), dim=-1)
    M = m.repeat_interleave(int(o), dim=-1)
    Y = torch.cat((M * X.unsqueeze(0) + C, y[..., nt-1:nt]), dim=-1)
    return Y


def expand_dim(xin, f_dim_pad=0, b_dim_pad=0):
    
    f_dim = (None,) * f_dim_pad
    b_dim = (..., ) + (None, ) * b_dim_pad
    
    return xin[f_dim][b_dim]


def Interp_Linear_1D(y, size=None):

    assert size is not None, "Enter output size"
    # Calculating slope and intercept
    nt = y.shape[-1]
    o = size/nt
    x = expand_dim(
        torch.linspace(0, 1, nt, device=y.device),
        f_dim_pad=y.dim()-1
        )
    m = (y[..., 1:] - y[..., :-1]) / (x[..., 1:] - x[..., :-1])
    c = (-m*x[..., :-1]) + y[..., :-1]
    i_dx = torch.arange(0, nt-1)
    dx = (x[..., i_dx + 1] - x[..., i_dx])/o
    # Linear interpolation
    DX = torch.kron(
        dx.unsqueeze(-1), torch.diag(torch.arange(0, o, device=y.device))
        )    
    X = (torch.kron(
        x[..., i_dx].unsqueeze(-1), torch.eye(int(o), device=y.device)
        ) + DX).sum(-1)
    C = c.repeat_interleave(int(o), dim=-1)
    M = m.repeat_interleave(int(o), dim=-1)
    Y = torch.cat((M * X + C, y[..., nt-1:nt]), dim=-1)
    return Y