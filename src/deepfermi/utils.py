import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time as exe_time
from datetime import date
from itertools import combinations
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

def secs2time(seconds):
    """
	function for printing seconds in the format weeks, days, hours, minutes
	"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    w, d = divmod(d, 7)

    time_string = '{:d} w : {:02d} d : {:02d} h : {:02d} m'.format(np.int(w), np.int(d), np.int(h), np.int(m))

    return time_string

def svd_approx(xin, k=4):

    # x has shape (mb,ch,nx,ny,nt)
    mb,nx,ny,nt = xin.shape
    # Convert tensors to matrices
    x = xin.reshape(mb,1,nx*ny,nt)
    u, sdiag, v_H = torch.linalg.svd(x)
    sdiag[...,k:] = 0
    offset=((nx*ny)-nt)
    sbar = torch.diag_embed(sdiag, offset=offset)[...,offset:]
    xbar = u @ sbar @ v_H
    xout = xbar.reshape(mb,nx,ny,nt)
    return xout

def expand_dim(xin, f_dim_pad=0, b_dim_pad=0):
    
    f_dim = (None,) * f_dim_pad
    b_dim = (..., ) + (None, ) * b_dim_pad
    
    return xin[f_dim][b_dim]

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
        u_lim = range_list[plt_count][1]    
        im.set_clim(l_lim, u_lim)
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    plt.close()
    
    return subplot


def basic_imshow(xin, path, fig_name='figure', figsize=None, range=(None,None), dpi=500):
    
    '''
    Example: basic_imshow(xenc[0,0,:].cpu(), '/data/brahma01/DCEPerfusion/Experiments/Debug/')
    Use this if convinient: matplotlib.use('TkAgg')
    '''
    # Save a basic figure
    fig = plt.figure(figsize=figsize)
    plt.imshow(xin, vmin=range[0], vmax=range[1])
    plt.colorbar()
    fig.savefig(Path.joinpath(Path(path), fig_name), dpi=dpi)    
    plt.close()
    
def basic_plot(path, x, y=None, fig_name='figure'):
    
    '''
    Example 1: basic_plot('/data/brahma01/DCEPerfusion/Experiments/Debug/', x='it_vect_debug.npy', y='recon_loss_train_debug.npy')
    Example 2: basic_plot(save_path, x='it_vect_debug.npy', y='recon_loss_train_debug.npy', fig_name='Training Loss')
    Example 3: basic_plot('/data/brahma01/DCEPerfusion/Experiments/Debug/', x=it_vect_debug, fig_name='Training Loss')
    '''
    
    # Save a basic plot
    if isinstance(x, str):
        x = np.load(Path.joinpath(Path(path), x))  
    if isinstance(y, str):
        y = np.load(Path.joinpath(Path(path), y))        
    fig = plt.figure()
    if y==None:
        plt.plot(x, linewidth=1)
    else:
        plt.plot(x, y, linewidth=1)    
    plt.xlabel('x')
    plt.ylabel('y')
    fig.savefig(Path.joinpath(Path(path), fig_name), dpi=500)    
    plt.close()

# Data processing

def xyt2xt_yt(xin,reshape_type):

	#x has shape (mb,ch,nx,ny,nt)		
	mb,nch,nx,ny,nt = xin.shape

	if reshape_type=='xt':
		xout = xin.permute(0,2,1,3,4).reshape(mb*nx, nch, ny, nt)

	elif reshape_type =='yt':
		xout = xin.permute(0,3,1,2,4).reshape(mb*ny, nch, nx, nt)
	
	return xout 

def xt_yt2xyt(xin,reshape_type,mb):

	if reshape_type =='xt':

		_,nch,ny,nt=xin.shape
		nx = np.int(xin.shape[0]/mb)

		xout = xin.reshape(mb,nx,nch,ny,nt).permute(0,2,1,3,4)
	
	elif reshape_type=='yt':

		_,nch,nx,nt=xin.shape
		ny = np.int(xin.shape[0]/mb)

		xout = xin.reshape(mb,ny,nch,nx,nt).permute(0,2,3,1,4)
	
	return xout

def xyt2xy(xin):

	#x has shape (mb,ch,nx,ny,nt)		
	mb,nch,nx,ny,nt = xin.shape

	xout = xin.permute(0,4,1,2,3).reshape(mb*nt, nch, nx, ny)
	
	return xout 

def xy2xyt(xin, mb):
    
    #x has shape (mb,ch,nx,ny,nt)     
    _,nch,nx,ny=xin.shape
    nt = np.int(xin.shape[0]/mb)

    xout = xin.reshape(mb, nt, nch, nx, ny).permute(0,2,3,4,1)
	
    return xout   

# Cubic Hermite Splines interpolate
def h_poly(t):
    tt = t[None, :]**torch.arange(4, device=t.device)[:, None]
    A = torch.tensor([
        [1, 0, -3, 2],
        [0, 1, -2, 1],
        [0, 0, 3, -2],
        [0, 0, -1, 1]
    ], dtype=t.dtype, device=t.device)
    return A @ tt

def interp_cubic_1D(y, size=None):
    assert size!=None, "Enter output size"
    # Calculating slope and intercept 
    _, nt = y.shape
    o = size/nt
    x = torch.linspace(0, 1, nt, device=y.device)
    m = (y[:, 1:] - y[:, :-1]) / (x[1:][None,:] - x[:-1][None,:])
    m = torch.cat([m[:,[0]], (m[:, 1:] + m[:, :-1]) / 2, m[:,[-1]]], 1)
    i_dx = torch.arange(0,nt-1)
    dx = (x[i_dx + 1] - x[i_dx])/o
    # Cubic interpolation
    dx = torch.kron(dx.unsqueeze(-1), torch.diag(torch.arange(0,o, device=y.device))).sum(-1)
    x_nearest = torch.kron(x[i_dx].unsqueeze(-1), torch.eye(int(o), device=y.device)).sum(-1)    
    xs = torch.cat((x_nearest + dx,torch.tensor([1], device=y.device)), dim=0)
    idxs = torch.searchsorted(x[1:], xs[:])
    dx = (x[idxs + 1] - x[idxs])
    hh = h_poly((xs - x[idxs]) / dx)
    return hh[0][None,:] * y[:, idxs] + hh[1][None,:] * m[:, idxs] * dx[None,:] + hh[2][None,:] * y[:, idxs + 1] + hh[3][None,:] * m[:, idxs + 1] * dx[None,:]


def Interp_Linear_1D(y, size=None):
    assert size!=None, "Enter output size"
    # Calculating slope and intercept 
    nt = y.shape[-1]
    o = size/nt
    x = expand_dim(torch.linspace(0, 1, nt, device=y.device), f_dim_pad=y.dim()-1)
    m = (y[..., 1:] - y[..., :-1]) / (x[...,1:] - x[...,:-1])
    c =  (-m*x[...,:-1]) + y[..., :-1]
    i_dx = torch.arange(0,nt-1)
    dx = (x[...,i_dx + 1] - x[...,i_dx])/o
    # Linear interpolation
    DX = torch.kron(dx.unsqueeze(-1), torch.diag(torch.arange(0,o, device=y.device)))    
    X = (torch.kron(x[...,i_dx].unsqueeze(-1), torch.eye(int(o), device=y.device)) + DX).sum(-1)
    C = c.repeat_interleave(int(o), dim=-1)
    M = m.repeat_interleave(int(o), dim=-1)
    Y = torch.cat((M * X + C,y[...,nt-1:nt]),dim=-1)
    return Y

def interp_linear_1D(y, size=None):
    assert size!=None, "Enter output size"
    # Calculating slope and intercept 
    nt = y.shape[-1]
    o = size/nt
    x = torch.linspace(0, 1, nt, device=y.device)
    m = (y[..., 1:] - y[..., :-1]) / (x[1:][None,:] - x[:-1][None,:])
    c =  (-m*x[:-1][None,:]) + y[..., :-1]
    i_dx = torch.arange(0,nt-1)
    dx = (x[i_dx + 1] - x[i_dx])/o
    # Linear interpolation
    DX = torch.kron(dx.unsqueeze(-1), torch.diag(torch.arange(0,o, device=y.device)))    
    X = (torch.kron(x[i_dx].unsqueeze(-1), torch.eye(int(o), device=y.device)) + DX).sum(-1)
    C = c.repeat_interleave(int(o), dim=-1)
    M = m.repeat_interleave(int(o), dim=-1)
    Y = torch.cat((M * X.unsqueeze(0) + C,y[...,nt-1:nt]),dim=-1)
    return Y

def interp_outlier(y, mask, mode='linear'):
    assert mode=='linear' , "Only linear mode supported"
    mask[0] = mask[-1] = 1 # Ignore special case
    nt = y.shape[-1]
    o = 1
    x = torch.arange(0, nt, device=y.device)
    m = torch.zeros(y[..., 1:].shape, device=y.device)
    c = torch.zeros(y[..., 1:].shape, device=y.device)
    x_mask = x[mask==1]
    y_mask = y[..., mask==1]
    m_mask = (y_mask[..., 1:] - y_mask[..., :-1]) / (x_mask[1:][None,:] - x_mask[:-1][None,:])    
    c_mask =  (-m_mask*x_mask[:-1][None,:]) + y_mask[..., :-1]    
    i_dx = torch.arange(0,nt-1)
    dx = (x[i_dx + 1] - x[i_dx])/o
    # Filling slope and intercept values
    m[...,mask[1:]==1] = m_mask
    c[...,mask[1:]==1] = c_mask
    for i in torch.arange(nt-1,0,-1)-1:
        if mask[1:][i]==0:
            c[...,i] = c[...,i+1]
            m[...,i] = m[...,i+1]    
    # Linear interpolation    
    DX = torch.kron(dx.unsqueeze(-1), torch.diag(torch.arange(0,o, device=y.device)))
    X = (torch.kron(x[i_dx].unsqueeze(-1), torch.eye(int(o), device=y.device)) + DX).sum(-1)
    C = c.repeat_interleave(int(o), dim=-1)
    M = m.repeat_interleave(int(o), dim=-1)
    Y = torch.cat((M * X.unsqueeze(0) + C,y[...,nt-1:nt]),dim=-1)    
    return Y

# def interp_cubic_1D(y, size=None):
#     assert size!=None, "Enter output size"
#     x = torch.linspace(0, 1, y.shape[-1], device=y.device)
#     xs = torch.linspace(0, 1, size, device=y.device)
#     m = (y[:, 1:] - y[:, :-1]) / (x[1:][None,:] - x[:-1][None,:])
#     m = torch.cat([m[:,[0]], (m[:, 1:] + m[:, :-1]) / 2, m[:,[-1]]], 1)
#     idxs = torch.searchsorted(x[1:], xs[:])
#     dx = (x[idxs + 1] - x[idxs])
#     hh = h_poly((xs - x[idxs]) / dx)
#     return hh[0][None,:] * y[:, idxs] + hh[1][None,:] * m[:, idxs] * dx[None,:] + hh[2][None,:] * y[:, idxs + 1] + hh[3][None,:] * m[:, idxs + 1] * dx[None,:]

def total_variation(xin):
    
    a = 1
    
    
    xout = xin
    
    return xout

def min_inter_cluster_distance(data, cluster_assignments):
    distances = []
    for i, j in combinations(range(len(data)), 2):
        if cluster_assignments[i] != cluster_assignments[j]:
            distance = pairwise_distances([data[i]], [data[j]])[0][0]
            distances.append(distance)
    return np.min(distances)

def max_intra_cluster_distance(data, cluster_assignments, cluster_id):
    distances = []
    for i, j in combinations(range(len(data)), 2):
        if cluster_assignments[i] == cluster_id and cluster_assignments[j] == cluster_id:
            distance = pairwise_distances([data[i]], [data[j]])[0][0]
            distances.append(distance)
    return max(distances)

def mean_intra_cluster_distance(data, cluster_assignments, cluster_id):
    distances = []
    for i, j in combinations(range(len(data)), 2):
        if cluster_assignments[i] == cluster_id and cluster_assignments[j] == cluster_id:
            distance = pairwise_distances([data[i]], [data[j]])[0][0]
            distances.append(distance)
    return np.mean(distances)

def centroid_intra_cluster_distance(data, cluster_assignments, cluster_id):    
    data = data[cluster_assignments==cluster_id]
    centroid = data.mean(0, keepdims=True)
    distances = np.linalg.norm(data - centroid, axis=1)    
    return np.mean(distances)

def dunn_index(data, cluster_assignments, diameter_def='max'):
    min_inter_distance = min_inter_cluster_distance(data, cluster_assignments)
    intra_distances = []
    
    if diameter_def == 'max':
        for cluster_id in set(cluster_assignments):
            intra_distances.append(max_intra_cluster_distance(data, cluster_assignments, cluster_id))
    elif diameter_def == 'mean':
        for cluster_id in set(cluster_assignments):
            intra_distances.append(mean_intra_cluster_distance(data, cluster_assignments, cluster_id))
    elif diameter_def == 'centroid':
        for cluster_id in set(cluster_assignments):
            intra_distances.append(centroid_intra_cluster_distance(data, cluster_assignments, cluster_id))
    intra_distances = max(intra_distances)    
    dunn_index = min_inter_distance / intra_distances
    return dunn_index