from network.layers import *
import numpy as np

def xyt2xt_yt(xin,reshape_type):

	#x has shape (mb,2,nx,ny,nt)		
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

class XTYT_EncNet(nn.Module):

    def __init__(self, ncin=2, nstage=3, nconv_stage=2, nfilters=16, ksize=3, res_connect=False, dropout=0, bias=False):
        super(XTYT_EncNet, self).__init__()

        # General Initializations
        dim = 2
        dsamp_fact = 2
        nfilters_start = nfilters
        self.zshape = None
        self.res_connect = res_connect

        self.upsamp_shape = []

        # XTYT functions
        self.reshape_op_xyt2xt_yt = XYT2XT_YT()
        self.reshape_op_xt_yt2xyt = XT_YT2XYT()

        # ENCODER
        # Initialization
        self.enc = nn.ModuleList()
        # Input Layer
        self.enc.append(ConvBlock(dim=dim, shape=1, nch=ncin, nfilters=nfilters, nconvs=1, pad=0))
        # Encoding Blocks
        nch = nfilters_start
        nfilters = nfilters_start
        for ns_count in range(nstage):
            self.enc.append(EncoderBlock(dim=dim, nch=nch, nfilters=nfilters, nconvs=nconv_stage, dropout=dropout, stage_indx=ns_count))
            nch = nfilters
            nfilters = dsamp_fact * nch
        self.enc.append(ConvLayer(dim=dim, shape=3, nch=nch, nfilters=nch, pad=1))

    def __call__(self, xin, type='xt'):

        # get the number of samples used; needed for re-assembling operation
        # x has the shape (mb,2,nx,ny,nt)
        mb = xin.shape[0]

        # Spatio-temporal slices
        xin_xt_yt = self.reshape_op_xyt2xt_yt(xin, type)

        # Encoding
        xenc = xin_xt_yt
        for i in range(len(self.enc)):
            xenc = self.enc[i](xenc)
            
        # Encoded latent space information            
        zout = xenc
            
        return zout

class EncoderBlock(nn.Module):

    def __init__(self, dim=2, nch=3, nfilters=3, nconvs=2, dropout=0, stage_indx=1, bias=True):
        super(EncoderBlock, self).__init__()

        self.enc_blk = nn.ModuleList()
        if stage_indx==0:
            self.enc_blk.append(Identity())
        else:
            self.enc_blk.append(Pooling(dim=dim, pooling_type="Max"))
        self.enc_blk.append(Dropout(dropout=dropout, dim=dim))
        self.enc_blk.append(ConvBlock(dim=dim, shape=3, nch=nch, nfilters=nfilters, nconvs=nconvs, bnorm=False, bias=bias))

    def __call__(self, xin):

        x1 = self.enc_blk[0](xin)
        x2 = self.enc_blk[1](x1)
        x3 = self.enc_blk[2](x2)
        xout = x3

        return xout
    
class XYT2XT_YT(nn.Module):
	""" 
	Class needed for the reshaping operator:
	Given x with shape (mb,2,Nx,Ny,Nt), x is reshped to have
	either shape (mb*Nx,2,Ny,Nt) for the yt-domain or 
	the shape (mb*Ny,2,Nx,Nt) for the xt-domain
	"""
	
	def __init__(self):
		super(XYT2XT_YT, self).__init__()

	def forward(self, x, reshape_type):

		return xyt2xt_yt(x, reshape_type)



class XT_YT2XYT(nn.Module):
	""" 
	Class needed for the reassembling the cine MR image to its original shape:
	reverses the operation XYT2XT_YT,
	note that the mini-batch size is needed
	"""
	
	def __init__(self):
		super(XT_YT2XYT, self).__init__()

	def forward(self, x, reshape_type,mb):
		
		return xt_yt2xyt(x, reshape_type,mb)