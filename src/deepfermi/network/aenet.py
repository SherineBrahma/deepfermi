import torch
from network.layers import *

class AENet(nn.Module):

    def __init__(self, enc=None, dec=None, mode='pre_training'):
        super(AENet, self).__init__()

        # General Initializations
        self.enc_net = enc
        self.dec_net = dec
        self.mode = mode

    def samp(self, xin, zin, out_mask, aif=None, ctc=None):
        
        with torch.no_grad():
            
            if self.mode=='pre_training':
                xsamp = self.dec_net(xin, zin, out_mask)
                
            elif self.mode in['fine_tuning', 'testing']:
                xsamp = self.dec_net(xin, zin, out_mask, aif, ctc)

        return xsamp
    
    def enc_latent(self, xenc):
        
        zenc = self.enc_net(xenc)

        return zenc

    def __call__(self, xenc, xin, seg, aif=None, ctc=None, indx_dc=None):
        
        zenc = self.enc_net(xenc)
        
        if self.mode=='pre_training':
            xout = self.dec_net(xin, zenc, seg)
                
        elif self.mode in['fine_tuning', 'testing']:
            xout = self.dec_net(xin, zenc, seg, aif, ctc, indx_dc)        

        return xout, zenc