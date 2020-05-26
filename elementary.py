import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import torch
import scipy.interpolate as spi
from scipy.io import loadmat
tensor_type = torch.DoubleTensor

def _swap_colums(ar, i, j):
    aux = np.copy(ar[:, i])
    ar[:, i] = np.copy(ar[:, j])
    ar[:, j] = np.copy(aux)
    return np.copy(ar)

def load_landmarks_torch(m_reca=None,m_ref=None):
    if m_reca is None :
        m_reca_out = np.load('./data/PsrcAnchors2.npy').T 
    else :
        m_reca_out = m_reca.copy().T     
    if m_ref is None :
        m_ref_out = np.load('./data/PtarAnchors2.npy').T  
    else :
        m_ref_out = m_ref.copy().T
    return _swap_colums(m_reca_out, 0, 1), _swap_colums(m_ref_out, 0, 1)



def interpolate_image(intensities, deformed_pixels, padding_width=1):
    '''
    This function, given original image in intensities, and positions of registeres pixels in original image, return the final registered image
    ------- 
    intensities : (nr,nc,k)
    deformed_pixels : (nr*nc,d)
    -------
    returns registered image in JregLD, of shape (nr,nc,k)
    
    '''
    nr,nc,_ = intensities.shape
    xim,yim = np.meshgrid(range(0,nc),range(0,nr))
    xim = xim.reshape(-1)
    yim = yim.reshape(-1)
    
    deformated_pixels_numpy = deformed_pixels.detach().numpy()

    
    pad = np.ones(intensities.shape)
    padded_image = np.concatenate((np.concatenate([pad,pad,pad],axis=1),np.concatenate([pad,intensities.numpy(),pad],axis=1),np.concatenate([pad,pad,pad],axis=1)),axis=0)

    
    JregLD = np.zeros((nr,nc,3))
    
    for i in range(len(xim)):
        value = padded_image[int(round(deformated_pixels_numpy[i,0]) + nr), int(round(deformated_pixels_numpy[i,1]) + nc),:]
        JregLD[yim[i],xim[i],:] = value

    return JregLD