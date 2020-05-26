import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import torch
import scipy.interpolate as spi
from scipy.io import loadmat
tensor_type = torch.DoubleTensor

from helper_plot import *
from elementary import *
from registration_helper import *
def LDDMM(Ireca,Iref,landmarks_reca,landmarks_ref,niter,kernel_width,gamma,eps):
    
    '''
    This is the principal function, which computes gradient descent to minimize error and find optimal trajectories for control points, alpha.
    ------
    Ireca : image to register, in 3 dimensions, of size (nr,nc,k)
    Iref : image of reference (red anchor), that you want to reach, of size (nr,nc,k)
    landmarks_reca : array of size (n_landmarks,2)
    landmarks_ref : array of size (n_landmarks,2)
    niter: number of iterations of the algorithm to optimize trajectories
    kernel_width : value
    gamma : value
    eps : coefficient in step of gradient descent
    ------
    returns the registered image, registered landmarks, and also optimized control points and alpha trajectories

    '''
    
    cp = torch.from_numpy(landmarks_reca).type(tensor_type)
    cp_ref = torch.from_numpy(landmarks_ref).type(tensor_type)
    Im_reca = torch.from_numpy(Ireca.copy()).type(tensor_type)
    Im_ref = torch.from_numpy(Iref).type(tensor_type)
    
    alpha = torch.zeros(cp.size()).type(tensor_type)
    alpha.requires_grad_(True)

    for it in range(niter):
        
        #### Compute an estimation of control points and alpha trajectories
        traj_cp, traj_alpha = shoot(cp, alpha, kernel_width, n_steps=10)
        
        ##### Registration of the landmarks
        deformed_points = register_points(traj_cp, traj_alpha, cp, kernel_width)[-1]
        
        ##### Computation of the error, function to minimize
        error = torch.sum((deformed_points.contiguous().view(-1) - cp_ref.contiguous().view(-1)) ** 2) + gamma * torch.sum(torch.mm(alpha.T,torch.mm(gaussian_kernel(cp,cp,kernel_width), alpha)))
        error.backward()
        
        ### Gradient descent
        eps_mom = eps/np.sqrt(np.sum(alpha.grad.numpy() ** 2))
        with torch.no_grad():
            alpha -=  eps_mom * alpha.grad  
        alpha.grad.zero_()

    #### Inversed of phi to register Im_reca
    registered_image = register_image(traj_cp, traj_alpha, Im_reca, kernel_width) 
    registered_cp = deformed_points.detach().numpy()
    
    

    return registered_image,registered_cp,traj_cp,traj_alpha