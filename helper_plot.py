import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import torch
import scipy.interpolate as spi
from scipy.io import loadmat
tensor_type = torch.DoubleTensor


from registration_helper import *
from elementary import *
def plot_large_deformation_grid(traj_cp, traj_alpha, image, kernel_width,ax):
    '''
    Returns a grid of the deformation field computed with traj_cp and traj_alpha lists.
    -----
    traj_cp : list of length equals to the number of steps of the algorithm (n_step)
    traj_alpha : list of length equals to the number of steps of the algorithm (n_step)
    image : (nr,nc,k)
    '''
    i,j,_ = image.shape
    downsampling_factor = 10
    i_ = i // downsampling_factor
    j_ = j // downsampling_factor
    
    points = np.array(np.meshgrid(range(i_), range(j_))) * downsampling_factor
    points = np.swapaxes(points, 0, 2).reshape(i_ * j_, 2)
    points = torch.from_numpy(points).type(tensor_type)
    deformed_points = register_points(traj_cp, traj_alpha, points, kernel_width)[-1]
    
    g = deformed_points.detach().numpy().reshape(i_, j_, 2)
    ax.plot([g[:, :-1, 1].ravel(), g[:, 1:, 1].ravel()],
            [g[:, :-1, 0].ravel(), g[:, 1:, 0].ravel()], 'k', linewidth=1.2, marker='o', markersize=0.2)
    ax.plot([g[:-1, :, 1].ravel(), g[1:, :, 1].ravel()],
            [g[:-1, :, 0].ravel(), g[1:, :, 0].ravel()], 'k', linewidth=1.2, marker='o', markersize=0.2)

