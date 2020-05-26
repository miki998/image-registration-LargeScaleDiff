import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import torch
import scipy.interpolate as spi
from scipy.io import loadmat
tensor_type = torch.DoubleTensor

from elementary import *
from helper_plot import *
def _differences(x, y):
    """ 
    x is of shape (n, 2)
    y is of shape (m, 2)
    --------
    returns the difference between each element of x and y in a (2,n,m) tensor
    
    """
    x_col = x.t().unsqueeze(2)  # (n,2) -> (2,n,1)
    y_lin = y.t().unsqueeze(1)  # (m,2) -> (2,1,m)
    return x_col - y_lin

def _squared_distances(x, y):
    """ 
    x is of shape (n, 2)
    y is of shape (m, 2)
    
    --------
    returns the squared euclidean distance between each element of x and y in a (n,m) tensor
    
    """
    
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    
    return dist

def gaussian_kernel(x, y, kernel_width):
    """ 
    x is of shape (n, 2)
    y is of shape (m, 2)
    kernel_width is a value
    
    --------
    returns the gaussian kernel value between each element of x and y in a (n,m) tensor
    
    """
    squared_dist = _squared_distances(x, y)
    return torch.exp(- squared_dist / kernel_width **2 )


def h_gradx(cp, alpha, kernel_width):
    '''
    This function computes derivative of the kernel for each couple (cp_i,alpha_i), with cp_i a control point(landmark).
    ---------
    
    cp is of shape (n_landmarks, 2)
    alpha is of shape (n_landmarks, 2)
    kernel_width is a value
    
    --------
    returns a tensor of shape (n_landmarks, 2)
    '''
    sq = _squared_distances(cp, cp)
    A = torch.exp(-sq / kernel_width **2)
    B = _differences(cp, cp) * A

    return (- 2 * torch.sum(alpha * (torch.matmul(B, alpha)), 2) / (kernel_width ** 2)).t()
    
def discretisation_step(cp, alpha, dt, kernel_width):
    
    '''
    TO DO
    ---------
    This function computes a step of discretized equations for both alpha and control points on one step. 
    Compute here a displacement step  of control points an alpha, from discretized system seen in class.
    ---------
    
    cp is of shape (n_landmarks, 2)
    alpha is of shape (n_landmarks, 2)
    dt is your time step 
    kernel_width is a value
    
    --------
    
    returns resulting control point and alpha displacements in tensors of size (n_landmarks,2)
    
    '''
    
    #We compute the kernel distances stored in A
    sq = _squared_distances(cp, cp)
    A = torch.exp(-sq / kernel_width **2)
    #Update landpoints
    result_cp = cp + dt*torch.matmul(A,alpha)
    
    #we add the grad of hamiltonian therefore it is coherent with lecture note and not just kernel
    result_alpha = alpha + dt*h_gradx(cp,alpha,kernel_width)
        
    return result_cp, result_alpha



def shoot(cp, alpha, kernel_width, n_steps=10):
    
    """
    TO DO
    ------------
    This is the trajectory of a Hamiltonian dynamic, with system seen in lecture notes. 
    Compute here trajectories of control points and alpha from t=0 to t=1.
    ------------
    cp is of shape (n_landmarks, 2)
    alpha is of shape (n_landmarks, 2)
    n_step : number of steps in your hamiltonian trajectory, use to define your time step
    kernel_width is a value
    --------
    returns traj_cp and traj_alpha trajectories of control points and alpha in lists. 
    The length of a list is equal to n_step. 
    In each element of the list, you have a tensor of size (n_landmarks,2) returned by rk2_step_with_dp() function.
    
    
    """
    dt = 1/n_steps
    traj_cp = [cp]
    traj_alpha = [alpha]

    #We pay attention to traj_cp and traj_alpha to have the same length n_steps
    for i in range(1,n_steps):
        cp, alpha = traj_cp[-1], traj_alpha[-1]
        result_cp,result_alpha = discretisation_step(cp,alpha,dt,kernel_width)
        traj_cp.append(result_cp); traj_alpha.append(result_alpha)

    return traj_cp, traj_alpha

def register_points(traj_cp, traj_alpha, y, kernel_width):
    """
    TO DO
    ------------
    This is the application of the computed trajectories on a set of points (landmarks or new points).
    ------------
    
    traj_cp is the list containing the trajectory of your landmarks 
    traj_alpha is is the list containing the trajectory of your alpha 
    y : points you want to register (landmarks or other points), size (n,2)
    kernel_width is a value
    
    --------
    
    returns traj_y,  the trajectory of points y, in a list of lenght n_step. 
    In each element of the list, you should have an array of dimension (n,2) (same dimensions as y)
    
    
    """
    traj_y = [y]
    
    for i in range(0,len(traj_cp)):

        sq = _squared_distances(traj_cp[i], traj_y[-1])
        A = torch.exp(-sq / kernel_width **2)
        y_new = traj_y[-1] + torch.matmul(A.T,traj_alpha[i])
        traj_y.append(y_new)

    #we keep the length of traj_y and traj_cp to be the same
    return traj_y[:-1]

def register_image(traj_cp, traj_alpha, image, kernel_width):
    """
    TO DO
    ------------
    This is the application of the computed trajectories on an image, by computation of inversed phi_1.
    ATTENTION : Compute inverse deformation of image points in a tensor named deformed_points, by using registered_points() function

    ------------
    
    traj_cp is the list containing the trajectory of your landmarks 
    traj_alpha is the list containing the trajectory of your alpha 
    image : image to register, of size (nr,nc,k), k is number of channels/ pixels' values, 3 for colored image, one for grey image
    kernel_width is a value
    
    --------
    
    returns the registered image, of same dimensions as image, (nr,nc,k)
    
    """
    
    #Meshgrid to apply inverse map onto
    xim,yim = np.meshgrid(range(image.shape[1]),range(image.shape[0]))
    xim = xim.reshape(1,-1)
    yim = yim.reshape(1,-1)
    pts = np.stack((yim.reshape(yim.shape[1]),xim.reshape(xim.shape[1]))).T
    pts = torch.from_numpy(pts).type(tensor_type)
    
    #compute inverse map on all the pixels so we can then use interpolate_image to compute registered images
    deformed_points = register_points(traj_cp[::-1], [-alpha for alpha in traj_alpha[::-1]], pts, kernel_width)[-1]
    
    #deformed_points should be of size (nr*nc,d), the image is of size (nr,nc,k)
    return interpolate_image(image, deformed_points)
    