#Imported libs
import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import torch
import scipy.interpolate as spi
from scipy.io import loadmat
tensor_type = torch.DoubleTensor

#helper libs
from helper_plot import *
from lddmm import *
from elementary import *
from registration_helper import *



def demo():
	Iref = np.array(mpimg.imread("./Images/image_redanch.jpg"))  / 255.
	Ireca =  np.array(mpimg.imread("./Images/image_greyanch.jpg")) / 255.

	# Output Image - padding
	Iref = np.concatenate((Iref,np.ones((45,300,3),dtype=np.int8)),axis=0)

	# one-padding to avoid out-of-bounds problems when one do interpolation
	nr,nc,_ = Ireca.shape
	pad = np.ones((nr,nc,3),dtype=np.int8);
	Iref = np.concatenate((np.concatenate([pad,pad,pad],axis=1),np.concatenate([pad,Iref,pad],axis=1),np.concatenate([pad,pad,pad],axis=1)),axis=0)
	Ireca = np.concatenate((np.concatenate([pad,pad,pad],axis=1),np.concatenate([pad,Ireca,pad],axis=1),np.concatenate([pad,pad,pad],axis=1)),axis=0)

	# positions of "utils pixels"
	rutil = np.arange(nr,(2*nr))
	cutil = np.arange(nc,(2*nc))

	# Preca2 and Pref2 are reference points in each picture : Preca2 for image to register (grey anchor) and Pref2 for target image (red anchor).

	Preca2 = np.load('./data/PsrcAnchors2.npy') 
	Pref2 = np.load('./data/PtarAnchors2.npy')
	n_landmarks = Preca2.shape[1]

	Preca = np.load('./data/PsrcAnchors.npy') 
	Pref = np.load('./data/PtarAnchors.npy') 
	PtsInd = np.arange(0,n_landmarks)
	Npts = np.size(Preca,1)


	mPreca = np.mean(Preca,1)
	mPref = np.mean(Pref,1)
	Preca_copy = Preca - np.repeat(mPreca.reshape(2,-1),Npts,axis=1)
	Pref_copy = Pref - np.repeat(mPref.reshape(2,-1),Npts,axis=1)

	A = Pref_copy.dot(Preca_copy.T).dot(np.linalg.inv(Preca_copy.dot(Preca_copy.T)))
	b = mPref - A.dot(mPreca)


	############################
	#####  We build a grid with coordinates of original pixels in registered image image
	xim,yim = np.meshgrid(np.arange(0,nc),np.arange(0,nr))
	xim = xim.reshape(1,-1)
	yim = yim.reshape(1,-1)

	gridRegInv = np.linalg.inv(A).dot(np.concatenate([xim,yim],axis=0) - np.repeat(b.reshape(2,1),[xim.shape[1]],axis=1))

	##################
	# Registered image obtained by inversed transformation

	Jreg = np.zeros((nr,nc,3))
	for i in range(xim.shape[1]):
	    value = Ireca[int(round(gridRegInv[1,i]) + nr), int(round(gridRegInv[0,i]) + nc),:]
	    Jreg[yim[:,i],xim[:,i],:] = value

	Jreg = JregPreca = np.load('./data/PsrcAnchors.npy') 
	Pref = np.load('./data/PtarAnchors.npy') 
	PtsInd = np.arange(0,n_landmarks)
	Npts = np.size(Preca,1)


	mPreca = np.mean(Preca,1)
	mPref = np.mean(Pref,1)
	Preca_copy = Preca - np.repeat(mPreca.reshape(2,-1),Npts,axis=1)
	Pref_copy = Pref - np.repeat(mPref.reshape(2,-1),Npts,axis=1)

	A = Pref_copy.dot(Preca_copy.T).dot(np.linalg.inv(Preca_copy.dot(Preca_copy.T)))
	b = mPref - A.dot(mPreca)


	############################
	#####  We build a grid with coordinates of original pixels in registered image image
	xim,yim = np.meshgrid(np.arange(0,nc),np.arange(0,nr))
	xim = xim.reshape(1,-1)
	yim = yim.reshape(1,-1)

	gridRegInv = np.linalg.inv(A).dot(np.concatenate([xim,yim],axis=0) - np.repeat(b.reshape(2,1),[xim.shape[1]],axis=1))

	##################
	# Registered image obtained by inversed transformation

	Jreg = np.zeros((nr,nc,3))
	for i in range(xim.shape[1]):
	    value = Ireca[int(round(gridRegInv[1,i]) + nr), int(round(gridRegInv[0,i]) + nc),:]
	    Jreg[yim[:,i],xim[:,i],:] = value

	Jreg = Jreg
	
	###################################
	##### Affine transformation of new landmarks with parameters from first set of landmarks
	Preg = A.dot(Preca) + np.repeat(b.reshape(2,1),Npts,axis=1)
	Preg2 = A.dot(Preca2) + np.repeat(b.reshape(2,1),n_landmarks,axis=1)



	## This function is here to transform control points.
	## After application of this function, the shape of control points is (n_landmarks,d).
	## !!!!!! This time, in cp_reca and cp_ref : first column corresponds to y-axis, second column to x-axis, to correspond to images dimensions.
	cp_reca,cp_ref = load_landmarks_torch(m_reca=Preg2,m_ref=Pref2)


	eps = 1 ## Don't play  with this parameter

	#########################
	#### Play with parameters

	kernel_width = 20
	gamma = 0
	niter = 500


	### Jreg is the grey anchor registered with affine registration (as in first practical session). 

	Im_reg,Cp_reg,traj_cp,traj_alpha = LDDMM(Jreg,Iref[:,cutil,:][rutil],
											cp_reca,cp_ref,niter,
											kernel_width,gamma,eps)

	fig = plt.figure(figsize=(20,20))
	axes = fig.add_subplot(2, 1, 1)   
	axes.imshow(Iref[:,cutil,:][rutil],alpha=1)
	axes.imshow(Im_reg,alpha=0.5)
	plt.scatter(Cp_reg[:, 1], Cp_reg[:, 0], marker='o')
	plt.scatter(Pref2[0,:], Pref2[1,:], marker='+')
	plt.title('Registered image and target image superimposed')

	axes = fig.add_subplot(2, 1, 2)
	plot_large_deformation_grid(traj_cp, traj_alpha, Jreg, kernel_width,axes)
	fig.gca().invert_yaxis()
	plt.title('Deformation field')
	plt.show()

if __name__ == '__main__':
	demo()
	#main()