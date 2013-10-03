import numpy as np
import scipy as sp 
import autodiff 
import pylab as plt 
from sklearn.base import BaseEstimator

import theano.tensor as T 
from theano import config, shared, function

def plot_images(imgs, layout, img_sz = 0.7, suptitle = ''):
	"""
	Plot mulitple images in a compact matrix fomrat 
	imgs : list of img or 3D matrix 
	layout is the nrows x ncols of the plot matrix 
	img_size is the size of individual images in the matrix 
	"""
	nrows, ncols = layout 
	fig, axes = plt.subplots(nrows, ncols, 
		figsize = (img_sz * ncols, img_sz * nrows))
	axes = axes.ravel()
	fig.subplots_adjust(hspace = 0, wspace = 0)
	fig.suptitle(suptitle)
	for i, img in enumerate(imgs):
		axes[i].get_xaxis().set_visible(False)
		axes[i].get_yaxis().set_visible(False)
		axes[i].imshow(img)

def sample_patches(images, npatches, patch_sz):
	"""
	randomly generate n_patches patches from image pool images, 
	each image patch should be of patch_sz x patch_sz
	selected patches may have overlaps with each other
	"""
	nimages, nrows, ncols = images.shape
	img_index = np.random.randint(0, nimages, npatches)
	row_index = np.random.randint(0, nrows-patch_sz, npatches)
	col_index = np.random.randint(0, ncols-patch_sz, npatches)
	patches = np.empty((npatches, patch_sz, patch_sz))
	for i, (img, row, col) in enumerate(zip(img_index, row_index, col_index)):
		patches[i] = images[img, row:row+patch_sz, col:col+patch_sz]
	return patches

def sigmoid(u):
	return 1. / (1. + np.exp(-u))

def softmax(X):
	result = np.zeros_like(X)
	for i in xrange(X.shape[0]):
		row = X[i]
		result[i] = np.exp(row - max(row))
		result[i] /= np.sum(result[i])
	return result 

def soft_absolute(u):
	"""
	soft approx to |u|
	"""
	epsilon = 1e-8
	return np.sqrt(epsilon + u * u)

def share_gpu_data(data, return_type = None, borrow = True):
	shared_data = shared(np.asarray(data, dtype = config.floatX), 
						borrow = borrow)
	if return_type:
		shared_data = T.cast(shared_data, dtype = return_type)
	return shared_data

def sgd_optimize(n_epochs, n_train_batches, train_fn, validate_fn, params,
				patience = None, patience_increase = 2, 
				improvement_thr = 0.995, verbose = True):
	## parameters
	patience = patience or 10 * n_train_batches
	validation_freq = min(n_train_batches, patience / 2)
	best_params = None
	best_validation_error = np.inf

	for epoch in xrange(n_epochs):
		for batch in xrange(n_train_batches):
			train_error = train_fn(batch)
			niter = epoch * n_train_batches + batch 
			if (niter + 1) % validation_freq == 0:
				validation_error = validate_fn()
				if verbose:
					print ('epoch %i, batch %i/%i, validation_error %g, patience %i/%i' % 
						(epoch, batch, n_train_batches, validation_error, 
							niter, patience))
				if validation_error < best_validation_error:
					if validation_error < best_validation_error * improvement_thr:
						patience = max(niter * patience_increase, patience)
					best_validation_error = validation_error
					best_params = [p.get_value(borrow = True) for p in params]
			if patience <= niter:
				print 'out of patience ...'
				return best_params
	return best_params
