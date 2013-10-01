import numpy as np
import scipy as sp 
import autodiff 
import pylab as plt 
from sklearn.base import BaseEstimator

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
	axes = axes.flatten()
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

def normalize_image01(image_features):
	"""
	normalize **Natural Images** pixel values to [0, 1], 
	based on the assumption that different images may have different 
	exposure level (mean) but the same variance (std).
	The normalization is done by (1) removing the mean PER Images 
	(2) translate pixels to [-1, +1] by dividing +/- 3stds
	(3) make it into [0.1, 0.9] by a linear transformation 

	param: image_features - a 2D ndarray nrows x nfeats, feats are 
	flattened image vector or a 3D ndarray nimags x nrows x ncols 

	This is quite a **hard** way of normalizing images compared to 
	PCA and ZCA whitening
	"""
	## (1) mean per image
	image_means = np.mean(image_features, axis = 1)
	image_feats = image_features - image_means[:, np.newaxis]
	## (2) global std of all image/patches pool
	image_std = np.std(image_feats)
	image_feats = np.maximum(np.minimum(image_feats, 3*image_std), -3*image_std)
	image_feats /= 3*image_std 
	## (3)
	image_feats = (image_feats + 1.) * 0.4 + 0.1
	return image_feats 

class PCATransform(BaseEstimator):
	"""
	It is different from the PCA class in sklearn in that 
	the zero-mean happends row-wise (per instance) instead of 
	col-wise (per feature )
	"""
	def __init__(self, ncomponents, epsilon = 0.):
		"""
		ncomponents: # of PC components to reserve
		epsilon: regularization parameter for smoothing whitening 
		"""
		self.ncomponents = ncomponents
		self.epsilon = epsilon
	def fit(self, X, whiten = False, axis = 1):
		"""
		X: data matrix of nsample x nfeats
		axis decides the way of doing zero-mean, axis = 1
		usually used for image data, axis = 0 for other types
		"""
		self.mean_axis_ = axis 
		self.whiten_ = whiten
		## zero mean
		self.mean_ = np.mean(X, axis = axis)
		X0 = (X - self.mean_[:, np.newaxis] 
				if self.mean_axis_ == 1 else X - self.mean_)
		## calculate covariance and SVD
		cov = np.dot(X0.T, X0) / X0.shape[0]
		U, S, V = np.linalg.svd(cov)
		self.U_ = U
		self.components_ = (U 
						if not whiten 
						else np.dot(U, np.diag(1. / np.sqrt(S + self.epsilon))))

		return self
	def transform(self, X):
		X0 = (X - self.mean_[:, np.newaxis] 
				if self.mean_axis_ == 1 else X - self.mean_)
		return np.dot(X0, 
					self.components_[:, :self.ncomponents])
	def fit_transform(self, X, whiten = False, axis = 1):
		return self.fit(X, whiten, axis).transform(X)


class ZCATransform(BaseEstimator):
	"""
	It is different from the PCA class in sklearn in that 
	the zero-mean happends row-wise (per instance) instead of 
	col-wise (per feature )
	"""
	def __init__(self, epsilon = 0.):
		"""
		ncomponents: # of PC components to reserve
		epsilon: regularization parameter for smoothing whitening 
		NO REDUCTION IS USED for ZCA, so ncomponents is not in use 
		"""
		self.epsilon = epsilon
	def fit(self, X, whiten = False, axis = 1):
		self.ncomponents = X.shape[1]
		self.pca_ = PCATransform(self.ncomponents, self.epsilon)
		self.pca_.fit(X, whiten, axis)
		return self
	def transform(self, X):
		X_pca = self.pca_.transform(X)
		X_zca = np.dot(X_pca, self.pca_.U_.T)
		return X_zca
	def fit_transform(self, X, whiten = False, axis = 1):
		return self.fit(X, whiten, axis).transform(X)

def sigmoid(u):
	return 1. / (1. + np.exp(-u))

class SparseAuoEncoder(object):
	"""
	params: W1, b1, W2, b2
	objective: least_square + lambda*l2_norm + beta*sparsity
	where least_square = .5 * mean(sum((X - X_hat) ** 2, axis = 1))
	l2_norm = .5 * (sum(W1**2) + sum(W2**2)) 
	rhos = mean(Y), where Y is the avtivaion
	sparsity = sum(rho * log(rho / rhos) + (1-rho) * log((1-rho)/(1-rhos)))
	"""
	def __init__(self, n_vis, n_hid, sparsity, l2_coeff, sparsity_coeff):
		"""
		n_vis, n_hid: input and hidden dimensionality
		sparsity: reference sparisty for hidden neurons, (rho in the formular)
		l2_coeff: coefficient for l2 norm penalty
		sparsity_coeff: coefficient for sparsity penalty

		self.params: theta params 
		"""
		self.n_vis = n_vis
		self.n_hid = n_hid 
		self.sparsity = sparsity
		self.l2_coeff = l2_coeff
		self.sparsity_coeff = sparsity_coeff
		self.params = None 
	def initial_params_value(self):
		"""
		good initialization of param values 
		initialization as float64 for l_bfgs_b legacy code
		"""
		n_vis, n_hid = self.n_vis, self.n_hid
		params = np.zeros(2*n_vis*n_hid + n_hid + n_vis)
		## W1 and W2
		params[:2*n_vis*n_hid] = np.random.uniform(
									low = -4. * np.sqrt(6. / (n_vis + n_hid)),
									high = 4. * np.sqrt(6. / (n_vis + n_hid)), 
									size = 2*n_vis*n_hid)
		## b1 and b2 as zeros - leave them as is
		return params 
	def restore_params(self):
		n_vis, n_hid = self.n_vis, self.n_hid 
		W1 = self.params[:n_vis*n_hid].reshape((n_vis, n_hid))
		W2 = self.params[n_vis*n_hid:2*n_vis*n_hid].reshape((n_hid, n_vis))
		b1 = self.params[2*n_vis*n_hid:2*n_vis*n_hid+n_hid]
		b2 = self.params[2*n_vis*n_hid+n_hid:]
		return W1, W2, b1, b2 
	def get_objective_fn(self, data_X):
		"""
		bind the formula to data 
		"""
		def _objective(param_value):
			## restore params shapes
			sparsity = self.sparsity
			l2_coeff = self.l2_coeff
			sparsity_coeff = self.sparsity_coeff

			self.params = param_value
			W1, W2, b1, b2 = self.restore_params()
			## hidden value Y
			Y = sigmoid(np.dot(data_X, W1) + b1)
			## sparsity of hidden neuros
			rhos = np.mean(Y, axis = 0)
			## reconstructed output
			Z = sigmoid(np.dot(Y, W2) + b2)
			## cost = likelihood + coeff * l2norm + coeff * sparsity_term
			likelihood = .5 * np.mean(np.sum((data_X - Z) ** 2, axis = 1))
			l2norm = .5 * (np.sum(W1**2) + np.sum(W2 ** 2))
			sparsity_term = np.sum(sparsity * np.log(sparsity / rhos)
							+ (1-sparsity) * np.log((1-sparsity) / (1-rhos)))
			cost = (likelihood 
					+ l2_coeff * l2norm 
					+ sparsity_coeff * sparsity_term)
			return cost 

		return _objective
	def fit(self, data_X, *args, **kwargs):
		"""
		data_X should be scaled to [0, 1]
		"""
		f = self.get_objective_fn(data_X)
		param0 = self.initial_params_value()
		optimal_param = autodiff.optimize.fmin_l_bfgs_b(f, param0, *args, **kwargs)
		self.params = optimal_param
		return self
	def transform(self, data_X):
		"""
		data_X should be scaled to [0, 1]. 
		As feature learning step, transfomation of sparse auto encoder 
		will be the activation of hidden neurouns
		"""
		W1, W2, b1, b2 = self.restore_params()
		Y = sigmoid(np.dot(data_X, W1) + b1)
		return Y 