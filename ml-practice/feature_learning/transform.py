from util import *

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
		#X0 = (X - self.mean_[:, np.newaxis] 
		#		if self.mean_axis_ == 1 else X - self.mean_)
		X0 = (X - np.mean(X, axis = self.mean_axis_)[:, np.newaxis] 
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