from util import *

class SparseFilter(BaseEstimator):
	"""
	Unlike sparse auto encoder, and similiar to linear decoder,
	the SparseFilter does not need the inputs to be scaled to [0, 1]
	"""
	def __init__(self, n_vis, n_hid):
		self.epsilon = 1e-8
		self.n_vis = n_vis
		self.n_hid = n_hid
		self.params = None 
	def get_objective_fn(self, data_X):
		def _objective(param_value):
			self.params = param_value
			W = self.restore_params()
			Y = soft_absolute(np.dot(data_X, W))
			## noramlize column-wise
			YY = Y / np.sqrt(np.sum(Y*Y, axis = 0) + self.epsilon)
			## nomralize row-wise
			YYY = YY / (np.sqrt(np.sum(YY*YY, axis = 1) + self.epsilon)[:, np.newaxis])
			cost = np.sum(YYY)
			return cost 
		return _objective
	def initial_params_value(self):
		"""
		W is params
		"""
		W = np.random.randn(self.n_vis, self.n_hid)
		return W
	def restore_params(self):
		return self.params
	def fit(self, data_X, *args, **kwargs):
		"""
		We do NOT even need data_X to be scaled to [0, 1]
		"""
		f = self.get_objective_fn(data_X)
		param0 = self.initial_params_value()
		optimal_param = autodiff.optimize.fmin_l_bfgs_b(f, param0, *args, **kwargs)
		self.params = optimal_param
		return self
	def transform(self, data_X):
		"""
		data_X do NOT even need to be scaled to [0, 1]
		"""
		W = self.restore_params()
		Y = soft_absolute(np.dot(data_X, W))
		## noramlize column-wise
		YY = Y / np.sqrt(np.sum(Y*Y, axis = 0) + self.epsilon)
		## nomralize row-wise
		YYY = YY / (np.sqrt(np.sum(YY*YY, axis = 1) + self.epsilon)[:, np.newaxis])
		return YYY