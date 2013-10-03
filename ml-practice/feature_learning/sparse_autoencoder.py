
from util import *


class SparseAutoEncoder(BaseEstimator):
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
			likelihood = np.mean(np.sum((data_X - Z) ** 2, axis = 1))
			l2norm = np.sum(W1**2) + np.sum(W2 ** 2)
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

