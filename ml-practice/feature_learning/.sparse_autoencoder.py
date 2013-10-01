
from util import *


class SparseAuoEncoder(BaseEstimator):
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

@deprecated
class SGDSparseAutoEncoder(BaseEstimator):
	"""
	practically not very useful
	"""
	def __init__(self, n_vis, n_hid, 
				sparsity, l2_coeff, sparsity_coeff, 
				batch_size, learning_rate = 0.01,
				X = None, params = None):
		self.n_vis = n_vis
		self.n_hid = n_hid 
		self.sparsity = sparsity
		self.l2_coeff = l2_coeff
		self.sparsity_coeff = sparsity_coeff
		self.batch_size = batch_size
		self.learning_rate = learning_rate

		self.X = X or T.matrix('sae.X')
		self.params = params or shared(self.initial_params_value(), name = 'sae.params')

		self.W1 = self.params[:n_vis*n_hid].reshape((n_vis, n_hid))
		self.W2 = self.params[n_vis*n_hid:2*n_vis*n_hid].reshape((n_hid, n_vis))
		self.b1 = self.params[2*n_vis*n_hid:2*n_vis*n_hid+n_hid]
		self.b2 = self.params[2*n_vis*n_hid+n_hid:]
		self.Y = T.nnet.sigmoid(T.dot(self.X, self.W1) + self.b1)
		self.Z = T.nnet.sigmoid(T.dot(self.Y, self.W2) + self.b2)
		self.rhos = T.mean(self.Y, axis = 0)

		self.likelihood = T.mean(T.sum((self.X - self.Z)**2, axis = 1))
		self.l2norm = T.sum(self.W1**2) + T.sum(self.W2**2)
		self.sparsity_term = T.sum(self.sparsity * T.log(self.sparsity/self.rhos) 
                                   + (1-self.sparsity)*T.log((1-self.sparsity)/(1-self.rhos)))
		self.cost = (self.likelihood 
					+ self.l2_coeff * self.l2norm 
					+ self.sparsity_coeff * self.sparsity_term)
		self.gparams = T.grad(self.cost, wrt = self.params)

	def get_train_fn(self, data_X):
		shared_X = share_gpu_data(data_X)
		index = T.lscalar('sae.index')
		updates = [(self.params, self.params - self.learning_rate * self.gparams)]
		return function(inputs = [index],
						outputs = self.cost,
						updates = updates,
						givens = {
							self.X: shared_X[index*self.batch_size:(index+1)*self.batch_size]
						})

	def get_validate_fn(self, data_X):
		shared_X = share_gpu_data(data_X)
		return function(inputs = [],
						outputs = self.cost, 
						givens = {self.X: shared_X})

	def fit(self, train_X, valid_X = None, n_epochs = 1000, *args, **kwargs):
		valid_X = valid_X or train_X
		train_fn = self.get_train_fn(train_X)
		validate_fn = self.get_validate_fn(valid_X)
		n_train_batches = train_X.shape[0] / self.batch_size
		optimal_param_value = sgd_optimize(n_epochs, n_train_batches, 
			train_fn = train_fn, validate_fn = validate_fn, 
			params = [self.params], *args, **kwargs)
		self.params.set_value(optimal_param_value[0])

	def initial_params_value(self):
		"""
		good initialization of param values 
		initialization as float64 for l_bfgs_b legacy code
		"""
		n_vis, n_hid = self.n_vis, self.n_hid
		params = np.zeros(2*n_vis*n_hid + n_hid + n_vis, dtype=config.floatX)
		## W1 and W2
		params[:2*n_vis*n_hid] = np.random.uniform(
									low = -4. * np.sqrt(6. / (n_vis + n_hid)),
									high = 4. * np.sqrt(6. / (n_vis + n_hid)), 
									size = 2*n_vis*n_hid)
		## b1 and b2 as zeros - leave them as is
		return params 
