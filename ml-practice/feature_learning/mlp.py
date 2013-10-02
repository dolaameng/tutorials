from util import *

class SoftmaxMLP(BaseEstimator):
	def __init__(self, l2_coeff,
			layer_dims, layer_params = None):
		"""
		layer_params should be in format [(W1, b1), (W2, b2), ...]
		layer_dims should be in format [nin, n1, n2, ..., nout]
		"""
		self.l2_coeff = l2_coeff
		self.layer_dims = layer_dims
		if layer_params:
			assert len(layer_dims)-1 == len(layer_params)
			self.params = self.flatten_params(layer_params)
		else:
			self.params = self.initialize_params()
	def flatten_params(self, layer_params):
		"""
		Use flatten to make a copy
		"""
		param_len = sum(map(lambda pair : pair[0] * pair[1] + pair[1], 
							zip(self.layer_dims[:-1], self.layer_dims[1:])))
		params = np.empty(param_len)
		acc_len = 0
		for (W, b) in layer_params, self.layer_dims[1:-1]:
			n1, n2 = W.shape
			params[acc_len:acc_len+n1*n2] = W.ravel()
			params[acc_len+n1*n2:acc_len+n1*n2+n2] = b 
			acc_len += n1*n2+n2 
		return params 
	def initialize_params(self):
		param_len = sum(map(lambda pair : pair[0] * pair[1] + pair[1], 
							zip(self.layer_dims[:-1], self.layer_dims[1:])))
		params = np.zeros(param_len)

		acc_len = 0
		for n1, n2 in zip(self.layer_dims[:-1], self.layer_dims[1:]):
			params[acc_len:acc_len+n1*n2] = np.random.uniform(
													low = -4.*np.sqrt(6./(n1+n2)),
													high = 4.*np.sqrt(6./(n1+n2)), 
													size = n1*n2)
			## self.params[acc_len+n1*n2:acc_len+n1*n2+n2] = zeros...
			acc_len += n1*n2+n2
		return params 

	def restore_params(self):
		"""
		return [(W1, b1), (W2, b2), ...]
		"""
		layer_params = []
		acc_len = 0
		for n1, n2 in zip(self.layer_dims[:-1], self.layer_dims[1:]):
			W = self.params[acc_len:acc_len+n1*n2].reshape((n1, n2))
			b = self.params[acc_len+n1*n2:acc_len+n1*n2+n2]
			acc_len += n1*n2 + n2
			layer_params.append((W, b))
		return layer_params 
	def get_objective_fn(self, X, y):
		def _objective(param_value):
			self.params = param_value
			layer_params = self.restore_params()

			l2norm = 0
			## hidden layers forwarding
			Y = X 
			for i, (W, b) in enumerate(layer_params):
				Y = sigmoid(np.dot(Y, W) + b)
				l2norm += np.sum(W ** 2)
			## output layers
			Z = softmax(Y)
			## negative log-likelihood
			likelihood = -np.mean(np.log(Z)[np.arange(Z.shape[0]), self.y_indices_])
			cost = likelihood + self.l2_coeff * l2norm
			return cost

		return _objective
	def fit(self, X, y):
		## TODO
		self.classes_ = np.unique(y)
		self.y_indices_ = np.asarray([np.nonzero(yi == self.classes_)[0][0] 
								for yi in y])
		pass
		## TODO

if __name__ == '__main__':
	smlp = SoftmaxMLP([10, 5, 5, 10])
	print smlp.layer_dims
	print smlp.params.shape, 10*5+5 + 5*5+5 + 5*10+10 
	(W1, b1), (W2, b2), (W3, b3) = smlp.restore_params()
	print W1.shape, b1.shape
	print W2.shape, b2.shape
	print W3.shape, b3.shape
	print 'all tests passed...'