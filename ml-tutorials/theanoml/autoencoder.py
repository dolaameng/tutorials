from formula import share_data
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

################# Denoising Auto Encoder ########################

class DenoisingAutoEncoderFormula(object):
	"""
	Contractive Auto Encoder.
     References :
   		- P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   		Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   		2008
   		- Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   		Training of Deep Networks, Advances in Neural Information Processing
   		Systems 19, 2007
	"""
	def __init__(self,  n_visible, n_hidden,
					corruption_level = 0.1,
					X = None, W = None, bhid = None, bvis = None, 
					rng = None):
		"""
		rng = np.random.RandomState
		n_visible = n_input_feats
		n_hidden = n_hidden_nodes
		batch_size = n_sample of a batch
		x
		W, W.prime = hidden params, W.T
		bhid, bvis = forward and backward bias through hidden layer  
		"""
		rng = rng or np.random.RandomState(0)
		self.n_visible = n_visible
		self.n_hidden = n_hidden
		self.X = X or T.matrix(name = 'X')
		self.corruption_level = corruption_level
		if not W:
			W_bound = 4. * np.sqrt(6. / (n_hidden + n_visible))
			W = theano.shared(value = np.asarray(rng.uniform(
										low = -W_bound,
										high = W_bound,
										size = (n_visible, n_hidden)),
										dtype = theano.config.floatX),
								name = 'W', 
								borrow = True)
		if not bvis:
			bvis = theano.shared(value = np.zeros(n_visible, 
                                    dtype = theano.config.floatX),
                    borrow = True)
		if not bhid:
			bhid = theano.shared(value = np.zeros(n_hidden,
                                    dtype = theano.config.floatX), 
                    borrow = True)
		self.W = W
		self.W_prime = self.W.T
		self.b = bhid
		self.b_prime = bvis
		self.theano_rng = RandomStreams(rng.randint(2 ** 30))
		self.params = [self.W, self.b, self.b_prime]
	def corrupted_input(self, X):
		return self.theano_rng.binomial(size = X.shape, n = 1, 
					p = 1 - self.corruption_level, 
					dtype = theano.config.floatX) * X 
	def bound_input(self, X):
		self.X = X
	def hidden_value(self, X):
		return T.nnet.sigmoid(T.dot(X, self.W) + self.b)
	def reconstructed_input(self, hidden):
		return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
	def cost(self):
		tilde_X = self.corrupted_input(self.X)
		y = self.hidden_value(tilde_X)
		z = self.reconstructed_input(y)
		## we sum over the size of a datapoint; if we are using minibatches,
		## L will be a vector, with one entry per example in minibatch
		L = -T.sum(self.X * T.log(z) + (1-self.X)*T.log(1-z), axis = 1)
		return T.mean(L)


class DenoisingAutoEncoder(object):
	def __init__(self, corruption_level = 0.1,  n_hidden = 500, batch_size = 10,
					n_epochs = 20, learning_rate = 0.01, verbose = True):
		self.corruption_level = corruption_level
		self.batch_size = batch_size
		self.n_hidden = n_hidden
		self.n_epochs = n_epochs
		self.learning_rate = learning_rate
		self.verbose = verbose
		self.da_ = None
	def fit(self, X):
		n_samples, n_feats = X.shape
		n_train_batches = n_samples / self.batch_size
		v_X = share_data(X)

		self.da_ = DenoisingAutoEncoderFormula(n_visible = n_feats, 
									n_hidden = self.n_hidden, 
									corruption_level = self.corruption_level)
		index = T.lscalar('index')
		cost = self.da_.cost()
		gparams = T.grad(cost, self.da_.params)
		updates = [(param, param - self.learning_rate*gparam) 
					for (param, gparam) in zip(self.da_.params, gparams)]
		train_model = theano.function(inputs = [index], 
							outputs = cost,
							updates = updates,
							givens = {
								self.da_.X: v_X[index*self.batch_size:(index+1)*self.batch_size]
						})
		for epoch in xrange(self.n_epochs):
			cost = [train_model(i) for i in xrange(n_train_batches)]
			if self.verbose:
				print 'training epoch %d, recall cost %f ' % (
					epoch, np.mean(cost))
		return self
	def transform(self, X):
		v_X = share_data(X)
		return self.da_.hidden_value(v_X).eval()
		##return self.da_.hidden_value(self.da_.corrupted_input(v_X)).eval()
	def fit_transform(self, X):
		return self.fit(X).transform(X)


################## Contractive Auto Encoder #####################

class ContractiveAutoEncoderFormula(object):
	"""
	Contractive Auto Encoder.
     References :
       - S. Rifai, P. Vincent, X. Muller, X. Glorot, Y. Bengio: Contractive
       Auto-Encoders: Explicit Invariance During Feature Extraction, ICML-11

       - S. Rifai, X. Muller, X. Glorot, G. Mesnil, Y. Bengio, and Pascal
         Vincent. Learning invariant features through local space
         contraction. Technical Report 1360, Universite de Montreal

       - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
       Training of Deep Networks, Advances in Neural Information Processing
       Systems 19, 2007
	"""
	def __init__(self,  n_visible, n_hidden, batch_size = 1,
					contraction_level = 0.1,
					X = None, W = None, bhid = None, bvis = None, rng = None):
		"""
		rng = np.random.RandomState
		n_visible = n_input_feats
		n_hidden = n_hidden_nodes
		batch_size = n_sample of a batch
		x
		W, W.prime = hidden params, W.T
		bhid, bvis = forward and backward bias through hidden layer  
		"""
		rng = rng or np.random.RandomState(0)
		self.n_visible = n_visible
		self.n_hidden = n_hidden
		self.batch_size = batch_size
		self.X = X or T.matrix(name = 'X')
		self.contraction_level = contraction_level
		if not W:
			W_bound = 4. * np.sqrt(6. / (n_hidden + n_visible))
			W = theano.shared(value = np.asarray(rng.uniform(
										low = -W_bound,
										high = W_bound,
										size = (n_visible, n_hidden)),
										dtype = theano.config.floatX),
								name = 'W', 
								borrow = True)
		if not bvis:
			bvis = theano.shared(value = np.zeros(n_visible, 
                                    dtype = theano.config.floatX),
                    borrow = True)
		if not bhid:
			bhid = theano.shared(value = np.zeros(n_hidden,
                                    dtype = theano.config.floatX), 
                    borrow = True)
		self.W = W
		self.W_prime = self.W.T
		self.b = bhid
		self.b_prime = bvis
		self.params = [self.W, self.b, self.b_prime]
	def bound_input(self, X):
		self.X = X
	def hidden_value(self, X):
		return T.nnet.sigmoid(T.dot(X, self.W) + self.b)
	def jacobian(self, hidden, W):
		reshaped_hidden = T.reshape(hidden * (1-hidden), (self.batch_size, 1, self.n_hidden))
		reshaped_W = T.reshape(W, (1, self.n_visible, self.n_hidden))
		return reshaped_hidden * reshaped_W
	def reconstructed_input(self, hidden):
		return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
	def cost(self):
		y = self.hidden_value(self.X)
		z = self.reconstructed_input(y)
		J = self.jacobian(y, self.W)
		self.L_rec = - T.sum(self.X*T.log(z) + (1-self.X)*T.log(1-z), axis = 1)
		self.L_jacob = T.sum(J ** 2) / self.batch_size
		combined_cost = T.mean(self.L_rec) + self.contraction_level * T.mean(self.L_jacob)
		return combined_cost

class ContractiveAutoEncoder(object):
	def __init__(self, contraction_level = 0.1, batch_size = 10, n_hidden = 500, 
					n_epochs = 20, learning_rate = 0.01, verbose = True):
		self.contraction_level = contraction_level
		self.batch_size = batch_size
		self.n_hidden = n_hidden
		self.rng = np.random.RandomState(0)
		self.n_epochs = n_epochs
		self.learning_rate = learning_rate
		self.verbose = verbose
		self.ca_ = None
	def fit(self, X):
		n_samples, n_feats = X.shape
		n_train_batches = n_samples / self.batch_size
		v_X = share_data(X)

		self.ca_ = ContractiveAutoEncoderFormula(n_visible = n_feats, 
									n_hidden = self.n_hidden, 
									batch_size = self.batch_size,
									contraction_level = self.contraction_level)
		index = T.lscalar('index')
		cost = self.ca_.cost()
		gparams = T.grad(cost, self.ca_.params)
		updates = [(param, param - self.learning_rate*gparam) 
					for (param, gparam) in zip(self.ca_.params, gparams)]
		train_model = theano.function(inputs = [index], 
							outputs = [T.mean(self.ca_.L_rec), self.ca_.L_jacob],
							updates = updates,
							givens = {
								self.ca_.X: v_X[index*self.batch_size:(index+1)*self.batch_size]
						})
		for epoch in xrange(self.n_epochs):
			recall, jacob = zip(*[train_model(i) for i in xrange(n_train_batches)])
			if self.verbose:
				print 'training epoch %d, recall cost %f, jacobian norm %f ' % (
					epoch, np.mean(recall), np.mean(np.sqrt(jacob))
				)
		return self
	def transform(self, X):
		v_X = share_data(X)
		return self.ca_.hidden_value(v_X).eval()
	def fit_transform(self, X):
		return self.fit(X).transform(X)