import theano.tensor as T 
from theano import shared, function, config
from util import *

class LogisticRegressionLayer(object):
	def __init__(self, nin, nout, l2_coeff,
		X = None, y = None, 
		theta = None):
		self.nin = nin 
		self.nout = nout 
		self.X = X or T.matrix('X')
		self.y = y or T.ivector('y')
		self.theta = theta or shared(value = np.zeros((nin*nout+nout), 
										dtype = config.floatX),
									name = 'theta', borrow = True)		
		self.W = self.theta[:nin*nout].reshape((nin, nout))
		self.b = self.theta[nin*nout:nin*nout+nout].reshape((nout, ))

		self.p_y_given_x = T.nnet.softmax(T.dot(self.X, self.W) + self.b)
		self.yhat = T.argmax(self.p_y_given_x, axis = 1)
		self.nll = -T.mean(T.log(self.p_y_given_x[T.arange(self.y.shape[0]), 
															self.y]))
		self.l2norm = T.sum(self.W ** 2)
		self.cost = self.nll + l2_coeff * self.l2norm 
		self.gparams = T.grad(self.cost, wrt = self.theta)

	def get_f_and_g(self, data_X, data_y):
		shared_X = share_gpu_data(data_X)
		shared_y = share_gpu_data(data_y, return_type = 'int32')
		self.cost_fn = function(inputs = [], 
							outputs = self.cost, 
							givens = {
								self.X: shared_X,
								self.y: shared_y
							})
		self.gradient_fn = function(inputs = [],
							outputs = self.gparams,
							givens = {
								self.X: shared_X,
								self.y: shared_y 
							})
		def _objective(theta_value):
			theta_value = theta_value.astype(config.floatX)
			self.theta.set_value(theta_value)
			return self.cost_fn().astype('float64')
		def _gradient(theta_value):
			theta_value = theta_value.astype(config.floatX)
			self.theta.set_value(theta_value)
			return self.gradient_fn().astype('float64')
		return _objective, _gradient