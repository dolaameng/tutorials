## Machine Learning Packages based on Theano
## The interface of the machine learning methods
## resembles sklearn BaseEstimator

import theano
import theano.tensor as T
import numpy as np
from sklearn.base import BaseEstimator

class LogisticRegression(BaseEstimator):
	"""
	Multiclass LogisticRegression, based on 
	the tutorial at 
	https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/logistic_sgd.py
	The optimization method can be either sgd (for large data) or cg (for smaller)
	"""
	def __init__(self, optimizer = None, batch_size = 500, 
						learning_rate = 0.13, n_iters = 1000):
		"""
		optimizer = ??
		"""
		## meta
		self.rng = np.random
		self.optimizer = optimizer
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.n_iters = n_iters

		## symbolic variables and representations
		## borrow = True indicating a link between raw data and theano variable
		## value will be reset according to X.shape
		X = T.matrix('X')
		y = T.ivector('y')
		self.W = theano.shared(value = np.zeros((1, 1)), 
								name = 'W', borrow = True)
		self.b = theano.shared(value = np.zeros((1,)), 
								name = 'b', borrow = True)
		p_y_given_x = T.nnet.softmax(T.dot(X, self.W) + self.b)
		y_pred = T.argmax(p_y_given_x, axis = 1)
		negative_log_likelihood = -T.mean(T.log(p_y_given_x)[:, y])
		#negative_log_likelihood = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
		g_W, g_b = T.grad(cost = negative_log_likelihood, 
							wrt = [self.W, self.b])
		updates = [(self.W, self.W - self.learning_rate * g_W), 
				   (self.b, self.b - self.learning_rate * g_b)]	
		score = T.mean(T.eq(y_pred, y))

		## functions
		self.fit_fn = theano.function(inputs = [X, y], 
								outputs = negative_log_likelihood,
								updates = updates)
		self.predict_proba_fn = theano.function(inputs = [X],
								outputs = p_y_given_x)
		self.predict_fn = theano.function(inputs = [X],
								outputs = y_pred)
		self.score_fn = theano.function(inputs = [X, y],
								outputs = score)
	def fit(self, X, y):
		epoch, done_looping = 0, False
		while (epoch < self.n_iters) and (not done_looping):
			epoch = epoch + 1
			##TODO
	def predict(self, X):
		return self.predict_fn(X)
	def predict_proba(self, X):
		return self.predict_proba_fn(X)
	def score(self, X, y):
		return self.score_fn(X, y)
	def _shared_data(data, borrow = True):
		"""
		Use shared data model for memory efficiency among mini-batch learning
		and faster running on GPU 
		"""
		shared_data = theano.shared(np.asarray(data), borrow=borrow)
		return shared_data