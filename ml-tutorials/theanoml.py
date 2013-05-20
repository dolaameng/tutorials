## Machine Learning Packages based on Theano
## The interface of the machine learning methods
## resembles sklearn BaseEstimator

import theano
import theano.tensor as T
import numpy as np
from sklearn.base import BaseEstimator
from collections import namedtuple

class LogisticRegression(BaseEstimator):
	"""
	Multiclass LogisticRegression, based on 
	the tutorial at 
	https://github.com/lisa-lab/DeepLearningTutorials/blob/master/code/logistic_sgd.py
	The optimization method can be either sgd (for large data) or cg (for smaller)
	"""
	def __init__(self, classes, optimizer = None, batch_size = 500, 
						learning_rate = 0.13, n_epochs = 1000):
		"""
		optimizer = {'sgd' or 'cg'}
		"""
		self.classes = classes 
		self.optimizer = optimizer
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.n_epochs = n_epochs
	def fit(self, train_raw_X, train_raw_y, 
			validation_raw_X = None, validation_raw_y = None):
		if validation_raw_X is None or validation_raw_y is None:
			validation_raw_X, validation_raw_y = train_raw_X, train_raw_y
		## build the shared objects
		train_X = self._share_data(train_raw_X)
		train_y = self._share_data(train_raw_y, dtype = 'int32')
		validation_X = self._share_data(validation_raw_X)
		validation_y = self._share_data(validation_raw_y, dtype = 'int32')
		## build functions based on symbolic variables
		## train_model parameterized by batch_index
		train_model = self._build_train_model(train_X, train_y)
		## validate_mode parameterized by batch_index
		validate_model = self._build_valiate_model(validation_X, validation_y)
		## optimize using built symbolic system
		## TODO - ??
		pass
	def predict(self, X):
		pass
	def predict_proba(self, X):
		pass
	def score(self, X, y):
		pass
	def _build_train_model(self, train_X, train_y):
		"""
		train_X = shared object of T.TensorObject
		train_y = shared object of T.TensorObject
		"""
		## need W, b, p_y_given_x, y_pred - X related
		## negative_log_likelihood, score - y related
		## g_W, g_b
		batch_size = self.batch_size
		index = T.lscalar('index')
		batch_X = train_X[index*batch_size:(index+1)*batch_size]
		batch_y = train_y[index*batch_size:(index+1)*batch_size]
		symbols = self._build_symbols()
		(W, b, p_y_given_x, y_pred, negative_log_likelihood, score) = symbols
		return 
		## TODO
	def _build_validate_model(self, validation_X, validation_y):
		"""
		validation_X = shared object of T.TensorObject
		valiation_y = shared object of T.TensorObject
		"""
		## need W, b, p_y_given_x, y_pred - X related
		## negative_log_likelihood, score - y related
		## g_W, g_b
		pass
	def _build_symbols(self):
		"""
		batch_X = TensorObject based on shared object 
		batch_y = TensorOjbect based on shared object
		"""
		batch_X = T.matrix('batch_X')
		batch_y = T.ivector('batch_y')
		## shape information
		n_feats = 1 #batch_X.shape[1]
		n_samples = 1 #batch_X.shape[0]
		n_classes = len(self.classes)
		classes = theano.shared(np.asarray(self.classes), 'classes')
		W = theano.shared(value=np.array((n_feats, n_classes), 
								dtype=theano.config.floatX),
							name = 'W', borrow = True)
		b = theano.shared(value=np.array((n_classes,), 
								dtype=theano.config.floatX),
							name = 'b', borrow = True)
		p_y_given_x = T.nnet.softmax(T.dot(batch_X, W) + b)
		y_pred = classes[T.argmax(p_y_given_x, axis = 1).reshape((-1, ))]
		negative_log_likelihood = -T.mean(T.log(p_y_given_x)[T.arange(n_classes), batch_y])
		score = T.mean(T.eq(y_pred, batch_y))
		return (W, b, p_y_given_x, y_pred, negative_log_likelihood, score)
	def _share_data(self, data, dtype = None):
		"""
		build shared data - mainly for GPU programming
		The SHARED RAW data must be of float dtype to be in GPU,
		but the return the shared object will of dtype 
		(e.g., for class lablels it will be 'int32')
		data = raw data (eg. np.array)
		return = theano shared object 
		"""
		dtype = dtype or theano.config.floatX
		## make it loadable on GPU
		shared_data = theano.shared(np.asarray(data, dtype = theano.config.floatX),
									borrow = True)
		return T.cast(shared_data, dtype)