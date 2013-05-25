## Machine Learning Packages based on Theano
## The interface of the machine learning methods
## resembles sklearn BaseEstimator

import theano
import theano.tensor as T
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cross_validation import train_test_split


class LogisticRegression(BaseEstimator):
	def __init__(self, classes, validation_size = 0.2, 
				optimizer = 'sgd', n_epochs = None, batch_size = None,
				learning_rate = None, verbose = True):
		"""
		optimizer = {'sgd', 'cg'}, generally 'sgd' for large scale and 'cg' for smaller data
		n_epochs = number of iterations for minibatch learning, 1000 for sgd, 50 for cg
		batch_size = size of the minibatch
		learning_rate = only effective for sgd learning
		"""
		if not all(classes == range(len(classes))):
			raise RuntimeError('classes need to be coded as exactly range(n_class)')
		self.classes = classes
		self.validation_size = validation_size
		self.optimize = self._sgd if optimizer == 'sgd' else self._cg
		self.n_epochs = n_epochs
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.verbose = verbose
		self.W_, self.b_ = None, None
	def fit(self, X, y):
		## re-initialize the parameters anyway
		self.n_feats =  X.shape[1]
		self.n_classes = len(self.classes)
		## initialize parameters
		if self.optimize == self._cg:
			self.theta_ = theano.shared(value = np.zeros(self.n_feats*self.n_classes+self.n_classes, 
													dtype = theano.config.floatX),
									name = 'theta',
									borrow = True)
			self.W_ = self.theta_[:self.n_feats*self.n_classes].reshape((self.n_feats, self.n_classes))
			self.b_ = self.theta_[self.n_feats*self.n_classes:]
		elif self.optimize == self._sgd:
			self.W_ = theano.shared(value = np.zeros((self.n_feats, self.n_classes),
									dtype = theano.config.floatX),
							name = 'W', borrow = True)
			self.b_ = theano.shared(value = np.zeros((self.n_classes),
									dtype = theano.config.floatX),
							name = 'W', borrow = True)
		else:
			raise RuntimeError('Unimplemented Optimization Method')
		## optimize
		self.optimize(X, y)
		return self 
	def partial_fit(self, X, y):
		"""
		increamental learning model for new data
		"""
		## only re-initialize parameter the first time
		if self.W_ is None or self.b_ is None:
			return self.fit(X, y)
		else:
			## optimize directly
			self.optimize(X, y)
			return self
	def predict(self, X):
		return self._predict(X)[0]
	def predict_proba(self, X):
		return self._predict(X)[1]
	def score(self, X, y):
		cost, error = self._build_symbols(X, y)
		return error.eval()
	def _predict(self, X):
		v_X = self._share_data(X)
		predict_model = self._build_predict_model(v_X)
		y_pred, p_y_given_x = predict_model()
		return y_pred, p_y_given_x
	def _sgd(self, X, y):
		learning_rate = self.learning_rate or 0.13
		n_epochs = self.n_epochs or 1000
		batch_size = self.batch_size or 600
		print 'Training LogisticRegression modle with SGD ...'
		## split and share data
		train_X, validation_X, train_y, validation_y = train_test_split(
									X, y, test_size = self.validation_size)
		v_train_X = self._share_data(train_X)
		v_validation_X = self._share_data(validation_X)
		v_train_y = self._share_data(train_y, dtype='int32')
		v_validation_y = self._share_data(validation_y, dtype='int32')
		## shape information
		## for params used to interact raw_data and tensor_data
		## get them as early as possible
		## symoblic model functions
		train_model = self._build_train_model(v_train_X, v_train_y, 
							batch_size, learning_rate)
		validate_model = self._build_validate_model(v_validation_X, 
								v_validation_y, batch_size)
		## number of batches
		n_train_batches = v_train_X.get_value(borrow=True).shape[0] / batch_size
		n_validation_batches = v_validation_X.get_value(borrow=True).shape[0] / batch_size
		## iterative optimization
		## ALGORITHM: iter through epoch -> iter through mini-batch in train
		## check validation performance if it is time; 
		## increase patience limit if validation performance is getting significantly better
		## stop if we run out of patience
		## ITERATION PARAMETERS
		## look at this many examples regardless
		patience = 5000
		## how much more patience gained when a new significant best achieved
		patience_increase = 2
		## how much improvement is considered significant
		improvement_threshold = 0.995
		## frequency of doing validation through iterative optimization
		## now we check it every mini_train_batch
		validation_frequency = min(n_train_batches, patience / 2)
		## iterative process
		epoch = 0
		out_of_patience = False
		best_validation_error = np.inf 
		best_params = self.W_, self.b_
		## each epoch till running out o patience
		while (epoch < n_epochs) and (not out_of_patience):
			epoch += 1
			## each mini train batch
			for minibatch_index in xrange(n_train_batches):
				## train the model
				minibatch_cost = train_model(minibatch_index)
				## update the total iteration number
				iter = (epoch - 1) * n_train_batches + minibatch_index
				## do validation when it is time
				if (iter+1) % validation_frequency == 0:
					## get the current validation error rate
					this_validation_error = np.mean([validate_model(i)
													for i in xrange(n_validation_batches)])
					if self.verbose:
						print 'epoch %i, minibatch %i/%i, validation error %f %%' % (
							epoch, minibatch_index+1, n_train_batches,
							this_validation_error * 100.
						)
					## increase the patience if a significant improvement is found
					if this_validation_error < best_validation_error:
						if this_validation_error < best_validation_error * improvement_threshold:
							patience = max(patience, iter*patience_increase)
						best_validation_error = this_validation_error
						best_params = [self.W_, self.b_]
				## if running out of patience, quit the iterative optimization
				if patience <= iter:
					out_of_patience = True
					break
		if self.verbose:
			print 'Optimization complete with best validation score %f %%' % (best_validation_error * 100.)
		## save the best found params based on the validation performance
		self.W_, self.b_ = best_params

	def _cg(self, X, y):
		n_epochs = self.n_epochs or 50
		batch_size = self.batch_size or 600
		print 'Training LogisticRegression modle with CG ...'
		## split and share data
		train_X, validation_X, train_y, validation_y = train_test_split(
									X, y, test_size = self.validation_size)
		v_train_X = self._share_data(train_X)
		v_validation_X = self._share_data(validation_X)
		v_train_y = self._share_data(train_y, dtype='int32')
		v_validation_y = self._share_data(validation_y, dtype='int32')
		## build symoblic models
		validate_model = self._build_validate_model(v_validation_X, 
					v_validation_y, batch_size)
		train_cost = self._build_train_cost(v_train_X,
					v_train_y, batch_size)
		train_grad = self._build_train_grad(v_train_X,
					v_train_y, batch_size)
		## number of batches
		n_train_batches = v_train_X.get_value(borrow=True).shape[0] / batch_size
		n_validation_batches = v_validation_X.get_value(borrow=True).shape[0] / batch_size
		## build helper functions for scipy optimize
		def train_fn(theta_value):
			self.theta_.set_value(theta_value, borrow = True)
			train_loss = np.mean([train_cost(i) for i in xrange(n_train_batches)])
			return train_loss
		def train_fn_grad(theta_value):
			self.theta_.set_value(theta_value, borrow = True)
			grad = sum([train_grad(i) for i in xrange(n_train_batches)]) / n_train_batches
			return grad
		def callback(theta_value):
			self.theta_.set_value(theta_value, borrow = True)
			validation_loss = np.mean([validate_model(i) for i in xrange(n_validation_batches)])
			print 'validation error', validation_loss
		import scipy.optimize
		best_theta = scipy.optimize.fmin_cg(
			f = train_fn,
			x0 = np.zeros((self.n_feats+1)*self.n_classes, dtype = v_train_X.dtype),
			fprime = train_fn_grad,
			callback = callback if self.verbose else None,
			disp = 0,
			maxiter = n_epochs
		)
		self.theta_.set_value(best_theta, borrow=True)
	def _build_train_model(self, v_train_X, v_train_y, batch_size, learning_rate):
		index = T.lscalar()
		X = v_train_X[index * batch_size: (index + 1) * batch_size]
		y = v_train_y[index * batch_size: (index + 1) * batch_size]
		cost, error = self._build_symbols(X, y)
		g_W, g_b = T.grad(cost = cost, wrt = [self.W_, self.b_])
		updates = [(self.W_, self.W_ - learning_rate*g_W), 
				   (self.b_, self.b_ - learning_rate*g_b)]
		train_model = theano.function(inputs = [index],
						outputs = cost, 
						updates = updates)
		return train_model
	def _build_train_cost(self, v_train_X, v_train_y, batch_size):
		index = T.lscalar()
		X = v_train_X[index * batch_size: (index + 1) * batch_size]
		y = v_train_y[index * batch_size: (index + 1) * batch_size]
		cost, error = self._build_symbols(X, y)
		train_cost = theano.function(inputs = [index],
						outputs = cost)
		return train_cost
	def _build_train_grad(self, v_train_X, v_train_y, batch_size):
		index = T.lscalar()
		X = v_train_X[index * batch_size: (index + 1) * batch_size]
		y = v_train_y[index * batch_size: (index + 1) * batch_size]
		cost, error = self._build_symbols(X, y)
		train_grad = theano.function(inputs = [index],
						outputs = T.grad(cost, self.theta_))
		return train_grad
	def _build_validate_model(self, v_validation_X, v_validation_y, batch_size):
		index = T.lscalar()
		X = v_validation_X[index * batch_size: (index + 1) * batch_size]
		y = v_validation_y[index * batch_size: (index + 1) * batch_size]
		_, error = self._build_symbols(X, y)
		validate_model = theano.function(inputs = [index],
					outputs = error)
		return validate_model
	def _build_predict_model(self, v_X):
		X = T.matrix('X')
		p_y_given_x = T.nnet.softmax(T.dot(X, self.W_) + self.b_)
		classes = theano.shared(value = self.classes, borrow = True)
		y_pred = classes[T.argmax(p_y_given_x, axis = 1).reshape((-1,))]
		return theano.function(inputs = [],
					outputs = [y_pred, p_y_given_x],
					givens = {
						X: v_X
					})
	def _build_symbols(self, X, y):
		p_y_given_x = T.nnet.softmax(T.dot(X, self.W_) + self.b_)
		classes = theano.shared(value = self.classes, borrow = True)
		y_pred = classes[T.argmax(p_y_given_x, axis = 1).reshape((-1,))]
		error = T.mean(T.neq(y_pred, y))
		## HERE NEEDS Y TO BE EXACTULY 0..N_CLASS-1
		negative_log_likelihood = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]),y])
		return negative_log_likelihood, error
	def _share_data(self, data, dtype=theano.config.floatX):
		shared_data = theano.shared(value = np.asarray(data,
										dtype = theano.config.floatX),
						borrow = True)
		return T.cast(shared_data, dtype = dtype)