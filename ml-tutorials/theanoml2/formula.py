## Collection of Theano Formula Classes
## Each formula should have in common:
## (1) model params
## (2) model inputs - X, y or both
## (3) cost to minimize (with an optimizer)
## (4) error to get the accuracy (for validation)
## (5) prediction to get the output 
## the calculation in (3) - (5) is done by setting model input params
## e.g. model.X, or model.y 
## The methods involved in formla classes should be minimized, whereas
## The access to calculations is encouraged to use variables directly.
## This simply the usage of these formula classes - 
## ** The final purpose of formula classes is to create normal functions
## that can be used with optimizers or other functions, the fields in 
## formula classes play the roles of inputs and outputs in those functions 

import theano
import theano.tensor as T 
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from sklearn.base import BaseEstimator
from sklearn.cross_validation import train_test_split
from sklearn import metrics

from functools import partial
import abc
import numpy as np

from optimize import batch_sgd_optimize



class SupervisedModel(BaseEstimator):
	__metaclass__ = abc.ABCMeta
	def __init__(self):
		pass
	def fit(self, X, y):
		self.formula_ = self._create_formula(X, y)
		return self._optimize(X, y)
	def partial_fit(self, X, y):
		if self.formula_ is None:
			return self.fit(X, y)
		else:
			return self._optimize(X, y)
	def predict(self, X):
		v_X = share_data(X)
		predict_model = build_predict_model(self.formula_, {self.formula_.X: v_X})
		r = predict_model()
		if isinstance(r, list):
			yhat, yproba = r
		else:
			yhat = r
		return yhat
	def predict_proba(self, X):
		v_X = share_data(X)
		predict_model = build_predict_model(self.formula_, {self.formula_.X: v_X})
		_, yproba = predict_model()
		return yproba
	def score(self, X, y):
		if self.model_type == 'classification':
			yhat = self.predict(X)
			return np.mean(yhat == y)
		elif self.model_type == 'regression':
			yhat = self.predict(X)
			return metrics.explained_variance_score(y, yhat)
		else:
			raise RuntimeError('unknown model type')
	@abc.abstractmethod
	def _create_formula(self, X, y):
		return None
	def _optimize(self, X, y):
		train_X, validation_X, train_y, validation_y = train_test_split(X, y, 
			test_size = self.validation_size)
		v_train_X, v_validation_X = map(share_data, 
			[train_X, validation_X])
		if self.model_type == 'classification':
			v_train_y, v_validation_y = map(partial(share_data, dtype = 'int32'), 
				[train_y, validation_y])
		elif self.model_type == 'regression':
			v_train_y, v_validation_y = map(share_data, 
				[train_y, validation_y])
		else:
			raise RuntimeError('unknown model type')
		
		model_infor = build_batch_sgd_model_infor(self.formula_, 
        	v_train_X, v_train_y, v_validation_X, v_validation_y, 
        	batch_size = self.batch_size)
		best_params = batch_sgd_optimize(model_infor, n_epochs = self.n_epochs)
		self.formula_.params = best_params	
		return self

def build_batch_sgd_model_infor(model, v_train_X, v_train_y,
				v_validation_X, v_validation_y, batch_size, 
				learning_rate = 0.01):
	n_train_samples = v_train_X.get_value(borrow = True).shape[0]
	n_validation_samples = v_validation_X.get_value(borrow = True).shape[0]
	n_train_batches = n_train_samples / batch_size
	n_validation_batches = n_validation_samples / batch_size
	train_model = build_train_model(model, {
			model.X: v_train_X,
			model.y: v_train_y}, 
		batch_size, learning_rate)
	validate_model = build_validate_model(model, {
			model.X: v_validation_X,
			model.y: v_validation_y}, 
		batch_size)
	model_infor = {
		'params': model.params,
		'train_model': train_model,
		'validate_model': validate_model,
		'n_train_batches': n_train_batches,
		'n_validation_batches': n_validation_batches,
		'batch_size': batch_size
	}
	return model_infor

def build_train_model(formula, data_bindings, batch_size, learning_rate = 0.01):
	"""
	data_bindings = {formula.x: v_train_X, formula.y: v_train_y}
	"""
	index = T.lscalar('index')
	gparams = T.grad(formula.cost, formula.params)
	updates = [(param, param - learning_rate * gparam)
				for (param, gparam) in zip(formula.params, gparams)]
	givens = {var:data[index*batch_size:(index+1)*batch_size]
				for (var, data) in data_bindings.items()}
	train_model = theano.function(inputs = [index],
					outputs = formula.cost, 
					updates = updates,
					givens = givens)
	return train_model
def build_validate_model(formula, data_bindings, batch_size):
	index = T.lscalar('index')
	givens = {var:data[index*batch_size:(index+1)*batch_size]
				for (var, data) in data_bindings.items()}
	validate_model = theano.function(inputs = [index],
					outputs = formula.error, 
					givens = givens)
	return validate_model

def build_predict_model(formula, data_bindings):
	"""
	return a function with no input params, 
	it saves logic than putting X as an input 
	"""
	predict_model = theano.function(inputs = [],
					outputs = formula.prediction,
					givens = data_bindings)
	return predict_model


def share_data(data, dtype = theano.config.floatX):
	"""
	create shared variable from raw data, to make them 
	eligible in GPU computing 
	"""
	shared_data = theano.shared(np.asarray(data, 
								dtype = theano.config.floatX),
					borrow = True)
	return T.cast(shared_data, dtype = dtype)

class FLogisticRegression(object):
	"""
	Formula for Logistic Regression.
	params = self.W, self.b 
	input params = self.X, self.y
	self.cost = negative log likelihood
	self.error = classification error
	self.prediction = y_pred and p_y_given_x
	"""
	def __init__(self, n_in, n_out, X=None, y=None):
		## model params
		self.W = theano.shared(value = np.zeros((n_in, n_out), 
								dtype = theano.config.floatX),
					name = 'LR_W', borrow = True)
		self.b = theano.shared(value = np.zeros((n_out, ), 
								dtype = theano.config.floatX),
					name = 'LR_b', borrow = True)
		self.params = (self.W, self.b)
		## model inputs - integer y for classification
		self.X = X or T.matrix('X')
		self.y = y or T.ivector('y')
		## model output 
		self.p_y_given_x = T.nnet.softmax(T.dot(self.X, self.W) + self.b)
		self.y_pred = T.argmax(self.p_y_given_x, axis = 1)
		self.prediction = (self.y_pred, self.p_y_given_x)
		## model cost and error and score
		self.cost = -T.mean(T.log(self.p_y_given_x)[T.arange(self.y.shape[0]), self.y])
		self.error = T.mean(T.neq(self.y_pred, self.y))

class FLinearRegression(object):
	def __init__(self, n_in, X = None, y = None):
		## model params 
		self.W = theano.shared(value = np.zeros((n_in, ), 
								dtype = theano.config.floatX),
					name = 'LR_W', borrow = True)
		self.b = theano.shared(value = np.zeros((), 
								dtype = theano.config.floatX),
					name = 'LR_b', borrow = True)
		self.params = (self.W, self.b)
		## model inputs - integer y for classification
		self.X = X or T.matrix('X')
		self.y = y or T.vector('y')
		## model output 
		self.y_pred = T.dot(self.X, self.W) + self.b
		self.prediction = self.y_pred
		## model cost and error 
		self.cost = T.mean((self.y_pred - self.y)**2)
		self.error = T.mean(abs(self.y_pred - self.y))

class FHiddenLayer(object):
	"""Hidden Layer of MLP"""
	def __init__(self, n_in, n_out, activation = T.tanh, 
					X = None):
		## model params 
		rng = np.random.RandomState(0)
		W_value = np.asarray(rng.uniform(
							low = -np.sqrt(6. / (n_in + n_out)),
							high = np.sqrt(6. / (n_in + n_out)),
							size = (n_in, n_out)), 
						dtype = theano.config.floatX)
		if activation == T.nnet.sigmoid:
			W_value *= 4
		self.W = theano.shared(value = W_value, name = 'hidden_layer_W', borrow = True)
		b_value = np.zeros((n_out, ), dtype = theano.config.floatX)
		self.b = theano.shared(value = b_value, name = 'hidden_layer_b', borrow = True)
		self.params = (self.W, self.b)
		## model inputs - integer y for classification
		self.X = X or T.matrix('X')
		## model output 
		lin_output = T.dot(self.X, self.W) + self.b
		self.prediction = lin_output if activation is None else activation(lin_output)


class FMLPClassifier(object):
	"""MLP multi-class classifier
	"""
	def __init__(self, n_in, n_hidden, n_out, 
				l1_coeff = 0.0, l2_coeff = 0.0001, 
				X = None, y = None):
		rng = np.random.RandomState(0)
		## model inputs - integer y for classification
		self.X = X or T.matrix('X')
		self.y = y or T.ivector('y')
		## model params
		self.hidden_layer = FHiddenLayer(n_in = n_in, n_out = n_hidden, X = self.X)
		self.logregression_layer = FLogisticRegression(n_in = n_hidden, n_out = n_out, 
				X = self.hidden_layer.prediction, y = self.y)
		self.params = self.hidden_layer.params + self.logregression_layer.params
		## model prediction
		self.prediction = self.logregression_layer.prediction
		## model cost and error
		self.cost = self.logregression_layer.cost
		self.error = self.logregression_layer.error


class FLeNetConvPoolLayer(object):
	"""
	Pool layer of a convolutional network
	"""
	def __init__(self, image_shape, filter_shape, pool_size = (2, 2), 
					X = None, y = None):
		"""
		filter_shape = (n_filters, n_input_feats_maps, filter_ht, filter_wd)
		image_shape = (batch_size, n_input_feats_maps, img_ht, img_wd)
		pool_size = the downsampling (pooling) factor (n_rows, n_cols)
		"""
		assert image_shape[1] == filter_shape[1]
		rng = np.random.RandomState(0)
		## model inputs - integer y for classification
		self.X = X or T.matrix('X')
		## model params 
		## n_inputs_feats_maps * filter_ht * filter_wd inputs to each hidden unit
		fan_in = np.prod(filter_shape[1:])
		## each unit receives n_output_feats_maps * filter_ht * filter_wd / pool_size gradient
		fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) 
						/ np.prod(pool_size))
		W_bound = np.sqrt(6. / (fan_in + fan_out))
		self.W = theano.shared(np.asarray(rng.uniform(
                                        low = -W_bound,
                                        high = W_bound,
                                        size = filter_shape), 
                                    dtype=theano.config.floatX),
                                name = 'LeNet_W',
                                borrow = True)
		b_values = np.zeros((filter_shape[0], ), 
						dtype = theano.config.floatX)
		self.b = theano.shared(value = b_values, borrow = True, name = 'LeNet_b')
		self.params = (self.W, self.b)
		## model prediction
		conv_out = conv.conv2d(input = X, filters = self.W, 
                            filter_shape = filter_shape, image_shape = image_shape)
		pooled_out = downsample.max_pool_2d(input = conv_out, 
                            ds = pool_size, ignore_border = True)
		self.prediction = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
		## model cost and error
		## N.A.

class FLeNetClassifier(object):
	"""
	LeNet convolutional network
	"""
	def __init__(self, n_out, batch_size, image_size, 
		n_hidden = 500, filter_size = (5, 5), pool_size = (2, 2), n_kerns = (20, 50),
		X = None, y = None):
		"""
		n_kerns = list of n_hidden_nodes in each hidden layer (kernels)
		"""
		rng = np.random.RandomState(0)
		## model inputs - integer y for classification
		self.X = X or T.matrix('X')
		self.y = y or T.ivector('y')
		self.batch_size = batch_size
		## model params
		self.layer0_input = self.X.reshape((self.batch_size, 1, image_size[0], image_size[1]))
		self.layer0 = FLeNetConvPoolLayer(
				image_shape = (self.batch_size, 1, image_size[0], image_size[1]),
				filter_shape = (n_kerns[0], 1, filter_size[0], filter_size[1]),
				pool_size = pool_size, X = self.layer0_input)
		layer1_img_sz = [(image_size[i] - filter_size[i] + 1) / pool_size[i] 
							for i in xrange(2)]
		self.layer1 = FLeNetConvPoolLayer(
				image_shape = (self.batch_size, n_kerns[0], layer1_img_sz[0], layer1_img_sz[1]),
				filter_shape = (n_kerns[1], n_kerns[0], filter_size[0], filter_size[0]),
				pool_size = pool_size, X = self.layer0.prediction)
		layer2_img_sz = [(layer1_img_sz[i] - filter_size[i] + 1) / pool_size[i] 
							for i in xrange(2)]
		self.layer2 = FHiddenLayer(n_in = n_kerns[1]*layer2_img_sz[0]*layer2_img_sz[0], 
				n_out = n_hidden, X = self.layer1.prediction.flatten(2), 
				activation = T.tanh)
		self.layer3 = FLogisticRegression(n_in = n_hidden, n_out = n_out, 
				X = self.layer2.prediction, y = self.y)
		self.params = (self.layer3.params + self.layer2.params 
						+ self.layer1.params + self.layer0.params)
		## model prediction
		self.prediction = self.layer3.prediction
		## model cost and error
		self.cost = self.layer3.cost
		self.error = self.layer3.error





