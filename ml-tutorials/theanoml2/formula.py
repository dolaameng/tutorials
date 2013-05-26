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
import numpy as np 
from theano.tensor.shared_randomstreams import RandomStreams
from sklearn.base import BaseEstimator
from sklearn.cross_validation import train_test_split
from functools import partial
import abc
from optimize import batch_sgd_optimize
from sklearn import metrics


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





