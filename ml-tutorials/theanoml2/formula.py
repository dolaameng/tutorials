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

def share_data(data, dtype = theano.config.floatX):
	"""
	create shared variable from raw data, to make them 
	eligible in GPU computing 
	"""
	shared_data = theano.shared(np.asarray(data, 
								dtype = theano.config.floatX),
					borrow = True)
	return T.cast(shared_data, dtype = dtype)

def FLogisticRegression(object):
	"""
	Formula for Logistic Regression.
	params = self.W, self.b 
	input params = self.X, self.y
	self.cost = negative log likelihood
	self.error = classification error
	self.prediction = y_pred and p_y_given_x
	"""
	def __init__(self, n_in, n_out, 
				X=None, y=None):
		## model params
		self.W = theano.shared(value = np.zeros((n_in, n_out), 
								dtype = theano.config.floatX),
					name = 'LR_W', borrow = True)
		self.b = theano.shared(value = np.zeros((n_out, ), 
								dtype = theano.config.floatX),
					name = 'LR_b', borrow = True)
		self.params = (self.W, self.b)
		## model inputs 
		self.X = X or T.matrix('X')
		self.y = y or T.ivector('y')
		## model output 
		self.p_y_given_x = T.nnet.softmax(T.dot(self.X, self.W) + self.b)
		self.y_pred = T.argmax(self.p_y_given_x, axis = 1)
		self.prediction = (self.y_pred, self.p_y_given_x)
		## model cost and error 
		self.cost = -T.mean(T.log(self.p_y_given_x)[T.arange(self.y.shape[0]), self.y])
		self.error = T.mean(T.neq(self.y_pred, self.y))