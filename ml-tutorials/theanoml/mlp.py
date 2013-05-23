from formula import *
from sklearn.cross_validation import train_test_split
import numpy as np

class MLPClassifier(object):
	"""
	Multiple-layer perceptron classifier with one hidden layer 
	"""
	def __init__(self, n_classes, n_hidden, l1_penalty = 0.00, l2_penalty = 0.0001):
		"""
		n_classes = number of classes of output
			the output will be encoded as range(n_classes)
		n_hidden = number of hidden layers 
		l1_penalty = the coefficient of l1 penalty 
		l2_penalty = the coefficient of l2_square penalty
		"""
		self.n_classes = n_classes
		self.n_hidden = n_hidden
		self.l1_penalty = l1_penalty
		self.l2_penalty = l2_penalty
		self.mlp_ = None
	def fit(self, X, y):
		## intialize parameters always in fit()
		self.n_samples, self.n_feats = X.shape
		self.mlp_ = MLPClassifierFormula(n_in = self.n_feats, 
			n_hidden = self.n_hidden, n_out = self.n_classes, 
			L1_coeff = self.l1_penalty, L2_coeff = self.l2_penalty)
		self._optimize(X, y)
	def partial_fit(self, X, y):
		if self.mlp_ is None: # new fit
			self.fit(X, y)
		else:
			## no re-initialization of the classifier param
			self._optimize(X, y)
	def predict(self, X):
		y_pred, p_y_given_x = self.mlp_.prediction(X)
		return y_pred.eval()
	def predict_proba(self, X):
		y_pred, p_y_given_x = self.mlp_.prediction(X)
		return p_y_given_x.eval()
	def score(self, X, y):
		yhat = self.predict(X)
		return np.mean(yhat == y)
	def _optimize(self, X, y):
		train_X, validation_X, train_y, validation_y = train_test_split(X, y, test_size = 0.2)
		v_train_X, v_validation_X = share_data(train_X), share_data(validation_X)
		v_train_y, v_validation_y = share_data(train_y, dtype='int32'), share_data(validation_y, dtype='int32')
		sgd(v_train_X, v_train_y, v_validation_X, v_validation_y, self.mlp_)
	def _predict(self, X):
		v_X = share_data(X)
		return self.mlp_.prediction(v_X)