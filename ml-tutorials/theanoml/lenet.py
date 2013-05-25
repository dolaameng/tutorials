from formula import *
from sklearn.cross_validation import train_test_split
import numpy as np

class LeNetClassifier(object):
	"""
	LeNet Convolutionary Classifier
	"""
	def __init__(self, image_size, n_classes, batch_size = None):
		self.image_size = image_size
		self.n_classes = n_classes
		self.batch_size = batch_size or 500
		self.lenet_ = None
	def partial_fit(self, X, y):
		if self.lenet_ is None:
			return self.fit(X, y)
		else:
			self._optimize(X, y)
			return self
	def fit(self, X, y):
		self.lenet_ = LeNetClassifierFormula(self.n_classes, 
					self.batch_size, self.image_size)
		self._optimize(X, y)
		return self
	def predict(self, X):
		y_pred, p_y_given_x = self._predict(X)
		return y_pred.eval()
	def predict_proba(self, X):
		y_pred, p_y_given_x = self._predict(X)
		return p_y_given_x.eval()
	def score(self, X, y):
		yhat = self.predict(X)
		return np.mean(yhat == y)
	def _optimize(self, X, y):
		train_X, validation_X, train_y, validation_y = train_test_split(X, y, test_size = 0.2)
		v_train_X, v_validation_X = share_data(train_X), share_data(validation_X)
		v_train_y, v_validation_y = share_data(train_y, dtype='int32'), share_data(validation_y, dtype='int32')
		sgd(v_train_X, v_train_y, v_validation_X, v_validation_y, self.lenet_, 
			learning_rate = 0.01, n_epochs = 200, batch_size = self.batch_size, 
			verbose = True, patience = 1000, patience_increase = 2,
			improvement_threshold = 0.995)
	def _predict(self, X):
		v_X = share_data(X)
		return self.lenet_.prediction(v_X)