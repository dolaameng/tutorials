from formula import *
from optimize import *
from sklearn import metrics

class LeNetClassifier(SupervisedModel):
	def __init__(self, n_classes, n_epochs, image_size, n_hidden = 500, 
		filter_size = (5, 5), pool_size = (2, 2), n_kerns = (20, 50),
		validation_size = 0.2, batch_size = 50, learning_rate = 0.01):
		self.n_classes = n_classes
		self.n_hidden = n_hidden
		self.n_epochs = n_epochs
		self.image_size = image_size
		self.filter_size = filter_size
		self.pool_size = pool_size
		self.n_kerns = n_kerns
		self.validation_size = validation_size
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.formula_ = None
		self.model_type = 'classification'
	def _create_formula(self, X, y):
		formula = FLeNetClassifier(n_out = self.n_classes, 
			batch_size = self.batch_size, 
			image_size = self.image_size, n_hidden = self.n_hidden, 
			filter_size = self.filter_size, pool_size = self.pool_size, 
			n_kerns = self.n_kerns)
		return formula
	def predict(self, X):
		assert X.shape[0] == self.batch_size, "Current implementation of Lenet only supports batch_size prediction"
		return super(LeNetClassifier, self).predict(X)
	def predict_proba(self, X):
		assert X.shape[0] == self.batch_size, "Current implementation of Lenet only supports batch_size prediction"
		return super(LeNetClassifier, self).predict_proba(X)
