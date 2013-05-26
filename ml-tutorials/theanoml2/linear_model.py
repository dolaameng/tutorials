from formula import *
from optimize import *



class LogisticRegression(SupervisedModel):
	def __init__(self, n_classes, n_epochs, validation_size = 0.2, 
			batch_size = 50, learning_rate = 0.01):
		self.n_classes = n_classes
		self.n_epochs = n_epochs
		self.validation_size = validation_size
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		#super(LogisticRegression, self).__init__()
		self.formula_ = None
		self.model_type = 'classification'
	def _create_formula(self, X, y):
		n_feats = X.shape[1]
		formula = FLogisticRegression(n_in = n_feats, n_out = self.n_classes)
		return formula

class LinearRegression(SupervisedModel):
	def __init__(self, n_epochs,  validation_size = 0.2, 
			batch_size = 50, learning_rate = 0.01):
		self.n_epochs = n_epochs
		self.validation_size = validation_size
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.formula_ = None
		self.model_type = 'regression'
	def _create_formula(self, X, y):
		n_feats = X.shape[1]
		formula = FLinearRegression(n_in = n_feats)
		return formula