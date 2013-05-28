from formula import *
from optimize import *

class DenoisingAutoEncoder(UnsupervisedModel):
	def __init__(self, n_hidden, n_epochs, corruption_level = 0.1, 
				validation_size = 0.2, batch_size = 50, learning_rate = 0.01):
		self.n_hidden = n_hidden
		self.corruption_level = corruption_level
		self.n_epochs = n_epochs
		self.validation_size = validation_size
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.formula_ = None
	def _create_formula(self, X):
		n_feats = X.shape[1]
		formula = FDAE(n_visible = n_feats, n_hidden = self.n_hidden, 
			corruption_level = self.corruption_level)
		return formula

class ContractiveAutoEncoder(UnsupervisedModel):
	def __init__(self, n_hidden, n_epochs, contraction_level = 0.1, 
				validation_size = 0.2, batch_size = 50, learning_rate = 0.01):
		self.n_hidden = n_hidden
		self.n_epochs = n_epochs
		self.contraction_level = contraction_level
		self.validation_size = validation_size
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.formula_ = None
	def _create_formula(self, X):
		n_feats = X.shape[1]
		formula = FCAE(n_visible = n_feats, n_hidden = self.n_hidden, 
						batch_size = self.batch_size, 
						contraction_level = self.contraction_level)
		return formula