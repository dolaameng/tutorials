from formula import *
from optimize import *
from sklearn import metrics

class MLPClassifier(SupervisedModel):
	def __init__(self, n_classes, n_hidden, n_epochs, 
		l1_coeff = 0.00, l2_coeff = 0.001,
		validation_size = 0.2, batch_size = 50, learning_rate = 0.01):
		self.n_classes = n_classes
		self.n_hidden = n_hidden
		self.n_epochs = n_epochs
		self.l1_coeff = l1_coeff
		self.l2_coeff = l2_coeff
		self.validation_size = validation_size
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.formula_ = None
		self.model_type = 'classification'
	def _create_formula(self, X, y):
		n_feats = X.shape[1]
		formula = FMLPClassifier(n_in = n_feats, n_hidden = self.n_hidden, 
			n_out = self.n_classes, l1_coeff = self.l1_coeff, 
			l2_coeff = self.l2_coeff)
		return formula