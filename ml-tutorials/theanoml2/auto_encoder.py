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

class SdAClassifier(SupervisedModel):
	"""
	stacked denoising auto encoder classifier
	It is a mix of supervised and unsupervised model
	"""
	def __init__(self, n_classes, pretrain_epochs, finetune_epochs,
		hidden_layer_sizes = (100, 100, 100), corruption_levels = (0.1, 0.2, 0.3), 
		validation_size = 0.2, pretrain_batch_size = 1, finetune_batch_size = 50,
		pretrain_learning_rate = 0.001, finetune_learning_rate = 0.1):
		self.n_classes = n_classes
		self.pretrain_epochs = pretrain_epochs
		self.finetune_epochs = finetune_epochs
		self.hidden_layer_sizes = hidden_layer_sizes
		self.corruption_levels = corruption_levels
		self.validation_size = validation_size
		self.pretrain_batch_size = pretrain_batch_size
		self.finetune_batch_size = finetune_batch_size
		self.pretrain_learning_rate = pretrain_learning_rate
		self.finetune_learning_rate = finetune_learning_rate
		self.formula_ = None
		self.model_type = 'classification'
	def _create_formula(self, X, y):
		n_feats = X.shape[1]
		formula = FSDAClassifier(n_in = n_feats, n_out = self.n_classes, 
				hidden_layer_sizes = self.hidden_layer_sizes, 
				corruption_levels = self.corruption_levels)
		return formula
	def pretrain(self, X):
		"""
		pretraining of all dA layers are based on X input
		"""
		v_X = share_data(X)
		dA_layers = self.formula_.dA_layers
		for dA_layer in dA_layers:
			dA_layer_infor = build_batch_fixed_iter_model_infor(dA_layer, v_X, 
				batch_size = self.pretrain_batch_size, 
				learning_rate = self.pretrain_learning_rate, 
				model_X = self.formula_.X) # it is formula x instead of daLayer.X
			best_params = batch_fixed_iter_optimize(dA_layer_infor, 
					n_epochs = self.pretrain_epochs)
			for i in xrange(len(best_params)):
				dA_layer.params[i].set_value(best_params[i], borrow=True)
		return self
	def transform(self, X):
		"""
		get the hidden representation instead of the output from logistic regression layer
		"""
		v_X = share_data(X)
		predict_model = build_predict_model(self.formula_.dA_layers[-1], {self.formula_.X: v_X})
		return predict_model()
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
        	batch_size = self.finetune_batch_size, learning_rate = self.finetune_learning_rate)
		best_params = batch_sgd_optimize(model_infor, n_epochs = self.finetune_epochs)
		for i in xrange(len(best_params)):
			self.formula_.params[i].set_value(best_params[i], borrow=True)	
		return self