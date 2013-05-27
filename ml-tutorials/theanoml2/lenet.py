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
		"""
		challenge is: the predicion depends on the batch size
		"""
		raise RuntimeError('Not Implemented')
		batch_size = self.batch_size
		n_batches = X.shape[0] / batch_size
		v_Xs = [share_data(X[i*batch_size:(i+1)*batch_size]) for i in xrange(n_batches)]
		yhats = []
		for v_X in v_Xs:
			predict_model = build_predict_model(self.formula_, {self.formula_.X: v_X})
			yhat, yproba = predict_model()
			yhats.append(yhat)
		return np.vstack(yhats)
	def predict_proba(self, X):
		raise RuntimeError('Not Implemented')
		v_X = share_data(X)
		predict_model = build_predict_model(self.formula_, {self.formula_.X: v_X})
		_, yproba = predict_model()
		return yproba
