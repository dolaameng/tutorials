## collection of common formulas used for theano machine learning toolkit
## Usually the formulas are written in a way that separating those depending on X
## and those depending on y

## as well as a collection of optimization method (e.g. minibatch SGD with early stopping)

## The single purpose of those formulas is to bind with input and output data
## (usually represented as shared variable), and create functions that can be
## used to calculate relative quantities. 

import theano
import theano.tensor as T
import numpy as np

def share_data(data, dtype = theano.config.floatX):
	"""
	make raw_data fittable in GPU and return the acess to it in the right dtype form
	"""
	shared_data = theano.shared(np.asarray(data, 
								dtype = theano.config.floatX), 
								borrow = True)
	return T.cast(shared_data, dtype = dtype)

class LogisticRegressionFormula(object):
	"""
	Used for multi-class logistic regression (softmax)
	"""
	def __init__(self, n_in, n_out):
		self.n_in = n_in
		self.n_out = n_out
	## X related formula
	def bind_input(self, X):
		n_in, n_out = self.n_in, self.n_out
		self.W = theano.shared(value = np.zeros((n_in, n_out), 
												dtype = theano.config.floatX),
								name = 'logistic_regression_W', borrow = True)
		self.b = theano.shared(value = np.zeros((n_out, ), 
												dtype = theano.config.floatX),
								name = 'logistic_regression_b', borrow = True)
		self.p_y_given_x = T.nnet.softmax(T.dot(X, self.W) + self.b)
		self.y_pred = T.argmax(self.p_y_given_x, axis = 1)
		self.params = [self.W, self.b]
	## X, y related formula
	def negative_log_likelihood(self, y):
		"""
		y must be encoded as range(n_classes)
		"""
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
	def error(self, y):
		"""
		y must be encoded as range(n_classes)
		"""
		return T.mean(T.neq(self.y_pred, y))
	def cost(self, y):
		return self.negative_log_likelihood(y)
	def prediction(self, X):
		"""
		X = shared object to facilitate GPU calculation,
		return Tensor variable so that it can be evaluated later 
		by calling eval() on them
		"""
		p_y_given_x = T.nnet.softmax(T.dot(X, self.W) + self.b)
		y_pred = T.argmax(p_y_given_x, axis = 1)
		return (y_pred, p_y_given_x)

class HiddenLayerFormula(object):
	"""
	A hidden layer (linear with an activation function), e.g. used in MLP
	"""
	def __init__(self, rng, n_in, n_out, W = None, b = None, activation = T.tanh):
		self.rng, self.n_in, self.n_out = rng, n_in, n_out
		self.activation = activation
		## default W, b values 
		if W is None:
			W_value = np.asarray(self.rng.uniform(
							low = -np.sqrt(6. / (n_in + n_out)),
							high = np.sqrt(6. / (n_in + n_out)),
							size = (n_in, n_out)), 
						dtype = theano.config.floatX)
			if activation == T.nnet.sigmoid:
				W_value *= 4
			W = theano.shared(value = W_value, name = 'hidden_layer_W', borrow = True)
		if b is None:
			b_value = np.zeros((n_out, ), dtype = theano.config.floatX)
			b = theano.shared(value = b_value, name = 'hidden_layer_b', borrow = True)
		self.W, self.b = W, b
	def bind_input(self, X):
		"""
		rng = random generator seed
		X = input tensor variable 
		n_in, n_out = dimension of input and output to the hidden layer
		W, b = linear parameters 
		activation = onlinear activation of the hidden layer {T.tanh, T.nnet.sigmoid, None}
		"""
		## output
		lin_output = T.dot(X, self.W) + self.b
		self.output = lin_output if self.activation is None else self.activation(lin_output) 
		self.params = [self.W, self.b]
	def prediction(self, X):
		"""
		X = tensor variable, probabily a shared object to faciliatate GPU calculation
		return = output of the hidden layer, another tensor varialbe, call eval() to 
		get the real value
		"""
		lin_output = T.dot(X, self.W) + self.b
		return lin_output if self.activation is None else self.activation(lin_output)

class MLPClassifierFormula(object):
	"""
	MLP multi-class classifier (logisticregression) with one hidden layer
	"""
	def __init__(self, n_in, n_hidden, n_out, 
					L1_coeff = 0.00, L2_coeff = 0.0001, rng = None):
		rng = rng or np.random.RandomState(0)
		self.rng = rng
		self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
		self.L1_coeff = L1_coeff
		self.L2_coeff = L2_coeff 
		self.hiddenlayer = HiddenLayerFormula(rng = self.rng, n_in = n_in, n_out = n_hidden,
											activation = T.tanh)
		self.logregressionlayer = LogisticRegressionFormula(n_in = n_hidden, n_out = n_out)

	## X related formulas 
	def bind_input(self, X):
		## bind X variable
		self.hiddenlayer.bind_input(X)
		self.logregressionlayer.bind_input(self.hiddenlayer.output)
		## L1 norm 
		self.L1 = abs(self.hiddenlayer.W).sum() + abs(self.logregressionlayer.W).sum()
		## L2 norm
		self.L2_sqr = (self.hiddenlayer.W ** 2).sum() + (self.logregressionlayer.W ** 2).sum()
		self.params = self.hiddenlayer.params + self.logregressionlayer.params 
	## X and y related formulas 
	def negative_log_likelihood(self, y):
		return self.logregressionlayer.negative_log_likelihood(y)
	def error(self, y):
		return self.logregressionlayer.error(y)
	def cost(self, y):
		return self.negative_log_likelihood(y) + self.L1_coeff * self.L1 + self.L2_coeff * self.L2_sqr
	def prediction(self, X):
		"""
		X = tensor.variable, probabily a shared object to facilitate GPU claculation
		return Tensor variable as output, and call eval() them to get real values 
		"""
		hidden_output = self.hiddenlayer.prediction(X)
		y_pred, p_y_given_x = self.logregressionlayer.prediction(hidden_output)
		return (y_pred, p_y_given_x)

def sgd(v_train_X, v_train_y, v_validation_X, v_validation_y, classifier, 
					learning_rate = 0.01, n_epochs = 1000, batch_size = 20, verbose = True,
					patience = 10000, patience_increase = 2,
					improvement_threshold = 0.995):
	"""
	v_train_X, v_train_y = trainning data, tensor variables, usually shared variables
	v_validation_X, v_validation_y = validation data, tensor variables, usually shared variables
	classifier = formulas of the classifier, specially with cost(), and error() for training purpose 
	return = None, after optimization, the classifier.params will be set to the optimal value found in the process
	"""
	## number of batches and shape information 
	#n_feats = v_train_X.get_value(borrow = True).shape[1]
	n_train_samples = v_train_X.get_value(borrow = True).shape[0]
	n_valiation_samples = v_validation_X.get_value(borrow = True).shape[0]
	n_train_batches = n_train_samples / batch_size
	n_validation_batches = n_valiation_samples / batch_size
	## random generator state variable
	rng = np.random.RandomState(0)
	## general common variables used in the optimization, will be bound to 
	## specific values with givens param in the function 
	index = T.lscalar('index') ## minibatch index
	x = T.matrix('x', dtype = v_train_X.dtype) ## input
	y = T.vector('y', dtype = v_train_y.dtype) ## output - specific for classification 
	## bound classifier with variable x
	classifier.bind_input(x)
	## functions to optimize and update params 
	## functions parameterized by index variable now
	validate_model = theano.function(inputs = [index],
						outputs = classifier.error(y),
						givens = {
							x: v_validation_X[index*batch_size:(index+1)*batch_size],
							y: v_validation_y[index*batch_size:(index+1)*batch_size]
						})
	## gradient variables based on cost function of the training data 
	cost = classifier.cost(y)
	gparams = T.grad(cost, classifier.params)
	updates = [(param, param - learning_rate*gparam) 
				for (param, gparam) in zip(classifier.params, gparams)]
	train_model = theano.function(inputs = [index],
					outputs = cost, 
					updates = updates, 
					givens = {
						x: v_train_X[index*batch_size:(index+1)*batch_size],
						y: v_train_y[index*batch_size:(index+1)*batch_size]
					})
	## iterative training with early stop
	validation_frequency = min(n_train_batches, patience / 2)
	best_params = None
	best_validation_error = np.inf
	epoch = 0
	out_of_patience = False
	while (epoch < n_epochs) and (not out_of_patience):
		epoch += 1
		## each mini train batch
		for minibatch_index in xrange(n_train_batches):
			## train the model
			minibatch_cost = train_model(minibatch_index)
			## update the total iteration number 
			iter = (epoch - 1) * n_train_batches + minibatch_index
			## do validation when it is time
			if (iter + 1) % validation_frequency == 0:
				## get the current validation error rate
				this_validation_error = np.mean([validate_model(i)
											for i in xrange(n_validation_batches)])
				if verbose:
					print 'epoch %i, minibatch %i / %i, validation error %f %%' % (
						epoch, minibatch_index + 1, n_train_batches,
						this_validation_error * 100.
					)
				## increase the patience if a significant improvement is found
				if this_validation_error < best_validation_error:
					if this_validation_error < best_validation_error * improvement_threshold:
						patience = max(patience, iter * patience_increase)
					best_validation_error = this_validation_error
					best_params = classifier.params
			## if running out of patience, quit the iterative optimization 
			if patience <= iter:
				out_of_patience = True
				break 
	if verbose:
		print 'optimization complete with best valiation error %f %%' % (best_validation_error * 100.)
	## record best params into classifier 
	classifier.params = best_params
