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
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import sys

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
		self.W = theano.shared(value = np.zeros((n_in, n_out), 
												dtype = theano.config.floatX),
								name = 'logistic_regression_W', borrow = True)
		self.b = theano.shared(value = np.zeros((n_out, ), 
												dtype = theano.config.floatX),
								name = 'logistic_regression_b', borrow = True)
		self.params = [self.W, self.b]
	## X related formula
	def bind_input(self, X):
		n_in, n_out = self.n_in, self.n_out
		
		self.p_y_given_x = T.nnet.softmax(T.dot(X, self.W) + self.b)
		self.y_pred = T.argmax(self.p_y_given_x, axis = 1)
		
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
		self.params = [self.W, self.b]
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
	def prediction(self, X):
		"""
		X = tensor variable, probabily a shared object to faciliatate GPU calculation
		return = output of the hidden layer, another tensor varialbe, call eval() to 
		get the real value
		"""
		lin_output = T.dot(X, self.W) + self.b
		return lin_output if self.activation is None else self.activation(lin_output)


class LeNetConvPoolLayerFormula(object):
	"""
	Pool Layer of a convolutional network
	"""
	def __init__(self, rng, filter_shape, image_shape, poolsize=(2, 2)):
		"""
		rng = np.random.RandomState
		filter_shape = (n_filters, n_input_feats_maps, filter_height, filter_width)
		image_shape = (batch_size, n_input_feats_maps, img_height, img_width)
		poolsize = the downsampling (pooling) factor (n_rows, n_cols)
		"""
		assert image_shape[1] == filter_shape[1]
		self.rng = rng
		self.filter_shape = filter_shape
		self.image_shape = image_shape
		self.poolsize = poolsize

		## n_input_feats_maps * filter_ht * filter_wd inputs to 
		## each hidden unit
		fan_in = np.prod(filter_shape[1:])
		## each unit in the lower layer receives a gradient from
		## n_output_feats_maps * filter_ht * filter_wd / poolsize
		fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) 
						/ np.prod(poolsize))
		## initalize weights
		W_bound = np.sqrt(6. / (fan_in + fan_out))
		self.W = theano.shared(np.asarray(rng.uniform(
                                        low = -W_bound,
                                        high = W_bound,
                                        size = filter_shape), 
                                        dtype=theano.config.floatX),
                                borrow = True)
		b_values = np.zeros((filter_shape[0], ), 
						dtype = theano.config.floatX)
		self.b = theano.shared(value = b_values, borrow = True)
		self.params = [self.W, self.b]
	def bind_input(self, X):
		"""
		X = input, theano.tensor.dtensor4
		"""
		filter_shape, image_shape = self.filter_shape, self.image_shape
		rng,  poolsize = self.rng, self.poolsize
		
		## convolve input feature maps with fitlers
		conv_out = conv.conv2d(input = X, filters = self.W, 
                            filter_shape = filter_shape, image_shape = image_shape)
		## downsample each feature map individually using maxpooling
		pooled_out = downsample.max_pool_2d(input = conv_out, 
                            ds = poolsize, ignore_border = True)
		## set the output and params
		self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
		
	def prediction(self, X):
		conv_out = conv.conv2d(input = X, filters = self.W, 
                            filter_shape = self.filter_shape, image_shape = self.image_shape)
		## downsample each feature map individually using maxpooling
		pooled_out = downsample.max_pool_2d(input = conv_out, 
                            ds = self.poolsize, ignore_border = True)
		return T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

class LeNetClassifierFormula(object):
	"""
	LeNet convolutional network
	"""
	def __init__(self, n_out, batch_size, image_size, filter_size = (5, 5), 
				pool_size = (2, 2), n_kerns = (20, 50), rng = None):
		"""
		image_size = (image_height, image_width)
		n_kerns = list of n_hidden_nodes in each hidden layer (kernels)
		"""
		self.n_out = n_out
		self.image_size = image_size
		self.filter_size = filter_size
		self.pool_size = pool_size
		self.n_kerns = n_kerns
		self.batch_size = batch_size
		rng = rng or np.random.RandomState(0)
		self.rng = rng

		self.layer0 = LeNetConvPoolLayerFormula(self.rng, 
				image_shape = (batch_size, 1, image_size[0], image_size[1]),
				filter_shape = (n_kerns[0], 1, filter_size[0], filter_size[1]),
				poolsize = pool_size)

		layer1_img_sz = [(image_size[i] - filter_size[i] + 1) / pool_size[i] for i in xrange(2)]
		self.layer1 = LeNetConvPoolLayerFormula(self.rng, 
				image_shape = (batch_size, n_kerns[0], layer1_img_sz[0], layer1_img_sz[1]),
				filter_shape = (n_kerns[1], n_kerns[0], filter_size[0], filter_size[0]),
				poolsize = pool_size)

		layer2_img_sz = [(layer1_img_sz[i] - filter_size[i] + 1) / pool_size[i] for i in xrange(2)]
		self.layer2 = HiddenLayerFormula(self.rng, n_in = n_kerns[1]*layer2_img_sz[0]*layer2_img_sz[0], 
								n_out = 500, activation = T.tanh)

		self.layer3 = LogisticRegressionFormula(n_in = 500, n_out = n_out)
		self.params = self.layer3.params + self.layer2.params + self.layer1.params + self.layer0.params
	def bind_input(self, X):
		n_out = self.n_out
		image_size = self.image_size
		filter_size = self.filter_size
		pool_size = self.pool_size
		n_kerns = self.n_kerns
		batch_size = self.batch_size
		rng = self.rng

		layer0_input = X.reshape((batch_size, 1, image_size[0], image_size[1]))
		self.layer0.bind_input(layer0_input)
		self.layer1.bind_input(self.layer0.output)
		self.layer2.bind_input(self.layer1.output.flatten(2))
		self.layer3.bind_input(self.layer2.output)
		
	def cost(self, y):
		return self.layer3.negative_log_likelihood(y)
	def error(self, y):
		return self.layer3.error(y)
	def prediction(self, X):
		print >> sys.stderr, 'The prediction() in LeNetClassifierFormula does NOT look right'
		n_batches = X.get_value(borrow = True).shape[0] / self.batch_size
		batch_ys = []
		batch_probas = []
		for i in xrange(n_batches):
			subX = X[i*self.batch_size:(i+1)*self.batch_size]
			self.bind_input(subX)
			"""
			layer0_input = subX.reshape((self.batch_size, 1, self.image_size[0], self.image_size[1]))
			layer1_input = self.layer0.prediction(layer0_input)
			layer2_input = self.layer1.prediction(layer1_input).flatten(2)
			layer3_input = self.layer2.prediction(layer2_input)
			y, proba = self.layer3.prediction(layer3_input)
			batch_ys.append(y)
			batch_probas.append(proba)
			"""
			batch_ys.append(self.layer3.y_pred)
			batch_probas.append(self.layer3.p_y_given_x)
		return T.concatenate(batch_ys, axis = 0), T.concatenate(batch_probas, axis = 0)
		"""
		layer0_input = X.reshape((self.batch_size, 1, self.image_size[0], self.image_size[1]))
		layer1_input = self.layer0.prediction(layer0_input)
		layer2_input = self.layer1.prediction(layer1_input).flatten(2)
		layer3_input = self.layer2.prediction(layer2_input)
		return self.layer3.prediction(layer3_input)
		"""

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
		self.params = self.hiddenlayer.params + self.logregressionlayer.params
	## X related formulas 
	def bind_input(self, X):
		## bind X variable
		self.hiddenlayer.bind_input(X)
		self.logregressionlayer.bind_input(self.hiddenlayer.output)
		## L1 norm 
		self.L1 = abs(self.hiddenlayer.W).sum() + abs(self.logregressionlayer.W).sum()
		## L2 norm
		self.L2_sqr = (self.hiddenlayer.W ** 2).sum() + (self.logregressionlayer.W ** 2).sum()
		 
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
