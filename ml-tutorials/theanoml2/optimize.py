## Collection of optimization methods, used to optimize theano models
## the common interface is usually: 
## A dictionary object consisting of 
## (1) train_model(minibatch_index) method 
## (2) validate_model(minibatch_index) method 
## (3) a params object storing params
## (1), (2), and (3) are usually connected through the same formula object
## (4) n_train_batches, n_validation_batches, batch_size
## the data information is embedded in train_model() and validate_model()
## methods 

import theano
import theano.tensor as T
import numpy as np 


def batch_sgd_optimize(model_infor, n_epochs, verbose = True, patience = 1000, 
		patience_increase = 2, improvement_threshold = 0.995):
	## get model information
	params = model_infor['params']
	train_model = model_infor['train_model']
	validate_model = model_infor['validate_model']
	n_train_batches = model_infor['n_train_batches']
	n_validation_batches = model_infor['n_validation_batches']
	batch_size = model_infor['batch_size']
	## params 
	validation_frequency = min(n_train_batches, patience / 2)
	best_params = None
	best_validation_error = np.inf
	## iterative optimize with early stopping
	epoch = 0
	out_of_patience = False 
	while (epoch < n_epochs) and (not out_of_patience):
		epoch += 1
		for minibatch_index in xrange(n_train_batches):
			train_cost = train_model(minibatch_index)
			iter = (epoch - 1) * n_train_batches + minibatch_index
			if (iter + 1) % validation_frequency == 0:
				this_validation_error = np.mean([validate_model(i)
									for i in xrange(n_validation_batches)])
				if verbose:
					print 'epoch %i, minibatch %i/%i, validation_error %f %%' % (
						epoch, minibatch_index+1, 
						n_train_batches, this_validation_error * 100.
					)
				if this_validation_error < best_validation_error:
					if this_validation_error < best_validation_error * improvement_threshold:
						patience = max(patience, iter * patience_increase)
					best_validation_error = this_validation_error
					best_params = [p.get_value(borrow=True) for p in params] ## should be OK to update this way
			if patience <= iter:
				out_of_patience = True 
				break 
	if verbose:
		print 'optimization complete with best validation error: %f %%' % (best_validation_error * 100.)
	return best_params 

def batch_fixed_iter_optimize(model_infor, n_epochs, verbose = True):
	## get information needed for optimization
	params = model_infor['params'] # reference to params used in train_model
	train_model = model_infor['train_model']
	n_train_batches = model_infor['n_train_batches']
	batch_size = model_infor['batch_size']
	## optimization params
	best_params = None 
	best_train_cost = np.inf
	## iterative optimization for a fixed number of iterations 
	for epoch in xrange(n_epochs):
		train_cost = np.mean([train_model(i) for i in xrange(n_train_batches)])
		if verbose:
			print 'training epoch %i, train cost %f' % (epoch, train_cost)
		if train_cost < best_train_cost:
			best_train_cost = train_cost
			best_params = [p.get_value(borrow=True) for p in params]
	if verbose:
		print 'optimization complete with best train cost: %f' % best_train_cost
	return best_params