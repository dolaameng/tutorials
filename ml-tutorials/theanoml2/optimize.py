## Collection of optimization methods, used to optimize theano models
## the common interface is usually: 
## A dictionary object consisting of 
## (1) train_model(minibatch_index) method 
## (2) validate_model(minibatch_index) method 
## (3) a classifier object storing params
## (4) n_train_batches, n_validation_batches, batch_size
## the data information is embedded in train_model() and validate_model()
## methods 

def batch_sgd(model_infor, n_epochs, verbose = True,
				patience = 1000, patience_increase = 2,
				improvement_threshold = 0.995):
	train_model = model_infor['train_model']
	validate_model = model_infor['validate_model']
	n_train_batches = model_infor['n_train_batches']
	n_validation_batches = model_infor['n_validation_batches']