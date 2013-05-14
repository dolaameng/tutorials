## Implementation feature engineering methods
## Basically, the methods implemented here extract patchs (subset of rows, subset of cols)
## from a data matrix, and some other feature transformation methods (e.g., tri-kmeans)
## The extracted new feature patches can be persisted in the ensemble for ensemble building

## TODO - ONLINE and OFFLINE two versions

import numpy as np

########### data partitioning ##################

def patch(data, rows, cols = None):
	"""
	data = data matrix, 1D or 2D array (matrix) 
	rows = iterator of rows (list) to select, None means selecting all rows
	cols = iterator of cols (list) to select, None means selecting all cols 
	return np.array (of the patch shape), but the DIM of return should be 
	the same as data (1D or 2D)
	"""
	data = np.asarray(data)
	dim = get_dim(data)
	if dim == 1:
		## ignore cols
		return data[rows] if rows else data
	elif dim == 2:
		nrows, ncols = data.shape
		rows = rows or xrange(nrows)
		cols = cols or xrange(ncols)
		return data[np.ix_(rows, cols)]
	else:
		raise RuntimeError('only supports 1D or 2D array') 

########## sequence generator #################
def strided_seqs(seq, stride, subsz):
	"""
	rng = the sequence to be selected from
	stride = stride (diff) between different sub_seqs
	subsz = the window size of all sub_seqs
	return iterable of subseqs 
	"""
	pass


##################### helper function ###############
def get_dim(data):
	"""
	return the dimension of the data - 1D or 2D np.array
	"""
	return len(np.asarray(data).shape)