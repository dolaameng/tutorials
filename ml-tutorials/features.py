## Implementation feature engineering methods
## Basically, the methods implemented here extract patchs (subset of rows, subset of cols)
## from a data matrix, and some other feature transformation methods (e.g., tri-kmeans)
## The extracted new feature patches can be persisted in the ensemble for ensemble building

## TODO - ONLINE and OFFLINE two versions

import numpy as np
from sklearn.cross_validation import Bootstrap

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
		return data[rows] if rows is not None else data
	elif dim == 2:
		nrows, ncols = data.shape
		rows = rows if rows is not None else xrange(nrows)
		cols = cols if cols is not None else  xrange(ncols)
		return data[np.ix_(rows, cols)]
	else:
		raise RuntimeError('only supports 1D or 2D array') 

########## sequence generator #################
def strided_seqs(seq, stride, subsize):
	"""
	seq = the sequence to be selected from
	stride = stride (diff) between different sub_seqs
	subsize = the window size of all sub_seqs
	return iterable of subseqs 
	"""
	extended_seq = seq + seq[:subsize]
	n_strides = len(seq) / stride
	sub_indices = [(i*stride, i*stride+subsize) for i in xrange(n_strides)]
	return [extended_seq[low:up] for (low, up) in sub_indices]

def bootstrap_seqs(seq, n_iter, subsize, random_state = 0):
	"""
	seq = the sequence to be selected from
	n_iter = number of sub sequences
	subsize = length of sub sequences
	return iterable of subseqs 
	"""
	bs = Bootstrap(len(seq), n_iter = n_iter, train_size = subsize, 
					random_state = random_state)
	sub_indices = [index for (index, _) in bs]
	seq_array = np.asarray(seq)
	return [seq_array[i] for i in sub_indices]



##################### helper function ###############
def get_dim(data):
	"""
	return the dimension of the data - 1D or 2D np.array
	"""
	return len(np.asarray(data).shape)