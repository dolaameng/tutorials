## Implementation feature engineering methods
## Basically, the methods implemented here extract patchs (subset of rows, subset of cols)
## from a data matrix, and some other feature transformation methods (e.g., tri-kmeans)
## The extracted new feature patches can be persisted in the ensemble for ensemble building

## TODO - ONLINE and OFFLINE two versions


import numpy as np
from sklearn.cross_validation import Bootstrap
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
import random, time, os, shutil
from os import path
from sklearn.externals import joblib
from IPython import parallel
from functools import partial
from itertools import cycle
from scipy import sparse
from sklearn.kernel_approximation import Nystroem

########### feature transformation #############
def kernel_approximation(Xs, client,  *args, **kwargs):
	"""
	Currently using Nystroem sampler to do the approximation 
	Xs = list of data to transform for kernel approximation
	*args, **kwargs = parameters to the Nystroem sampler 
	e.g., n_components, kernel {'rbf', 'polynomial', ...},
	gamma, random_state, degree and etc.
	"""
	dv = client[:]
	dv.block = True
	return dv.map(lambda (sampler, X): sampler.fit_transform(X), 
				zip([Nystroem(*args, **kwargs) for _ in range(len(Xs))], Xs))


class TriKmeansFeatures(BaseEstimator):
	def __init__(self, n_clusters, feat_patches, client,
					cache_dir = '/tmp', algo_name = 'KMeans', 
					sparse_result = True, random_state = 0):
		"""
		n_clusters = number of clusters used in KMeans or MiniBatchKmeans
		feat_patches = patches of feat indices to build clusters on, e.g., 
			[[feat_idx_i1, ..., feat_idx_j1], [feat_idx_i2, .., feat_idx_j2]]
			the feat_patches can be extracted by sequence generators such as 
			strided_seqs or bootstrap_seqs in the package.
		cache_dir = the cache dir for shared memory object - in parallel computing
		algo_name = the clustering algorithm used for now only {'KMeans', 'MiniBatchKmeans'}
		sparse_result = if the transformed result should be a sparse matrix coo_matrix or normal
		"""
		self.n_clusters = n_clusters
		self.feat_patches = feat_patches
		self.client = client or parallel.Client()
		self.cache_dir = cache_dir
		assert algo_name in ['KMeans', 'MiniBatchKMeans']
		self.algo_name = algo_name
		self.feat_to_kmeans_ = None
		self.sparse_result = sparse_result
		self.random_state = random_state
		random.seed(random_state)
	def fit(self, X, y = None):
		"""
		FITTING STEPS (unsupervised):
		1. extract patches from original features based on feat_patches 
		2. for each patch, train a clustering model, in parallel
		"""
		n_samples, n_features = X.shape
		## set shared memory object for parallel computing
		X_dir, X_path = self._persist_data(X, 'X')
		## doing clustering in parallel
		dv = self.client[:]
		## cannot use dv.execute to do from_import
		dv.block = False
		async_result = dv.map(TriKmeansFeatures._train_model, 
							zip(self.feat_patches, cycle([X_path]), 
								cycle([self.n_clusters]), 
								cycle([self.algo_name]), cycle([self.random_state])))
		async_result.wait_interactive()
		self.feat_to_kmeans_ = async_result.get()
		shutil.rmtree(X_dir)
		return self
	def transform(self, X):
		"""
		feature transformation in sequential 
		"""
		tri_feats = []
		for (feat_patch, kmeans) in self.feat_to_kmeans_:
			dist_to_clusters = kmeans.transform(X[:, feat_patch])
			meandist_per_cluster = np.mean(dist_to_clusters, axis = 0)
			tri_feat = np.apply_along_axis(lambda row: np.maximum(0, meandist_per_cluster-row),
											1, dist_to_clusters)
			tri_feats.append(tri_feat)

		tri_feats_X = sparse.coo_matrix(np.hstack(tri_feats)) if self.sparse_result else np.hstack(tri_feats)
		return tri_feats_X
	def transform_parallel(self, X):
		"""
		feature transformation in parallel
		"""
		dv = self.client[:]
		dv.block = True
		X_dir, X_path = self._persist_data(X, 'X')
		tri_feats = dv.map(TriKmeansFeatures._transform, zip(self.feat_to_kmeans_, cycle([X_path])))
		tri_feats_X = sparse.coo_matrix(np.hstack(tri_feats)) if self.sparse_result else np.hstack(tri_feats)
		shutil.rmtree(X_dir)
		return tri_feats_X
	def fit_transform(self, X, y = None):
		return self.fit(X, y).transform(X)

	@staticmethod 
	def _transform(args):
		"""
		DOEST NOT WORK
		"""
		raise RuntimeError('NOT IMPLEMENTED YET')
		(feat_patch, kmeans), X_path = args
		from sklearn.externals import joblib
		#from sklearn.cluster import KMeans
		#from sklearn.cluster import MiniBatchKMeans
		import numpy as np
		X = joblib.load(X_path)
		dist_to_clusters = kmeans.transform(X[:, feat_patch])
		meandist_per_cluster = np.mean(dist_to_clusters, axis = 0)
		tri_feat = np.apply_along_axis(lambda row: np.maximum(0, meandist_per_cluster-row),
											1, dist_to_clusters)
		return tri_feat
	@staticmethod	
	def _train_model(args):
		from sklearn.externals import joblib
		from sklearn.cluster import KMeans
		from sklearn.cluster import MiniBatchKMeans
		(feat_patch, X_path, n_clusters, algo_name, random_state) = args
		algorithm = (KMeans(n_clusters, random_state = random_state) 
							if algo_name == 'KMeans'
							else MiniBatchKmeans(n_clusters, random_state = random_state))
		X = joblib.load(X_path)
		return (feat_patch, algorithm.fit(X[:, feat_patch]))
	def _persist_data(self, X, X_name):
		tmstamp = time.ctime().replace(' ', '_')
		X_dir = path.abspath(path.join(self.cache_dir, tmstamp))
		os.mkdir(X_dir)
		X_path = path.join(X_dir, '%s.pkl' % X_name)
		joblib.dump(X, X_path)
		return (X_dir, X_path)


########### data partitioning ##################

def patch(data, rows, cols = None):
	"""
	data = data matrix, 1D or 2D array (matrix) 
	rows = iterator of rows (list) to select, None means selecting all rows
	cols = iterator of cols (list) to select, None means selecting all cols 
	return np.array (of the patch shape), but the DIM of return should be 
	the same as data (1D or 2D)
	if data is a sparse matrix, the return the matrix will be dense np.array
	"""
	if not sparse.issparse(data):
		data = np.asarray(data)
	dim = get_dim(data)
	if dim == 1:
		## ignore cols
		return data[rows] if rows is not None else data
	elif dim == 2:
		nrows, ncols = data.shape
		rows = rows if rows is not None else xrange(nrows)
		cols = cols if cols is not None else  xrange(ncols)
		if sparse.issparse(data):
			return data.toarray()[np.ix_(rows, cols)]
		else:
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
	try:
		return len(data.shape)
	except:
		return len(np.asarray(data).shape)

def _persist_data(self, X, X_name):
	tmstamp = time.ctime().replace(' ', '_')
	X_dir = path.abspath(path.join(self.cache_dir, tmstamp))
	os.mkdir(X_dir)
	X_path = path.join(X_dir, '%s.pkl' % X_name)
	joblib.dump(X, X_path)
	return (X_dir, X_path)