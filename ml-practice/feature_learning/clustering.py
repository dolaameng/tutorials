from util import *
from sklearn import cluster
from sklearn.feature_extraction import image
import math

class TriCluster(BaseEstimator):
	def __init__(self, n_features, *args, **kwargs):
		self.n_features = n_features
		self.model_ = cluster.MiniBatchKMeans(self.n_features, *args, **kwargs)
	def fit(self, data_X):
		self.model_.fit(data_X)
		self.mean_dists_ = self.model_.transform(data_X).mean(axis = 0)
		return self
	def get_feat_param(self):
		return self.model_.cluster_centers_
	def transform(self, data_X):
		dist_X = self.model_.transform(data_X)
		dist_X = np.maximum(0, self.mean_dists_ - dist_X)
		return dist_X

class RandomTriCluster(BaseEstimator):
	"""
	Unlike TriCluster, its random version finds local features
	by first randomly picking patches from original features 
	and then do clustering on them.
	It is not neccessary when TriCluster is used with Convolution
	operation and Pooling.
	"""
	def __init__(self, n_in, n_features, n_patches, 
					sampling_method = None,
					*args, **kwargs):
		"""
		n_in: number of features in original space
		n_features: total number of features 
		n_patches: how many random local patches to use in the original feature space
		so in each patch, there are n_features/n_patches clusters constructed
		sampling_method: {None, '1d', '2d'}: the way of finding random patchs 
			None: no continousness constraint 
			'1d': features sampled in the same patch are continous in 1D 
			'2d': features sampled in the same patch are continous in 2D (e.g. for image), 
				  assume image dimension is sqrt(nin) x sqrt(nin)

		TODO: to fix the conceptual bug that number_of_orignal_features_in_each_patch
		= number_clusters_in_each_patch
		"""
		self.n_in = n_in
		self.n_features = n_features
		self.n_patches = n_patches
		self.sampling_method = sampling_method
		if self.sampling_method is None: #pure randomness
			self.feat_indices_ = np.array_split(np.random.randint(low = 0, 
														high = n_in, 
														size = n_features),
												n_patches)
		elif self.sampling_method is '1d':
			nfeat_perpatch = n_features / n_patches
			nfeat_last = n_features - nfeat_perpatch * (n_patches-1)
			feat_starts = np.random.randint(low=0, high=n_in-nfeat_perpatch, size=n_patches)
			self.feat_indices_ = [np.arange(start, start+nfeat_perpatch) 
					for start in feat_starts[:-1]]
			self.feat_indices_.append(np.arange(feat_starts[-1], feat_starts[-1]+nfeat_last))
		elif self.sampling_method is '2d':
			nrows = int(math.sqrt(n_in))
			ncols = n_in / nrows
			grid = np.arange(nrows*ncols).reshape((nrows, ncols)) 
			nfeat_perpatch = n_features / n_patches
			nfeat_perrow = int(math.sqrt(nfeat_perpatch))
			nfeat_percol = nfeat_perpatch / nfeat_perrow
			nfeat_last = n_features - nfeat_perrow*nfeat_percol*n_patches

			self.feat_indices_ = (image.extract_patches(grid, (nfeat_perrow, nfeat_percol))
										.reshape(-1, nfeat_perrow*nfeat_percol))
			self.feat_indices_ = self.feat_indices_[np.random.choice(len(self.feat_indices_), n_patches)]
			if nfeat_last > 0:
				self.feat_indices_.append(np.random.randint(low = 0, 
														high = n_in, 
														size = nfeat_last))
		else:
			raise ValueError('param sampling_method=%s is not understandable' % self.sampling_method)
		self.models_ = [cluster.MiniBatchKMeans(len(findex), *args, **kwargs) 
								for findex in self.feat_indices_]
	def fit(self, data_X):
		self.mean_dists_ = []
		for findex, model in zip(self.feat_indices_, self.models_):
			sub_data = data_X[:, findex]
			model.fit(sub_data)
			self.mean_dists_.append(model.transform(sub_data).mean(axis = 0))
		return self
	def get_feat_param(self):
		W = np.zeros((self.n_features, self.n_in))
		ifeat = 0
		for findex, model in zip(self.feat_indices_, self.models_):
			for i, center in enumerate(model.cluster_centers_):
				W[ifeat, findex] = center
				ifeat += 1
		return W
	def transform(self, data_X):
		feats = []
		for findex, model, mean_dist in zip(self.feat_indices_, self.models_, self.mean_dists_):
			dist_X = model.transform(data_X[:, findex])
			dist_X = np.maximum(0, mean_dist - dist_X)
			feats.append(dist_X)
		return np.hstack(feats)