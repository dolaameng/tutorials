import numpy as np 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import RandomizedPCA
from sklearn.utils import check_random_state

def deprecated_soft_threshold(data, centers, threshold = 0.0):
	## normalize both data and centers PER SAMPLE
	C = normalize(centers)
	X = normalize(data)
	## calculate cosine distances 
	similarities = np.dot(X, C.T)
	nsamples, ncenters = similarities.shape
	feats = np.empty((nsamples, 2*ncenters))

	## thresholding - pos and neg
	feats[:, :ncenters] = similarities
	feats[similarities<threshold, :ncenters] = 0.0
	feats[:, ncenters:] = -similarities
	feats[-similarities<threshold, ncenters:] = 0.0
	return feats

def soft_threshold(data, centers, threshold = 0.0, 
				normalized = False):
	## normalize both data and centers PER SAMPLE
	C = normalize(centers)
	X = normalize(data)
	## calculate cosine distances 
	similarities = np.dot(X, C.T)

	## thresholding - pos only
	similarities[similarities < threshold] = 0.0
	if normalized: 
		return normalize(similarities, copy = True)
	else:
		return similarities

class SamplingSoftThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, n_components, threshold = 0.0, 
                 random_state = None):
        self.n_components = n_components
        self.threshold = threshold
        self.random_state = random_state
    def fit(self, X, y = None):
        """
        Randomly sample support vectors
        """
        random_state = check_random_state(self.random_state)
        X = np.asarray(X)
        sv_indices = np.arange(X.shape[0])
        random_state.shuffle(sv_indices)
        self.sv_indices_ = sv_indices[:self.n_components]
        self.components_ = X[self.sv_indices_]
        return self
    def transform(self, X):
        return soft_threshold(X, self.components_, 
                              threshold = self.threshold)
    
class KMeansSoftThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, n_components, threshold = 0.0, 
                 *args, **kwargs):
        self.n_components = n_components
        self.threshold = threshold
        self.model_ = MiniBatchKMeans(n_clusters=self.n_components, 
                                      *args, **kwargs)
    def fit(self, X, y = None):
        self.model_.fit(X)
        return self
    def transform(self, X):
        return soft_threshold(X, self.model_.cluster_centers_, 
                              threshold = self.threshold)