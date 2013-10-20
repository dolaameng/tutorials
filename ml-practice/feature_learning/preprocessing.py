from util import *
from sklearn.base import BaseEstimator, TransformerMixin

def normalize_image01(image_features):
	"""
	normalize **Natural Images** pixel values to [0, 1], 
	based on the assumption that different images may have different 
	exposure level (mean) but the same variance (std).
	The normalization is done by (1) removing the mean PER Images 
	(2) translate pixels to [-1, +1] by dividing +/- 3stds
	(3) make it into [0.1, 0.9] by a linear transformation 

	param: image_features - a 2D ndarray nrows x nfeats, feats are 
	flattened image vector or a 3D ndarray nimags x nrows x ncols 

	This is quite a **hard** way of normalizing images compared to 
	PCA and ZCA whitening
	"""
	## (1) mean per image
	image_means = np.mean(image_features, axis = 1)
	image_feats = image_features - image_means[:, np.newaxis]
	## (2) global std of all image/patches pool
	image_std = np.std(image_feats)
	image_feats = np.maximum(np.minimum(image_feats, 3*image_std), -3*image_std)
	image_feats /= 3*image_std 
	## (3)
	image_feats = (image_feats + 1.) * 0.4 + 0.1
	return image_feats 
