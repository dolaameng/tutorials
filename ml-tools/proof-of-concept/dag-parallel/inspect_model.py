import sys, os
from sklearn.externals import joblib

if __name__ == '__main__':
	fmodel, fresult = sys.argv[1:]
	mdl = joblib.load(os.path.join(fmodel, 'model.pkl'))
	feature_importances = mdl.feature_importances_
	with open(fresult, 'w') as f:
		f.write(repr(feature_importances))