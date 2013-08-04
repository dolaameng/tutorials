from sklearn import tree
import cPickle, sys
from sklearn.externals import joblib
import os, shutil

if __name__ == "__main__":
	fdata, fmodel = sys.argv[1:]
	mdl = tree.DecisionTreeClassifier(compute_importances=True)
	X, y = cPickle.load(open(fdata, 'r'))
	mdl.fit(X, y)
	print 'Model Score: ', mdl.score(X, y)
	print 'model features:', mdl.feature_importances_
	if os.path.exists(fmodel):
		shutil.rmtree(fmodel)
	os.mkdir(fmodel)
	joblib.dump(mdl, os.path.join(fmodel, 'model.pkl'))