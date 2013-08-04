from sklearn import datasets
import cPickle
import sys, time

if __name__ == "__main__":
	iris = datasets.load_iris()
	X, y = iris.data, iris.target
	fname = sys.argv[1]
	cPickle.dump((X, y), open(fname, 'w'))

	time.sleep(5)