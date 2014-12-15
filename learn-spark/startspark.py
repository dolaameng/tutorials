import os, sys
from os import path

SPARK_HOME = path.abspath("/home/dola/opt/spark-1.1.1/")
os.environ["SPARK_HOME"] = SPARK_HOME # for spark
sys.path.append(path.join(SPARK_HOME, "python/")) # for python

from pyspark import SparkConf, SparkContext

def create_spark_instance(master = "local", conf = None):
	"""
	master: default "local"
	conf: default 28 cores with 2g memory
	"""
	if not conf:
		conf = SparkConf()
		conf.set("spark.executor.memory", "2g")
		conf.set("spark.cores.max", "28")
		conf.setAppName("spark ipython notebook")

	spark_context = SparkContext(master, conf = conf)
	return spark_context
