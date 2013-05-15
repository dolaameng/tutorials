## Implementation of greedy ensemble package
## Ensemble is implemented as a folder with a programming API

import numpy as np 
import os
from os import path
import shutil
from sklearn.externals import joblib
import json
from IPython import parallel
from sklearn.base import BaseEstimator
from functools import partial
from scipy.stats import mode
import copy

################ greedy ensemble model class ################
class GreedyEnsemble(BaseEstimator):
	def __init__(self, ensemble_path, scorefn, votefn,
			random_seed = 0, client = None):
		"""
		scorefn = function used to score model (in greedy search)
			sig = scorefn(y, yhat) RETURNS score
		votefn = function used to combine different model outputs
			sig = votefn(yhats) RETURNS combined_yhat 
		client = client to IPython.parallel.Client, if None, create new one
		"""
		self.ensemble_path = ensemble_path
		self.scorefn = scorefn
		self.votefn = votefn
		self.random_seed = random_seed
		self.client = client or parallel.Client()
		self.ensemble_ = []
	def fit(self, model_names, data_type = 'validation_data', verbose = False):
		"""
		Fitting algorithm = greedy search based on the performance measured by scoring fn
		1. make predictions by model on data_type ('validation_data')
		2. greedy search to fill in ensemble (ensemble intialized as empty)
			2.1 evaluate ensemble predictions by votefn
			2.2 evaluate ensemble performance by scorefn
		"""
		## always re-initialize the ensemble
		self.ensemble_ = []
		## make predictions for all models
		target, model_predictions = self._predict_by_model(model_names, data_type)
		## greedy search
		self.ensemble_ = self._greedy_search(model_predictions, self.ensemble_, target, data_type, verbose)
		return self

	def partial_fit(self, model_names, data_type = 'validation_data', verbose = False):
		"""
		Similar as fit() method, except that the ensemble is initialized as what it already is.
		"""
		target, model_predictions = self._predict_by_model(model_names, data_type)
		new_ensemble = self._greedy_search(model_predictions, self.ensemble_, target, data_type, verbose)
		self.ensemble_ = new_ensemble
		#print 'CURRENT ENSEMBLE', self.ensemble_, 'new added models', new_models
		return self
	def predict(self, data_type):
		"""
		Make prediction on new data using ensemble_. 
		The voting method could be a simple average, majority-vote or others.
		"""
		target, ensemble_predictions = self._predict_by_model(self.ensemble_, data_type)
		combined_prediction = self.votefn(ensemble_predictions.values())
		return combined_prediction
	def score(self, data_type):
		target, ensemble_predictions = self._predict_by_model(self.ensemble_, data_type)
		combined_prediction = self.votefn(ensemble_predictions.values())
		return self.scorefn(target, combined_prediction)
	@staticmethod
	def vote_major_class(yhats):
		yarr = np.vstack(yhats)
		y_mode = mode(yarr)[0][0]
		return y_mode.astype(np.int)
	def _predict_by_model(self, model_names, data_type):
		"""
		Use predict() to predict if 'is_probabilistic' is None or False in model config,
		otherwise use predict_proba()
		==> (target, dict of {model_name : model_prediction})
		"""
		n_models = len(model_names)
		if n_models == 0:
			raise RuntimeError('There should be at least one model to predict')
		dv = self.client[:]
		is_probabilistic = dv.map(partial(read_model_meta, 
									self.ensemble_path, 
									keys=['is_probabilistic'], default=False), 
								model_names)
		"""
		is_probabilistic = map(partial(read_model_meta, 
									self.ensemble_path, 
									keys=['is_probabilistic'], default=False), 
								model_names)
		"""
		print 'THIS STEP DONE'
		is_probabilistic = [d['is_probabilistic'] for d in is_probabilistic]
		data_types = [data_type for _ in xrange(n_models)]
		params = zip(model_names, data_types, is_probabilistic)
		results = parallel_predict_model(self.ensemble_path, params, self.client, verbose=False)
		target = results[0][1][0]
		return (target, {mdl_name : yhat for (mdl_name, (y, yhat)) in results})
	def _greedy_search(self, candidate_predictions, init_ensemble, target, data_type, verbose):
		if init_ensemble:
			_, model_predictions = self._predict_by_model(init_ensemble, data_type)
			ensemble = copy.deepcopy(init_ensemble)#[m_name for m_name in init_ensemble]
			ensemble_score = self.scorefn(target, self.votefn(model_predictions.values()))
		else:
			model_predictions = {}
			ensemble = []
			ensemble_score = 0.0
		
		model_predictions.update(candidate_predictions)

		candidates = set(candidate_predictions.keys())
		#print 'ensemble', ensemble
		while candidates:
			scores = [(m, self.scorefn(target, 
									self.votefn(map(model_predictions.get, 
													ensemble+[m])))) 
								for m in candidates]
			#print scores
			next_model, next_score = max(scores,
										key = lambda (m, s): s)
			if next_score < ensemble_score:
				if verbose:
					print 'checking model', next_model, 'NO improvement from ', ensemble_score, 'to', next_score
				break
			else:
				if verbose:
					print 'checking model', next_model, 'improvement from ', ensemble_score, 'to', next_score
				ensemble_score = next_score
				ensemble.append(next_model)
				candidates.remove(next_model)
		return ensemble



################ ensembles, data, and models #########
def new_ensemble(ensemble_name, container_path):
	"""
	Create ensemlbe folder, including,
	(*) files: data.json, models.json (initialized as {})
	(*) subfolders: data, models
	The data.json and models.json contain meta-data of
	created data sets of models, namely they are single dicts
	of format {'name_of_data_or_model': its_meta_data}
	"""
	## check existence - make folder
	ensemble_path = path.abspath(path.join(container_path, ensemble_name))
	os.mkdir(ensemble_path)
	os.mkdir(_get_path(ensemble_path, 'data_folder'))
	os.mkdir(_get_path(ensemble_path, 'models_folder'))
	_new_json_file(_get_path(ensemble_path, 'data_json'))
	_new_json_file(_get_path(ensemble_path, 'models_json'))
	return ensemble_path

## remove_ensemble - rm -fR ensemble_folder

def batch_write_data(ensemble_path, data_infor):
	"""
	batch version of write data 
	data_infor = list of tuples (data_name, data, data_meta)
	main diff from write_data is that the json file will
	be written only once in the last 
	"""
	data_records = []
	for (data_name, data, data_meta) in data_infor:
		data_file = data_name + '.pkl'
		data_path = path.join(_get_path(ensemble_path, 'data_folder'), 
										data_file)
		store_files = _persist(data_path, data)
		data_meta = data_meta or {}
		data_record = {}
		data_record.update(data_meta)
		data_record.update({'stored_files': store_files
						, 'file': data_path})
		data_records.append((data_name, data_record))
	_write_json_record(_get_path(ensemble_path, 'data_json'),
						dict(data_records), overwrite = False)

def write_data(ensemble_path, data_name, data, data_meta = None):
	"""
	assume overwrite = True
	update data.json with {data_name, data_meta}
	persist data into data subfolder
	"""
	data_file = data_name + '.pkl'
	data_path = path.join(_get_path(ensemble_path, 'data_folder'), 
										data_file)
	store_files = _persist(data_path, data)
	data_meta = data_meta or {}
	data_record = {}
	data_record.update(data_meta)
	data_record.update({'stored_files': store_files
						, 'file': data_path})
	_write_json_record(_get_path(ensemble_path, 'data_json'), 
						{data_name : data_record}, overwrite = False)

def remove_data(ensemble_path, data_name):
	"""
	1. remove the persisted file
	2. remove the record in data.json
	"""
	data_json_path = _get_path(ensemble_path, 'data_json')
	data_record = _read_json_record(data_json_path, [data_name])[data_name]
	stored_files = data_record['stored_files']
	_remove(stored_files)
	_remove_json_record(data_json_path, [data_name])

def load_data(ensemble_path, data_name):
	"""
	return data record and data tuple
	"""
	data_json_path = _get_path(ensemble_path, 'data_json')
	data_record = _read_json_record(data_json_path, [data_name])[data_name]
	data = _load(data_record['file'])
	return (data_record, data)

def batch_write_model(ensemble_path, model_infor):
	"""
	batch version of write models 
	model_infor = list of tuples (model_name, model, model_meta)
	main diff from write_model is that the json file will
	be written only once in the last
	"""
	model_records = []
	for (model_name, model, model_meta) in model_infor:
		model_file = model_name + '.pkl'
		model_path = path.join(_get_path(ensemble_path, 'models_folder'),
										model_file)
		## IT seems to be OK to write directly without removing
		## old FILES first - there could be files NOT used by any
		## but the reading should be OK
		store_files = _persist(model_path, model)
		model_meta = model_meta or {}
		model_record = {}
		model_record.update(model_meta)
		## file and stored_files always overwrite model_meta
		model_record.update({
			'stored_files': store_files,
			'file': model_path
			})
		model_records.append((model_name, model_record))
	_write_json_record(_get_path(ensemble_path, 'models_json'), 
						dict(model_records), overwrite = False)

def write_model(ensemble_path, model_name, model, model_meta = None):
	"""
	model_meta = dict of model information, e.g., 'description', 'is_probabilistic'
	"""
	model_file = model_name + '.pkl'
	model_path = path.join(_get_path(ensemble_path, 'models_folder'),
										model_file)
	## IT seems to be OK to write directly without removing
	## old FILES first - there could be files NOT used by any
	## but the reading should be OK
	store_files = _persist(model_path, model)
	model_meta = model_meta or {}
	model_record = {}
	model_record.update(model_meta)
	## file and stored_files always overwrite model_meta
	model_record.update({
		'stored_files': store_files,
		'file': model_path
		})
	_write_json_record(_get_path(ensemble_path, 'models_json'), 
						{model_name : model_record}, overwrite = False)

def update_model_record(ensemble_path, model_name, updates):
	models_json_path = _get_path(ensemble_path, 'models_json')
	model_record = _read_json_record(models_json_path, [model_name])[model_name]
	model_record.update(updates)
	_write_json_record(models_json_path, {model_name:model_record}, overwrite=False)

def remove_model(ensemble_path, model_name):
	models_json_path = _get_path(ensemble_path, 'models_json')
	model_record = _read_json_record(models_json_path, [model_name])[model_name]
	stored_files = model_record['stored_files']
	_remove(stored_files)
	_remove_json_record(models_json_path, [model_name])
	print 'remove ', model_name, 'from ', models_json_path

def read_model_meta(ensemble_path, model_name, keys = None, default = None):
	"""
	return the model meta information (dict under key=model_name in models.json) with key
	in keys. If a key is not in meta, it returns default value None
	if param keys = None, return the whole model meta dict 
	"""
	models_json_path = _get_path(ensemble_path, 'models_json')
	model_record = _read_json_record(models_json_path, [model_name])[model_name]
	if keys is None:
		model_meta = model_record
	else:
		model_meta = {k : model_record.get(k, default) for k in keys}
	return model_meta

def load_model(ensemble_path, model_name):
	models_json_path = _get_path(ensemble_path, 'models_json')
	model_record = _read_json_record(models_json_path, [model_name])[model_name]
	model = _load(model_record['file'])
	return (model_record, model) 

def train_model(ensemble_path, model_name, data_type, write_json=True):
	"""
	data_type: {'train_data', 'validation_data', 'test_data'}
	write_json: if False, return the params to write_model but dont 
	write to json file immediately, it is used for parallel mode where
	models.js needs to be updated in a single thread.
	"""
	## load model
	model_record, model = load_model(ensemble_path, model_name)
	if data_type not in model_record:
		raise RuntimeError('data type ' + data_type + ' NOT in model config')
	## load data
	_, (X, y) = load_data(ensemble_path, model_record[data_type])
	model.fit(X, y)
	## update model and its configuration
	if write_json:
		write_model(ensemble_path, model_name, model, model_record)
	return (ensemble_path, model_name, model, model_record)

def parallel_train_models(ensemble_path, model_data_pairs, client, verbose = True):
	"""
	model_data_pairs: [(model_name, data_type), ...]
	"""
	tasks = [('ensemble.train_model', {
						'ensemble_path': ensemble_path
						, 'model_name': model_name
						, 'data_type': data_type
						, 'write_json' : False}) 
				for (model_name, data_type) 
				in model_data_pairs]
	results = _parallel(tasks, client, verbose)
	## update models.json
	model_infor = [r[1:] for r in results] # exclude ensemble_path
	batch_write_model(ensemble_path, model_infor)
	"""
	for model_record in results:
		write_model(*model_record)
	"""


def predict_model(ensemble_path, model_name, data_type, probabilistic):
	"""
	data_type: {'train_data', 'validation_data', 'test_data'}
	probabilistic: probabilistic prediction or not
	RETURN (model_name, (target[if any], prediction))
	"""
	## load model
	model_record, model = load_model(ensemble_path, model_name)
	if data_type not in model_record:
		raise RuntimeError('data type ' + data_type + 'NOT in model config')
	## load data
	_, (X, y) = load_data(ensemble_path, model_record[data_type])
	predict = model.predict_proba if probabilistic else model.predict
	return (model_name, (y, predict(X)))

def parallel_predict_model(ensemble_path, model_data_prob, client, verbose=True):
	"""
	model_data_prob: list of (model_name, data_type, probabilistic)
	"""
	tasks = [('ensemble.predict_model', {
					'ensemble_path': ensemble_path
					, 'model_name' : model_name
					, 'data_type': data_type
					, 'probabilistic': probabilistic})
				for (model_name, data_type, probabilistic)
				in model_data_prob]
	return _parallel(tasks, client, verbose)


################ helper functions ####################
## parallel computing
def _parallel(tasks, client, verbose = True):
	"""
	basic parallel execution - no explicit data scheduling and sharing
	tasks: {fname_or_fn : kwparams}
	"""
	dv = client[:]
	dv.block = True
	## initialize parallel environment
	current_folder = path.dirname(path.realpath(__file__))
	dv.execute('cd %s' % current_folder)
	dv.execute('import ensemble')
	dv.execute('reload(ensemble)')
	"""
	## method 1: dispatch tasks
	fn_tasks = [(fn_or_fname 
					if not isinstance(fn_or_fname, str) 
					else parallel.Reference(fn_or_fname), kwparams)
						for (fn_or_fname, kwparams) in tasks]
	return dv.map(lambda (fn, kwparams): fn(**kwparams), fn_tasks)
	"""
	## method 2: dispatch tasks
	lb = client.load_balanced_view()
	chunksize = len(tasks) / len(client)
	asyn_results = []
	for (fn_or_fname, kwparams) in tasks:
		fn = fn_or_fname if not isinstance(fn_or_fname, str) \
					else parallel.Reference(fn_or_fname)
		asyn_results.append(lb.apply(fn, **kwparams))
	if verbose:	
		pre_n_finished = 0
		while not all([ar.ready() for ar in asyn_results]):
			n_finished = sum([ar.ready() for ar in asyn_results])
			n_total = len(asyn_results)
			if n_finished != pre_n_finished:
				print 'Parallel Progress: finished %d out of %d tasks' % (n_finished, n_total) 
				pre_n_finished = n_finished
		print 'Parallel Progress: DONE'
	return [ar.get() for ar in asyn_results]

## io from/to disk ###
def _persist(stuff_path, stuff):
	"""
	stuff_path: path to a pickle file 
	stuff: data/model to be pickled
	return path of all stored files
	use joblib to pickle
	"""
	store_files = joblib.dump(stuff, stuff_path)
	return map(path.abspath, store_files)

def _remove(files):
	for f in files:
		os.remove(f)

def _load(file_path):
	return joblib.load(file_path)

## path ##
def _get_path(ensemble_path, type):
	"""
	type: {'models_folder', 'models_json', 'data_folder', 'data_json'}
	"""
	locations = {'models_folder': 'models'
				, 'data_folder': 'data'
				, 'models_json': 'models.json'
				, 'data_json': 'data.json'}
	return path.abspath(path.join(ensemble_path, locations[type]))
## json files ##
def _new_json_file(json_file):
	json.dump({}, open(json_file, 'wb'))

def _write_json_record(json_file, records, overwrite):
	"""
	overwrite: False will update existing records with new records
	True will write records as json 
	records: dict of {model_or_data_name: model_or_data_meta}
	"""
	data = _read_json_record(json_file, keys = None)
	if overwrite:
		data = records
	else:
		data.update(records)
	json.dump(data, open(json_file, 'wb'))

def _read_json_record(json_file, keys = None):
	"""
	json_file contains json of a single dict
	keys = None return the whole record
	"""
	data = json.load(open(json_file, 'rb'))
	if keys is None:
		return data
	else:
		return dict({k:data[k] for k in keys})
def _remove_json_record(json_file, keys = None):
	"""
	assume json file to be a single dict, remove 
	all records with key in keys
	"""
	data = _read_json_record(json_file, keys = None)
	for k in keys:
		del data[k]
	_write_json_record(json_file, data, overwrite = True)

