import numpy as np 
import os
from os import path
import shutil
from sklearn.externals import joblib
import json
from IPython import parallel


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

## remove_ensemble - rm -fR ensemble_folder

def write_data(ensemble_path, data_name, data, data_meta):
	"""
	assume overwrite = True
	update data.json with {data_name, data_meta}
	persist data into data subfolder
	"""
	data_file = data_name + '.pkl'
	data_path = path.join(_get_path(ensemble_path, 'data_folder'), 
										data_file)
	store_files = _persist(data_path, data)
	data_record = {}
	data_record.update(data_meta)
	data_record.update({'store_files': store_files
						, 'file': data_path})
	_write_json_record(_get_path(ensemble_path, 'data_json'), 
						data_record, overwrite = True)


################ helper functions ####################
def _persist(stuff_path, stuff, compress = 9):
	"""
	stuff_path: path to a pickle file 
	stuff: data/model to be pickled
	return path of all stored files
	use joblib to pickle
	"""
	store_files = joblib.dump(stuff, stuff_path, compress=compress)
	return map(path.abspath, store_files)
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
	overwrite: False will raise exception at any 
	existing key in keys
	records: dict of {model_or_data_name: model_or_data_meta}
	"""
	data = read_json_record(json_file, keys = None)
	if not overwrite and any([k in data for k in records.keys()]):
		raise RuntimeError('key already exists')
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
