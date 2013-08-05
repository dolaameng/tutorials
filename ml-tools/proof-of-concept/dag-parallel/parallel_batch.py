import networkx as nx
import os
from IPython.parallel import Client

def parallel_dag(view, dag, jobs):
	"""
	view: a load_balanced view
	dag: networkx.DiGraph - nodes (job_ids)
	jobs: dictionary of {node -> call and parameters}
	"""
	results = {} ## jobs msg_ids 
	for node in nx.topological_sort(dag):
		#jobs[node]['f'](jobs[node]['params'])
		deps = [results[n] for n in dag.predecessors(node)]
		with view.temp_flags(after=deps, block = False):
			results[node] = view.apply_async(jobs[node]['f'], jobs[node]['params'])
	view.wait(results.values())
	"""
	for i, r in enumerate(results.values()):
		r.wait_interactive()
		print r.get()
	"""

def subprocess_call(args):
	import subprocess, os
	#print ['python'] + args
	retcode = subprocess.call(['python'] + args)
	return os.getcwd()
	#return retcode 

if __name__ == '__main__':
	FDATA = 'iris.pkl'
	FMODEL = 'tree'
	FRESULT = 'result.txt'

	dag = nx.DiGraph()
	[dag.add_node(i) for i in range(1, 4)]
	dag.add_edge(1, 2)
	dag.add_edge(2, 3)

	jobs = {
		  1: {'f': subprocess_call, 
			'params': ['generate_data.py', FDATA]}
		, 2: {'f': subprocess_call, 
			'params':['build_model.py', FDATA, FMODEL]}
		, 3: {'f': subprocess_call, 
			'params': ['inspect_model.py', FMODEL, FRESULT]}
	}

	client = Client()
	lbv = client.load_balanced_view()
	dv = client[:]
	dv.execute("import os")
	print os.path.abspath(os.getcwd())
	## IMPORTANT HERE - CHANGE DIR FOR subprocess to WORK
	dv.execute("os.chdir('%s')" % os.path.abspath(os.getcwd()))

	parallel_dag(lbv, dag, jobs)