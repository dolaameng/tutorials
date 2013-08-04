import subprocess

if __name__ == '__main__':
	FDATA = 'iris.pkl'
	FMODEL = 'tree'
	FRESULT = 'result.txt'
	print subprocess.call(['python', 'generate_data.py', FDATA])
	print subprocess.call(['python', 'build_model.py', FDATA, FMODEL])
	print subprocess.call(['python', 'inspect_model.py', FMODEL, FRESULT])
	print subprocess.call(['cat', FRESULT])