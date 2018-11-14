import os
if __name__ == '__main__':
	print ('Data Processing...')
	os.system('python3 dataprocessing.py')
	print ('run model...')
	os.system('python3 run_model.py')
	print ('Finished!')