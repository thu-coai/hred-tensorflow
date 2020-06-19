import pytest
import random
import tensorflow as tf
from run import run
from main import main
import os
import json
import shutil
cwd = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(cwd, '..', 'cotk')

def setup_function(function):
	import sys
	sys.argv = ['python3']
	random.seed(0)
	import numpy as np
	np.random.seed(0)
	tf.set_random_seed(0)
	try:
		shutil.rmtree(cwd + '/output_test')
	except Exception:
		pass
	try:
		shutil.rmtree(cwd + '/tensorboard_test')
	except Exception:
		pass
	try:
		shutil.rmtree(cwd + '/model_test')
	except Exception:
		pass
	try:
		shutil.rmtree(cwd + '/cache_test')
	except Exception:
		pass
	os.mkdir(cwd + '/output_test')
	os.mkdir(cwd + '/tensorboard_test')
	os.mkdir(cwd + '/model_test')
	os.mkdir(cwd + '/cache_test')

def teardown_function(function):
	shutil.rmtree(cwd + '/output_test')
	shutil.rmtree(cwd + '/tensorboard_test')
	shutil.rmtree(cwd + '/model_test')
	shutil.rmtree(cwd + '/cache_test')

def modify_args(args):
	args.cuda = False
	args.restore = None
	args.wvclass = 'Glove'
	args.wvpath = path + '/tests/wordvector/dummy_glove/300d'
	args.embedding_size=300  #must be the same as the dim of wvpath
	args.out_dir = cwd + '/output_test'
	args.log_dir = cwd + '/tensorboard_test'
	args.model_dir = cwd + '/model_test'
	args.cache_dir = cwd + '/cache_test'

	args.name = 'test_hred_tensorflow'
	args.epochs = 1
	args.checkpoint_steps = 1
	args.datapath = path + '/tests/dataloader/dummy_ubuntucorpus#Ubuntu'

def test_train(mocker):
	def side_effect_train(args):
		modify_args(args)
		args.mode = 'train'
		main(args)
	def side_effect_restore(args):
		modify_args(args)
		args.mode = 'train'
		args.restore = 'last'
		main(args)
	def side_effect_cache(args):
		modify_args(args)
		args.mode = 'train'
		args.cache = True
		main(args)
	mock = mocker.patch('main.main', side_effect=side_effect_train)
	run()
	tf.reset_default_graph()
	mock.side_effect = side_effect_restore
	run()
	tf.reset_default_graph()
	mock.side_effect = side_effect_cache
	run()
	tf.reset_default_graph()

def test_test(mocker):
	def side_effect_test(args):
		modify_args(args)
		args.mode = 'test'
		main(args)
	mock = mocker.patch('main.main', side_effect=side_effect_test)
	run()
	old_res = json.load(open("./result.json", "r"))
	tf.reset_default_graph()
	run()
	new_res = json.load(open("./result.json", "r"))
	for key in old_res:
		if key[-9:] == 'hashvalue':
			assert old_res[key] == new_res[key]
	tf.reset_default_graph()
