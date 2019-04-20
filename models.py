import os

import numpy as np 
import torch
import torch.nn as nn

def get_model(model_name, training_mode):

	## Models used in the adversarial error experiment
	if model_name == 'ADV_MLP_B_0.03':
		model = adv_MLP_B_03().cuda()

	elif model_name == 'ADV_MLP_B_0.05':
		model = adv_MLP_B_05().cuda()

	elif model_name == 'ADV_MLP_B_0.1':
		model = adv_MLP_B_1().cuda()

	elif model_name == 'ADV_MLP_A_0.1':
		"""
		This model is taken from https://github.com/vtjeng/MIPVerify_data
		"""
		b1 = np.load("./weights/experiment_1/MIPVerify_data/weights/mnist/RSL18a/linf0.1/B1.npy")
		b2 = np.load("./weights/experiment_1/MIPVerify_data/weights/mnist/RSL18a/linf0.1/B2.npy")
		w1 = np.load("./weights/experiment_1/MIPVerify_data/weights/mnist/RSL18a/linf0.1/W1.npy")
		w2 = np.load("./weights/experiment_1/MIPVerify_data/weights/mnist/RSL18a/linf0.1/W2.npy")

		model = nn.Sequential(Flatten(),
							  nn.Linear(w1.shape[0],w1.shape[1]),
							  nn.ReLU(),
							  nn.Linear(w2.shape[0],w2.shape[1])
							  )
		model[1].weight.data = torch.Tensor(w1).t()
		model[1].bias.data = torch.Tensor(b1)
		model[3].weight.data = torch.Tensor(w2).t()
		model[3].bias.data = torch.Tensor(b2)
		model = model.cuda()

	elif model_name == 'NOR_MLP_B':
		model = normal_MLP_B().cuda()

	elif model_name == 'LPD_MLP_B_0.1':
		model = lpd_MLP_B_1().cuda()

	elif model_name == 'LPD_MLP_B_0.2':
		model = lpd_MLP_B_2().cuda()

	elif model_name == 'LPD_MLP_B_0.3':
		model = lpd_MLP_B_3().cuda()

	elif model_name == 'LPD_MLP_B_0.4':
		model = lpd_MLP_B_4().cuda()


	## Models used in the minimum adversarial distortion experiment 
	elif model_name == 'mnist_cnn_small':
		model = mnist_cnn_small(training_mode).cuda()

	elif model_name == 'mnist_cnn_wide_1':
		model = mnist_cnn_wide_1(training_mode).cuda()

	elif model_name == 'mnist_cnn_wide_2':
		model = mnist_cnn_wide_2(training_mode).cuda()

	elif model_name == 'mnist_cnn_wide_4':
		model = mnist_cnn_wide_4(training_mode).cuda()

	elif model_name == 'mnist_cnn_wide_8':
		model = mnist_cnn_wide_8(training_mode).cuda()

	elif model_name == 'mnist_cnn_deep_1':
		model = mnist_cnn_deep_1(training_mode).cuda()

	elif model_name == 'mnist_cnn_deep_2':
		model = mnist_cnn_deep_2(training_mode).cuda()

	elif model_name == 'MLP_9_500':
		model = MLP_9_500(training_mode).cuda()

	elif model_name == 'MLP_9_100':
		model = MLP_9_100(training_mode).cuda()

	elif model_name == 'MLP_2_100':
		model = MLP_2_100(training_mode).cuda()

	elif model_name == 'cifar_cnn_small':
		model = cifar_cnn_small(training_mode).cuda()

	elif model_name == 'cifar_cnn_wide_1':
		model = cifar_cnn_wide_1(training_mode).cuda()

	elif model_name == 'cifar_cnn_wide_2':
		model = cifar_cnn_wide_2(training_mode).cuda()

	elif model_name == 'cifar_cnn_wide_4':
		model = cifar_cnn_wide_4(training_mode).cuda()

	return model


# Experiment 1 models
def adv_MLP_B_03():
	net = _MLP_2_100_base()
	state_dict = torch.load('weights/experiment_1/ADV_MLP_B_0.03.pth')
	net.load_state_dict(state_dict)
	return net

def adv_MLP_B_05():
	net = _MLP_2_100_base()
	state_dict = torch.load('weights/experiment_1/ADV_MLP_B_0.05.pth')
	net.load_state_dict(state_dict)
	return net

def adv_MLP_B_1():
	net = _MLP_2_100_base()
	state_dict = torch.load('weights/experiment_1/ADV_MLP_B_0.1.pth')
	net.load_state_dict(state_dict)
	return net

# ------------------------------------------------------------------------------
def normal_MLP_B():
	net = _MLP_2_100_base()
	state_dict = torch.load('weights/experiment_1/NOR_MLP_B.pth')
	net.load_state_dict(state_dict)
	return net

# ------------------------------------------------------------------------------
def lpd_MLP_B_1():
	net = _MLP_2_100_base()
	state_dict = torch.load('weights/experiment_1/LPD_MLP_B_0.1.pth')
	net.load_state_dict(state_dict)
	return net

def lpd_MLP_B_2():
	net = _MLP_2_100_base()
	state_dict = torch.load('weights/experiment_1/LPD_MLP_B_0.2.pth')
	net.load_state_dict(state_dict)
	return net

def lpd_MLP_B_3():
	net = _MLP_2_100_base()
	state_dict = torch.load('weights/experiment_1/LPD_MLP_B_0.3.pth')
	net.load_state_dict(state_dict)
	return net

def lpd_MLP_B_4():
	net = _MLP_2_100_base()
	state_dict = torch.load('weights/experiment_1/LPD_MLP_B_0.4.pth')
	net.load_state_dict(state_dict)	
	return net




# Experiment 2 models

###################################################################
######################### MNIST models ############################
###################################################################

def mnist_cnn_small(training_mode=None):
	net = nn.Sequential(
		nn.Conv2d(1, 16, 4, stride=2, padding=1),
		nn.ReLU(),
		nn.Conv2d(16, 32, 4, stride=2, padding=1),
		nn.ReLU(),
		Flatten(),
		nn.Linear(32*7*7,100),
		nn.ReLU(),
		nn.Linear(100, 10)
	)

	if training_mode == 'NOR':
		weights_path = './weights/experiment_2/mnist_cnn_small_NOR.pth'
		
	elif training_mode == 'ADV':
		weights_path = './weights/experiment_2/mnist_cnn_small_ADV.pth'

	elif training_mode == 'LPD':
		weights_path = './weights/experiment_2/mnist_cnn_small_LPD.pth'

	else: 
		print('The "training_mode" is not understood... Returning a randomly initialized network.')
		return net

	state_dict = torch.load(weights_path)
	net.load_state_dict(state_dict)
	return net


def mnist_cnn_wide_1(training_mode=None): 
	net = _model_wide(1, 7, 1)

	if training_mode == 'NOR':
		weights_path = './weights/experiment_2/mnist_cnn_wide_1_NOR.pth'

	elif training_mode == 'ADV':
		weights_path = './weights/experiment_2/mnist_cnn_wide_1_ADV.pth'

	elif training_mode == 'LPD':
		weights_path = './weights/experiment_2/mnist_cnn_wide_1_LPD.pth'

	else: 
		print('The "training_mode" is not understood... Returning a randomly initialized network.')
		return net

	state_dict = torch.load(weights_path)
	net.load_state_dict(state_dict)
	return net

def mnist_cnn_wide_2(training_mode=None): 
	net = _model_wide(1, 7, 2)
	
	if training_mode == 'NOR':
		weights_path = './weights/experiment_2/mnist_cnn_wide_2_NOR.pth'

	elif training_mode == 'ADV':
		weights_path = './weights/experiment_2/mnist_cnn_wide_2_ADV.pth'

	elif training_mode == 'LPD':
		weights_path = './weights/experiment_2/mnist_cnn_wide_2_LPD.pth'

	else: 
		print('The "training_mode" is not understood... Returning a randomly initialized network.')
		return net

	state_dict = torch.load(weights_path)
	net.load_state_dict(state_dict)
	return net

def mnist_cnn_wide_4(training_mode=None): 
	net = _model_wide(1, 7, 4)

	if training_mode == 'NOR':
		weights_path = './weights/experiment_2/mnist_cnn_wide_4_NOR.pth'

	elif training_mode == 'ADV':
		weights_path = './weights/experiment_2/mnist_cnn_wide_4_ADV.pth'

	elif training_mode == 'LPD':
		weights_path = './weights/experiment_2/mnist_cnn_wide_4_LPD.pth'

	else: 
		print('The "training_mode" is not understood... Returning a randomly initialized network.')
		return net

	state_dict = torch.load(weights_path)
	net.load_state_dict(state_dict)
	return net

def mnist_cnn_wide_8(training_mode=None): 
	net = _model_wide(1, 7, 8)
	if training_mode == 'NOR':
		weights_path = './weights/experiment_2/mnist_cnn_wide_8_NOR.pth'

	elif training_mode == 'ADV':
		weights_path = './weights/experiment_2/mnist_cnn_wide_8_ADV.pth'

	else: 
		print('The "training_mode" is not understood... Returning a randomly initialized network.')
		return net

	state_dict = torch.load(weights_path)
	net.load_state_dict(state_dict)
	return net


def mnist_cnn_deep_1(training_mode=None):
	net = _model_deep(1, 7, 1)
	if training_mode == 'NOR':
		weights_path = './weights/experiment_2/mnist_cnn_deep_1_NOR.pth'

	elif training_mode == 'ADV':
		weights_path = './weights/experiment_2/mnist_cnn_deep_1_ADV.pth'

	elif training_mode == 'LPD':
		weights_path = './weights/experiment_2/mnist_cnn_deep_1_LPD.pth'

	else: 
		print('The "training_mode" is not understood... Returning a randomly initialized network.')
		return net

	state_dict = torch.load(weights_path)
	net.load_state_dict(state_dict)
	return net

def mnist_cnn_deep_2(training_mode=None):
	net = _model_deep(1, 7, 2)
	if training_mode == 'NOR':
		weights_path = './weights/experiment_2/mnist_cnn_deep_2_NOR.pth'

	elif training_mode == 'ADV':
		weights_path = './weights/experiment_2/mnist_cnn_deep_2_ADV.pth'

	else: 
		print('The "training_mode" is not understood... Returning a randomly initialized network.')
		return net

	state_dict = torch.load(weights_path)
	net.load_state_dict(state_dict)
	return net

def MLP_2_100(training_mode=None):
	net = _MLP_2_100_base()

	if training_mode == 'NOR':
		weights_path = './weights/experiment_2/mnist_MLP_2_100_NOR.pth'

	elif training_mode == 'ADV':
		weights_path = './weights/experiment_2/mnist_MLP_2_100_ADV.pth'

	elif training_mode == 'LPD':
		weights_path = './weights/experiment_2/mnist_MLP_2_100_LPD.pth'

	else: 
		print('The "training_mode" is not understood... Returning a randomly initialized network.')
		return net

	state_dict = torch.load(weights_path)
	net.load_state_dict(state_dict)
	return net

def MLP_9_100(training_mode=None):
	net = nn.Sequential(
		Flatten(),
		nn.Linear(784,100),
		nn.ReLU(),
		nn.Linear(100,100),
		nn.ReLU(),
		nn.Linear(100,100),
		nn.ReLU(),
		nn.Linear(100,100),
		nn.ReLU(),
		nn.Linear(100,100),
		nn.ReLU(),
		nn.Linear(100,100),
		nn.ReLU(),
		nn.Linear(100,100),
		nn.ReLU(),
		nn.Linear(100,100),
		nn.ReLU(),
		nn.Linear(100,100),
		nn.ReLU(),
		nn.Linear(100,10)
	)

	if training_mode == 'NOR':
		weights_path = './weights/experiment_2/mnist_MLP_9_100_NOR.pth'

	elif training_mode == 'ADV':
		weights_path = './weights/experiment_2/mnist_MLP_9_100_ADV.pth'

	elif training_mode == 'LPD':
		weights_path = './weights/experiment_2/mnist_MLP_9_100_LPD.pth'

	else: 
		print('The "training_mode" is not understood... Returning a randomly initialized network.')
		return net

	state_dict = torch.load(weights_path)
	net.load_state_dict(state_dict)
	return net


def MLP_9_500(training_mode=None):
	net = nn.Sequential(
		Flatten(),
		nn.Linear(784,500),
		nn.ReLU(),
		nn.Linear(500,500),
		nn.ReLU(),
		nn.Linear(500,500),
		nn.ReLU(),
		nn.Linear(500,500),
		nn.ReLU(),
		nn.Linear(500,500),
		nn.ReLU(),
		nn.Linear(500,500),
		nn.ReLU(),
		nn.Linear(500,500),
		nn.ReLU(),
		nn.Linear(500,500),
		nn.ReLU(),
		nn.Linear(500,500),
		nn.ReLU(),
		nn.Linear(500,10)
	)

	if training_mode == 'NOR':
		weights_path = './weights/experiment_2/mnist_MLP_9_500_NOR.pth'

	elif training_mode == 'ADV':
		weights_path = './weights/experiment_2/mnist_MLP_9_500_ADV.pth'

	else: 
		print('The "training_mode" is not understood... Returning a randomly initialized network.')
		return net

	state_dict = torch.load(weights_path)
	net.load_state_dict(state_dict)
	return net




###################################################################
######################### CIFAR models ############################
###################################################################
def cifar_cnn_small(training_mode=None):
	net = nn.Sequential(
		nn.Conv2d(3, 16, 4, stride=2, padding=1),
		nn.ReLU(),
		nn.Conv2d(16, 32, 4, stride=2, padding=1),
		nn.ReLU(),
		Flatten(),
		nn.Linear(32*8*8,100),
		nn.ReLU(),
		nn.Linear(100, 10)
	)	
	
	if training_mode == 'NOR':
		weights_path = './weights/experiment_2/cifar_cnn_small_NOR.pth'

	elif training_mode == 'ADV':
		weights_path = './weights/experiment_2/cifar_cnn_small_ADV.pth'

	elif training_mode == 'LPD':
		weights_path = './weights/experiment_2/cifar_cnn_small_LPD.pth'

	else: 
		print('The "training_mode" is not understood... Returning a randomly initialized network.')
		return net

	state_dict = torch.load(weights_path)
	net.load_state_dict(state_dict)
	return net


def cifar_cnn_wide_1(training_mode=None): 
	net = _model_wide(3, 8, 1)
	if training_mode == 'NOR':
		weights_path = './weights/experiment_2/cifar_cnn_wide_1_NOR.pth'

	elif training_mode == 'ADV':
		weights_path = './weights/experiment_2/cifar_cnn_wide_1_ADV.pth'

	elif training_mode == 'LPD':
		weights_path = './weights/experiment_2/cifar_cnn_wide_1_LPD.pth'

	else: 
		print('The "training_mode" is not understood... Returning a randomly initialized network.')
		return net

	state_dict = torch.load(weights_path)
	net.load_state_dict(state_dict)
	return net

def cifar_cnn_wide_2(training_mode=None): 
	net = _model_wide(3, 8, 2)
	if training_mode == 'NOR':
		weights_path = './weights/experiment_2/cifar_cnn_wide_2_NOR.pth'

	elif training_mode == 'ADV':
		weights_path = './weights/experiment_2/cifar_cnn_wide_2_ADV.pth'

	elif training_mode == 'LPD':
		weights_path = './weights/experiment_2/cifar_cnn_wide_2_LPD.pth'

	else: 
		print('The "training_mode" is not understood... Returning a randomly initialized network.')
		return net

	state_dict = torch.load(weights_path)
	net.load_state_dict(state_dict)
	return net

def cifar_cnn_wide_4(training_mode=None): 
	net = _model_wide(3, 8, 4)
	if training_mode == 'NOR':
		weights_path = './weights/experiment_2/cifar_cnn_wide_4_NOR.pth'

	elif training_mode == 'ADV':
		weights_path = './weights/experiment_2/cifar_cnn_wide_4_ADV.pth'

	elif training_mode == 'LPD':
		weights_path = './weights/experiment_2/cifar_cnn_wide_4_LPD.pth'

	else: 
		print('The "training_mode" is not understood... Returning a randomly initialized network.')
		return net

	state_dict = torch.load(weights_path)
	net.load_state_dict(state_dict)
	return net


#########################################################################################################
# Base models taken from https://github.com/locuslab/convex_adversarial/blob/master/examples/problems.py
#########################################################################################################

class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)

def _MLP_2_100_base():
	net = nn.Sequential(
		Flatten(),
		nn.Linear(784, 100),
		nn.ReLU(),
		nn.Linear(100, 100),
		nn.ReLU(),
		nn.Linear(100, 10)
	)
	return net

def _model_wide(in_ch, out_width, k): 
	model = nn.Sequential(
		nn.Conv2d(in_ch, 4*k, 4, stride=2, padding=1),
		nn.ReLU(),
		nn.Conv2d(4*k, 8*k, 4, stride=2, padding=1),
		nn.ReLU(),
		Flatten(),
		nn.Linear(8*k*out_width*out_width,k*128),
		nn.ReLU(),
		nn.Linear(k*128, 10)
	)
	return model

def _model_deep(in_ch, out_width, k, n1=8, n2=16, linear_size=100): 
	def group(inf, outf, N): 
		if N == 1: 
			conv = [nn.Conv2d(inf, outf, 4, stride=2, padding=1), 
						 nn.ReLU()]
		else: 
			conv = [nn.Conv2d(inf, outf, 3, stride=1, padding=1), 
						 nn.ReLU()]
			for _ in range(1,N-1):
				conv.append(nn.Conv2d(outf, outf, 3, stride=1, padding=1))
				conv.append(nn.ReLU())
			conv.append(nn.Conv2d(outf, outf, 4, stride=2, padding=1))
			conv.append(nn.ReLU())
		return conv

	conv1 = group(in_ch, n1, k)
	conv2 = group(n1, n2, k)

	model = nn.Sequential(
		*conv1, 
		*conv2,
		Flatten(),
		nn.Linear(n2*out_width*out_width,linear_size),
		nn.ReLU(),
		nn.Linear(100, 10)
	)
	return model

