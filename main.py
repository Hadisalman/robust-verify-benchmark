import argparse
import sys

import models
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from wong_kolter import robust_loss


def evaluate(model, loader, epsilon, alpha=2.0, niters=10, lp_greedy=False):
	dataset_size = len(loader.dataset.test_data)
	err_lpgreedy = 0
	err_pgd = 0
	err_normal = 0
	adv_examples_indices = []
	for X,y in loader:
		X,y = X.cuda(), y.cuda()

		X_pgd = X.clone().detach()
		for i in range(niters): 
			X_pgd.requires_grad_()
			opt = optim.Adam([X_pgd], lr=1e-3)
			opt.zero_grad()
			loss = nn.CrossEntropyLoss()(model(X_pgd), y)
			loss.backward()
			eta = alpha*X_pgd.grad.data.sign()
			X_pgd = X_pgd + eta

			# adjust to be within [-epsilon, epsilon]
			eta = torch.clamp(X_pgd - X, -epsilon, epsilon)
			
			# ensure valid pixel range
			X_pgd = torch.clamp(X + eta, 0, 1.0).detach()

		if lp_greedy:
			_, robust_err = robust_loss(model, epsilon, X, y, bounded_input=True)
			err_lpgreedy += robust_err * X.size(0)

		mismatch = model(X_pgd).data.max(1)[1] != y.data
		adv_examples_indices += mismatch.cpu().numpy().tolist()
		err_pgd += mismatch.float().sum()

		err_normal += (model(X).data.max(1)[1] != y.data).float().sum()
	if lp_greedy:
		return err_normal/dataset_size, err_pgd/dataset_size, err_lpgreedy/dataset_size, adv_examples_indices 
	else:
		return err_normal/dataset_size, err_pgd/dataset_size, None, adv_examples_indices

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--model', required=True, 
		choices=[
			# Exeriment 1 (adversarial error)
			# ------------------------------------
			'ADV_MLP_B_0.03',  	'LPD_MLP_B_0.1',
			'ADV_MLP_B_0.05',  	'LPD_MLP_B_0.2',
			'ADV_MLP_B_0.1',  	'LPD_MLP_B_0.3',
			'ADV_MLP_A_0.1',  	'LPD_MLP_B_0.4',
			'NOR_MLP_B',

			# Exeriment 2 (minimum adversarial distortion)
			# ------------------------------------
			'mnist_cnn_small',  
			'mnist_cnn_wide_1', 
			'mnist_cnn_wide_2', 
			'mnist_cnn_wide_4', 
			'mnist_cnn_wide_8',	'MLP_9_500',
			'mnist_cnn_deep_1',	'MLP_9_100',
			'mnist_cnn_deep_2',	'MLP_2_100',
			])

	parser.add_argument('--epsilon', type=float, help='The maximum allowed l_inf perturbation of the attack during test time.')
	parser.add_argument('--training-mode', choices=['ADV','LPD','NOR'], help='The training mode of the model. \
					- ADV: The PGD training of Madry et. al. \
					- LPD: dual formulation training of Wong and Kolter \
					- NOR: regular XEntropy training \
					If the model name starts with "ADV", "LPD", or "NOR", this argument is ignored; the training mode in \
					this case is infered from the model name.')
	parser.add_argument('--batch-size', type=int, default=100)
	parser.add_argument('--lp-greedy', action='store_true', help='A flag to include lp-greedy as an evaluation method \
																in addition to pgd and normal errors.')
	parser.add_argument('--data_dir', type=str, default='data',
		help='Directory where MNIST dataset is stored.')
	args = parser.parse_args()

	if args.model.startswith(('ADV','LPD','NOR')):
		args.training_mode = args.model.split('_')[0]


	###########################
	# Loading the model
	###########################
	print('--------------------------------------------------------------')
	print('Model: {} | Training mode: {} | Testing epsilon: {}'.format(args.model, args.training_mode, args.epsilon)) 
	print('--------------------------------------------------------------')
	model = models.get_model(args.model, args.training_mode)


	###########################
	# Loading the dataset
	###########################
	testset = datasets.MNIST(args.data_dir, train=False, download=True, transform=transforms.ToTensor())
	test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

	###########################
	# Evaluating the model
	###########################
	# PGD attack params
	niters = 40 # number of attack steps
	alpha = 0.01 # step size

	epsilon = args.epsilon
	err_normal, err_pgd, err_lpgreedy, _ = evaluate(model, test_loader, epsilon, alpha, niters, args.lp_greedy)

	print("Test error is {0:.2f} %".format(err_normal*100))
	print("PGD error is {0:.2f} %".format(err_pgd*100))
	if args.lp_greedy:
		print("LP-Greedy error is {0:.2f} %".format(err_lpgreedy*100))
	###########################

	print('==============================================================')
