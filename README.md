# Benchmark for LP-relaxed robustness verification of ReLU-networks

Recently, there has been much progress on fast and accurate verification of neural networks robustness, with many methods based on convex relaxation. However, as a result of our recent paper:

**A Convex Relaxation Barrier to Tight Robustness Verification of Neural Networks** <br>
*Hadi Salman, Greg Yang, Huan Zhang, Cho-Jui Hsieh, Pengchuan Zhang* <br>
https://arxiv.org/abs/1902.08722

we show that most of such convex-relaxed robustness verification methods (which are described by our framework presented in the paper) cannot obtain better results than the *optimal layer-wise LP-relaxed verification method (**LP-ALL**)*. LP-ALL is computationally expensive and not a practical algorithm, but with careful engineering and 22 CPU-years of compute we were able to obtain the results of this method on various networks, which serve as strict upper bounds on the aforementioned methods. We are opensourcing these models and the corresponding results as a benchmark to measure progress in robustness verification.

*Recommended Usage*.
Run your robustness verification algorithm on the pretrained models in this repo and compare against the numbers provided here (or in the paper). Try to beat the numbers while keeping your algorithm fast!

## Getting started

1.  `git clone https://github.com/Hadisalman/robust-verify-benchmark.git`

2.  Install dependencies:
```
conda create -n lp-relaxation
conda activate lp-relaxation
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch # for Linux
conda install jupyter matplotlib
pip install cvxpy
```
If everything goes well, you should be abe to run the `demo.ipynb` successfully.

## Overview of the Code
The code consists of two Python scripts and the notebook `demo.ipynb`.

- `main.py` is a skeleton code that is used to load our pretrained models and evaluate them (specific to MNIST models). By evaluate, we mean: 
	1. calculate the test error on the MNIST test set.
	2. calculate a lower bound on the  adversarial error using the PGD attack of [Madry et al. 2017](https://arxiv.org/abs/1706.06083).
	3. calculate an upper bound on the adversarial error using the LP-GREEDY certification method of [Wong and Kolter, 2018](https://arxiv.org/pdf/1711.00851.pdf).
- `models.py` contain all the models (MNIST and CIFAR-10) that we use in our paper.

### Parameters for `main.py`
- `model`: the name of the pretrained model to load and evaluate.
- `epsilon`: The maximum allowed l_inf perturbation of the attack/certification method during test time.
- `training-mode`: The training mode of the model:
	* ADV: adversarial training of [Madry et al. 2017](https://arxiv.org/abs/1706.06083)
	* LPD: dual formulation training of [Wong and Kolter, 2018](https://arxiv.org/pdf/1711.00851.pdf)
	* NOR: regular X-Entropy training
	
	 If the model name starts with "ADV", "LPD", or "NOR", this argument is ignored; the training mode in this case is inferred from the model's name.')
- `batch-size`: the batch size used during evaluation. *Decrease this if you experience memory issues*.
- `lp-greedy`: A flag to include LP-GREEDY as an evaluation method in addition to pgd and normal error. The LP-GREEDY certification method takes significant more time compared to calculating the pgd or normal test errors for large models, so we provide this flag in case you really need the LP-GREEDY bound.
- `data_dir`: Directory where the MNIST dataset is stored.

### Demo notebook
We provide a demo notebook `demo.ipynb` which contains a toy neural network verification problem using the different methods included in our paper: MILP, LP-ALL, LP-LAST, and LP-GREEDY. The purpose of this demo is to show the differences in the implementations of each, and to get a rough idea of how tight each of these methods are.

## Experiment 1: Finding the adversarial error
The aim of this experiment is to find bounds on the adverarial error (lower is better) using different certification techniques. The closer the bound is to the MILP bounds, the better the certification method is (from the table, note that LP-ALL is tighter than LP-GREEDY).

The table below summarizes the results of Section 6.1 in the paper. It is the same as Table 1 in our paper. 

Network | &epsilon; | Test Error | <sup>Lower Bound</sup> <br> PGD |<sup>Lower Bound</sup> <br> MILP |<sup>Upper Bound</sup> <br> MILP | <sup>Upper Bound</sup> <br> LP-ALL| <sup>Upper Bound</sup> <br> LP-GREEDY| 
------------|------------|----------------|-----------------------|---------------------|----------------|-----------|-------------|
Adv-MLP-B   | 0.03      |1.53 %            |4.17  %                 | 4.18 %                |5.78  %      | 10.04 % | 13.40% |
Adv-MLP-B   | 0.05      |1.62 %            |6.06  %                 | 6.11 %                |11.38 %      | 23.29 % | 33.09 %| 
Adv-MLP-B   | 0.1       |3.33 %            |15.86 %                 |16.25 %                |34.37 %      | 61.59 % | 71.34 %| 
Adv-MLP-A   | 0.1       |4.18 %            |11.51 %                 |14.36 %                |30.81 %      | 60.14 % | 67.50 %|
Nor-MLP-B   | 0.02      |2.05 %            |10.06 %                 |10.16 %                |13.48 %      | 26.41 % | 35.11 %| 
Nor-MLP-B   | 0.03      |2.05 %            |20.37 %                 |20.43 %                |48.67 %      | 65.70 % | 75.85 %|
Nor-MLP-B   | 0.05      |2.05 %            |53.37 %                 |53.37 %                |94.04 %      | 97.95 % | 99.39 %|  
LPd-MLP-B   | 0.1       |4.09 %            |13.39 %                 |14.45 %                |14.45 %      | 17.24 % | 18.32 %|  
LPd-MLP-B   | 0.2       |15.72 %           |33.85 %                 |36.33 %                |36.33 %      | 37.50 % | 41.67 %| 
LPd-MLP-B   | 0.3       |39.22 %           |57.29 %                 |59.85 %                |59.85 %      | 60.17 % | 66.85 %|
LPd-MLP-B   | 0.4       |67.97 %           |81.85 %                 |83.17 %                |83.17 %      | 83.62 % | 87.89 %|

* We provide code to replicate the Test Error, PGD, and LP-GREEDY columns. The MILP and LP-ALL columns took around 22 CPU years to compute on a cluster of 1000 CPUs. Therefore, we just include the entries as a benchmark for people to compare their results against.

To replicate the above table, simply run:
```
bash run_experiment_1.sh
```
This bash script loads all the pretrained models, and calculates the Test Error, PGD adversarial error, and LP-GREEDY adversarial error for each on MNIST test set.

* If you would like to test your own certification technique on these models and compare with our benchmark, check `main.py` which is a skeleton code for loading our **pretrained** models and evaluating them.

For example:
```
python main.py --model ADV_MLP_B_0.03 --epsilon 0.03 --lp-greedy
```
replicates the first row of the above table, i.e., loads the PGD-trained network `ADV_MLP_B_0.03` and evaluates it using a maximum allowed l_infinity perturbation &epsilon;=0.3.

## Experiment 2: Finding the minimum adversarial distortion
For this experiment, the aim is to find the minimum adversarial distortion &epsilon; using different certification techniques. We used binary search to find &epsilon;. Details can be found in Appendix E of our paper. Because this is computationally very expensive, we do it for only [10 samples of MNIST](https://github.com/Hadisalman/icml_public/blob/master/weights/experiment_2/ten_test_images.npy). We report the results in [Table 2 of our paper](https://github.com/Hadisalman/icml_public/blob/master/tables/table2.png).

* We provide code to load the **pretrained** models that we use in our paper. You can use this code along with your own certification method to search &epsilon; and compare with our results in Table 2 of our paper. Check `main.py` which is a skeleton code for loading our pretrained models and evaluating them.

For example:
```
python main.py --model mnist_cnn_small --epsilon 0.1 --training-mode LPD
```
loads a small mnist_cnn_small model (in accordance with the names in our paper), which is trained using the dual formulation of [Wong and Kolter 2018](https://arxiv.org/abs/1711.00851), and evaluates this model using a maximum allowed l_infinity perturbation &epsilon;=0.1.

* You can run this
```
bash run_experiment_2.sh
```
to load and evaluate all the models that are used in this experiment of our paper.
