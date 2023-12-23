# Statistical_Benefits_TD_experiments
Code for the experiments of the paper 'On the Statistical benefits of TD Learning' (https://arxiv.org/abs/2301.13289)

## Goal 

The paper ['On the Statistical Benefits of TD Learning'](https://arxiv.org/abs/2301.13289) provides a theoretical framework to comparing the statistical efficiency of Temporal Differnce (TD) and Monte Carlo (MC) for policy evaluation. This code complements the theory with experiments on synthetic data. Specially, we are interested in evaluating the MSE of these two methods depending on both the number of samples provided and the structure of the underlying MRP. 

## Code description

The code has 3 main parts: 
- A MRP class, defining the environments (Markov Reward Process) and generating the synthetic data. It is used to create Layered MRP (where the MRP has a structure similar to a fully connected Neural Network) and a variant with some loops (to simulate cyclic MRP).
- An estimate class that given trajectories from a MRP compute the TD and MC estimates. To evaluate the dependence on the number of samples, we need to compute TD and MC estimates for a variating number of samples ($n_1$ samples then $n_2 > n_1$ samples etc... up to $n_k$ samples). For statistical and computational efficiency, we generate $n_k$ trajectories then compute the estimates using only the first $n_1$ samples, the first $n_2$ samples etc... This is computationally efficient as we only generate $n_k$ trajectories instead of $n_1 + \dots + n_k$ and using the same samples across estimates enables to reduce the variance of the estimation. Similarly, we compute the estimates for multiple number of horizons by generating long trajectories and truncating them. The estimate class is able to efficiently handles this computation of estimates for multiple sample sizes and horizons.
- A jupyter notebook to run the experiment: Since a large number of estimates need to be computed for precise results (typically 10^4 estimates, each estimate requiring 2000 samples), the computations are distributed. One addressed challenge is that each process needs to have its own randomness in order to create independent samples. The computations of estimates is broken into two parts: first the estimates are computed in batch and each batch result is written to a file. Then the files are read to compute the MSE across estimates. This makes it easy to increase the number of estimates in the future as well as making it robust to issues that may arise during a long computational process. 

## Libraries required

The code runs in python3/numpy as it is the current default for ML code. Distributed computations are using pathos multiprocessing to avoid pickle issues.

## Running the code

The main code to run are in jupyter notebook. The two notebooks are very similar: one is to observe the effect of changing the horizon while the other is to observe the effect of the sample size. In both cases, it computes the theoretical MSE, compute a number of samples for each value of the variable parameter (either horizon or number of samples) and write it to files then read the files to compute the empirical MSE. Finally, the theoretical and empirical MSEs are ploted side to side. 