#%%% Imports 
import numpy as np
import time
import benchmark_toolkit as bt
import torch

#%% Fix seed 
np.random.seed(0)


#%% Benchmark parameters
N = 1000
N_loop = 10

print(f"--> Size of the matrices is {N}X{N}")
print(f"--> Number of iter is {N_loop}")

#%% Generate data

A = np.random.rand(N,N).astype(np.float32)
B = np.random.rand(N,N).astype(np.float32)

## Numpy
matrices_numpy = [A,B]
bt.numpy_test(methods = ["matmul"], N_loop=N_loop, matrices = matrices_numpy)


## torch
matrices_torch = [torch.tensor(A),torch.tensor(B)]


bt.torch_test(methods = ["matmul","einsum"], N_loop=N_loop, matrices = matrices_torch, devices = ["cpu","mps"])
