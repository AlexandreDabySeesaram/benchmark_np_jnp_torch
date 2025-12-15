#%%% Imports 
import numpy as np
import time
import benchmark_toolkit as bt
import torch

#%% Fix seed 
np.random.seed(0)

#%% Benchmark parameters
N_min = 10
N_max = 5000
N_d = 10
N_loop = 10
results = dict()

#%% Generate data
for N in np.linspace(N_min, N_max, N_d):
    print(f"--> Size of the matrices is {N}X{N}")
    # print(f"--> Number of iter is {N_loop}")
    N = int(np.floor(N))
    A = np.random.rand(N,N).astype(np.float32)
    B = np.random.rand(N,N).astype(np.float32)

    ## Numpy
    matrices_numpy = [A,B]
    results = bt.numpy_test(methods = ["matmul"], N_loop=N_loop, matrices = matrices_numpy, output=True, results=results, verbose=False)

    ## torch
    matrices_torch = [torch.tensor(A),torch.tensor(B)]
    results = bt.torch_test(methods = ["matmul","einsum"], N_loop=N_loop, matrices = matrices_torch, devices = ["cpu","mps"], output=True, results=results, verbose=False)


import matplotlib.pyplot as plt

plt.plot(list(dict.fromkeys(results["numpy"]["N"])), results["numpy"]["matmul"], label = "np matmul_cpu")
# plt.plot(list(dict.fromkeys(results["numpy"]["N"])), results["numpy"]["einsum"], label = "np einsum_cpu")
plt.plot(list(dict.fromkeys(results["torch"]["N"])), results["torch"]["matmul_cpu"], label = "torch matmul_cpu")
plt.plot(list(dict.fromkeys(results["torch"]["N"])), results["torch"]["einsum_cpu"], label = "torch einsum_cpu")
plt.plot(list(dict.fromkeys(results["torch"]["N"])), results["torch"]["matmul_mps"], label = "torch matmul_mps")
plt.plot(list(dict.fromkeys(results["torch"]["N"])), results["torch"]["einsum_mps"], label = "torch einsum_mps")
plt.xlabel("Size matrices")
plt.ylabel("Duration (ms)")
plt.yscale("log") 
plt.legend() 
plt.show()

import pickle

with open("results.pkl", "wb") as f:
    pickle.dump(results, f)


