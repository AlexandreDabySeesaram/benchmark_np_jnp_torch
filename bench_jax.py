#%%% Imports 
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np
import jax
import jax.numpy as jnp
import time
import benchmark_toolkit as bt

#%% Fix seed 
np.random.seed(0)

#%% Benchmark parameters
N = 5000
N_loop = 50

print(f"--> Size of the matrices is {N}X{N}")
print(f"--> Number of iter is {N_loop}")

#%% Generate data

A = np.random.rand(N,N).astype(np.float32)
B = np.random.rand(N,N).astype(np.float32)

## Numpy
matrices_numpy = [A,B]
bt.numpy_test(methods = ["matmul"], N_loop=N_loop, matrices = matrices_numpy)

## Jax
matrices_jax = [jnp.array(A),jnp.array(B)]
bt.jax_test(methods = ["matmul","einsum"], N_loop=N_loop, matrices = matrices_jax)

