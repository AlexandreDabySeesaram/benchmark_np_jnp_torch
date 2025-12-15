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
N_min = 10
N_max = 5000
N_d = 10
N_loop = 10
# results = dict()
import pickle

with open("results_torch_ssh_2025-12-15.pkl", "rb") as f:
    results = pickle.load(f)

#%% Generate data
for N in np.linspace(N_min, N_max, N_d):
    print(f"--> Size of the matrices is {N}X{N}")
    # print(f"--> Number of iter is {N_loop}")
    N = int(np.floor(N))
    A = np.random.rand(N,N).astype(np.float32)
    B = np.random.rand(N,N).astype(np.float32)

    ## Numpy
    matrices_numpy = [A,B]
    # results = bt.numpy_test(methods = ["matmul"], N_loop=N_loop, matrices = matrices_numpy, output=True, results=results, verbose=False)

    ## Jax
    matrices_jax = [jnp.array(A),jnp.array(B)]
    results = bt.jax_test(methods = ["matmul","einsum"], N_loop=N_loop, matrices = matrices_jax, output=True, results=results, verbose=False)



# import pickle

with open("results_jax_torch_ssh_2025-12-15.pkl", "wb") as f:
    pickle.dump(results, f)
# %%
