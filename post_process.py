import pickle

# with open("results_full.pkl", "rb") as file:
#     results = pickle.load(file)
with open("results_jax_torch_2025-12-15.pkl", "rb") as file:
    results = pickle.load(file)



import matplotlib.pyplot as plt

plt.plot(list(dict.fromkeys(results["numpy"]["N"])), results["numpy"]["matmul"], label = "np matmul_cpu")
plt.plot(list(dict.fromkeys(results["jax"]["N"])), results["jax"]["matmul"], label = "jax matmul_cpu")
plt.plot(list(dict.fromkeys(results["jax"]["N"])), results["jax"]["einsum"], label = "jax einsum_cpu")
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