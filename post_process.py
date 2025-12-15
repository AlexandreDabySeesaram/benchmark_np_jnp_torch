import pickle



# --- Color Palette Definitions ---
# NumPy: Blues
c_np = 'tab:blue'

# JAX: Purples
c_jax_matmul = 'rebeccapurple'
c_jax_einsum = 'mediumorchid'

# PyTorch: Reds & Oranges
# CPU (Reds)
c_torch_mat_cpu = '#b22222'  # Firebrick
c_torch_ein_cpu = '#f08080'  # Light Coral
# MPS (Oranges)
c_torch_mat_mps = '#ff8c00'  # Dark Orange
c_torch_ein_mps = '#ffbd0a'  # Saffron/Gold


with open("results_jax_torch_2025-12-15.pkl", "rb") as file:
    results = pickle.load(file)



import matplotlib.pyplot as plt



# --- Plotting ---
plt.figure(figsize=(15, 10)) # Added a figure size for better visibility

# NumPy (Blue)
plt.plot(list(dict.fromkeys(results["numpy"]["N"])), results["numpy"]["matmul"], 
         label="M4 pro np matmul_cpu", color=c_np, linewidth=2)

# JAX (Purples)
plt.plot(list(dict.fromkeys(results["jax"]["N"])), results["jax"]["matmul"], 
         label="M4 pro jax matmul_cpu", color=c_jax_matmul, linewidth=2)
plt.plot(list(dict.fromkeys(results["jax"]["N"])), results["jax"]["einsum"], 
         label="M4 pro jax einsum_cpu", color=c_jax_einsum)

# PyTorch (Reds for CPU, Oranges for MPS)
plt.plot(list(dict.fromkeys(results["torch"]["N"])), results["torch"]["matmul_cpu"], 
         label="M4 pro torch matmul_cpu", color=c_torch_mat_cpu, linewidth=2)
plt.plot(list(dict.fromkeys(results["torch"]["N"])), results["torch"]["einsum_cpu"], 
         label="M4 pro torch einsum_cpu", color=c_torch_ein_cpu)

plt.plot(list(dict.fromkeys(results["torch"]["N"])), results["torch"]["matmul_mps"], 
         label="M4 pro torch matmul_mps", color=c_torch_mat_mps, linewidth=2)
plt.plot(list(dict.fromkeys(results["torch"]["N"])), results["torch"]["einsum_mps"], 
         label="M4 pro torch einsum_mps", color=c_torch_ein_mps)
plt.yscale("log") 
plt.legend()
# plt.grid(True, alpha=0.3) # Added a light grid for readability
# plt.show()






with open("results_jax_torch_ssh_2025-12-15.pkl", "rb") as file:
    results_ssh = pickle.load(file)



# --- Plotting ---
# plt.figure(figsize=(10, 6)) # Added a figure size for better visibility

# NumPy (Blue)
plt.plot(list(dict.fromkeys(results_ssh["numpy"]["N"])), results_ssh["numpy"]["matmul"],"-+", 
         label="M2 Ultra np matmul_cpu", color=c_np, linewidth=2)

# JAX (Purples)
plt.plot(list(dict.fromkeys(results_ssh["jax"]["N"])), results_ssh["jax"]["matmul"],"-+", 
         label="M2 Ultra jax matmul_cpu", color=c_jax_matmul, linewidth=2)
plt.plot(list(dict.fromkeys(results_ssh["jax"]["N"])), results_ssh["jax"]["einsum"],"-+", 
         label="M2 Ultra jax einsum_cpu", color=c_jax_einsum)

# PyTorch (Reds for CPU, Oranges for MPS)
plt.plot(list(dict.fromkeys(results_ssh["torch"]["N"])), results_ssh["torch"]["matmul_cpu"],"-+", 
         label="M2 Ultra torch matmul_cpu", color=c_torch_mat_cpu, linewidth=2)
plt.plot(list(dict.fromkeys(results_ssh["torch"]["N"])), results_ssh["torch"]["einsum_cpu"],"-+", 
         label="M2 Ultra torch einsum_cpu", color=c_torch_ein_cpu)

plt.plot(list(dict.fromkeys(results_ssh["torch"]["N"])), results_ssh["torch"]["matmul_mps"], "-+",
         label="M2 Ultra torch matmul_mps", color=c_torch_mat_mps, linewidth=2)
plt.plot(list(dict.fromkeys(results_ssh["torch"]["N"])), results_ssh["torch"]["einsum_mps"], "-+",
         label="M2 Ultra torch einsum_mps", color=c_torch_ein_mps)
plt.yscale("log") 
plt.legend()
plt.grid(True, alpha=0.3) # Added a light grid for readability
plt.xlabel("Size of the matrices")
plt.ylabel("Duration (ms)")
plt.show()