import time

def numpy_test(methods, N_loop, matrices):
    import numpy as np
    for method in methods:
        match method:
            case "matmul":
                start = time.time()
                for i in range(N_loop):
                    C_0 = np.trace(matrices[0]@matrices[1]@matrices[0]@matrices[1]@matrices[0]@matrices[1])
                stop = time.time()
                duration = 1000*(stop-start)/N_loop
            case "einsum":
                try:
                    start = time.time()
                    for i in range(N_loop):
                        C_1 = np.einsum('ij,jk,kl,lm,mn,ni->', matrices[0], matrices[1],matrices[0], matrices[1],matrices[0], matrices[1])
                    stop = time.time()
                    duration = 1000*(stop-start)/N_loop
                except:
                    duration = "FAILED"

        print(f"* Duration numpy [{method}] (ms) = {duration}")

def torch_test(methods, N_loop, matrices, devices = ["cpu"]):
    import torch

    for device in devices:
        matrices[0] = matrices[0].to(device)
        matrices[1] = matrices[1].to(device)
        print(f"device is {matrices[1].device}")

        for method in methods:
            match method:
                case "matmul":
                    start = time.time()
                    for i in range(N_loop):
                        C_0 = torch.trace(matrices[0]@matrices[1]@matrices[0]@matrices[1]@matrices[0]@matrices[1])
                    stop = time.time()
                    duration = 1000*(stop-start)/N_loop

                case "einsum":
                    try:
                        start = time.time()
                        for i in range(N_loop):
                            C_1 = torch.einsum('ij,jk,kl,lm,mn,ni->', matrices[0], matrices[1],matrices[0], matrices[1],matrices[0], matrices[1])
                        stop = time.time()
                        duration = 1000*(stop-start)/N_loop
                    except:
                        duration = "FAILED"

            print(f"* Duration torch [{method} on {device}] (ms) = {duration}")


def jax_test(methods, N_loop, matrices):
    import os
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    import jax.numpy as jnp
    for method in methods:
        match method:
            case "matmul":
                start = time.time()
                for i in range(N_loop):
                    C_0 = jnp.trace(matrices[0]@matrices[1]@matrices[0]@matrices[1]@matrices[0]@matrices[1])
                stop = time.time()
                duration = 1000*(stop-start)/N_loop

            case "einsum":
                try:
                    start = time.time()
                    for i in range(N_loop):
                        C_1 = jnp.einsum('ij,jk,kl,lm,mn,ni->', matrices[0], matrices[1],matrices[0], matrices[1],matrices[0], matrices[1])
                    stop = time.time()
                    duration = 1000*(stop-start)/N_loop
                except:
                    duration = "FAILED"

        print(f"* Duration jax [{method}] (ms) = {duration}")