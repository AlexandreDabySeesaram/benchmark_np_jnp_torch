import time
import timeit
def numpy_test(methods, N_loop, matrices, output = False, results = None, verbose = True):
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
        if verbose:
            print(f"* Duration numpy [{method}] (ms) = {duration}")
        if output: 
            solver_name = "numpy"
            method_name = method
            size_name = matrices[0].shape[0]
            results.setdefault(solver_name, {}).setdefault(method_name, []).append(duration)
            results.setdefault(solver_name, {}).setdefault("N", []).append(size_name)
    return results


def torch_test(methods, N_loop, matrices, devices = ["cpu"], output = False, results = None, verbose = True):
    import torch

    for device in devices:
        matrices[0] = matrices[0].to(device)
        matrices[1] = matrices[1].to(device)
        if verbose:
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
            if verbose:
                print(f"* Duration torch [{method} on {device}] (ms) = {duration}")
            if output: 
                solver_name = "torch"
                method_name = method+"_"+device
                size_name = matrices[0].shape[0]
                results.setdefault(solver_name, {}).setdefault(method_name, []).append(duration)
                results.setdefault(solver_name, {}).setdefault("N", []).append(size_name)
        if verbose:
            print(f"* Relative error : {(C_1-C_0)/C_0:.7e}")
    return results

def jax_test(methods, N_loop, matrices, output = False, results = None, verbose = True):
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
        if output: 
            solver_name = "jax"
            method_name = method
            size_name = matrices[0].shape[0]
            results.setdefault(solver_name, {}).setdefault(method_name, []).append(duration)
            results.setdefault(solver_name, {}).setdefault("N", []).append(size_name)
        if verbose:
            print(f"* Duration jax [{method}] (ms) = {duration}")
    return results

## Timeit

def numpy_test_ti(methods, N_loop, matrices, output = False, results = None, verbose = True):
    import numpy as np
    for method in methods:
        match method:
            case "matmul":
                elapsed = timeit.timeit(
                                lambda: np.trace(matrices[0]@matrices[1]@matrices[0]@matrices[1]@matrices[0]@matrices[1]),
                                number=N_loop
                            )
                duration = 1000*elapsed/N_loop
            case "einsum":
                try:
                    elapsed = timeit.timeit(
                                    lambda: np.einsum('ij,jk,kl,lm,mn,ni->', matrices[0], matrices[1],matrices[0], matrices[1],matrices[0], matrices[1]),
                                    number=N_loop
                                )
                    duration = 1000*elapsed/N_loop                  
                except:
                    duration = "FAILED"
        if verbose:
            print(f"* Duration numpy ti [{method}] (ms) = {duration}")
        if output: 
            solver_name = "numpy"
            method_name = method
            size_name = matrices[0].shape[0]
            results.setdefault(solver_name, {}).setdefault(method_name, []).append(duration)
            results.setdefault(solver_name, {}).setdefault("N", []).append(size_name)
    return results