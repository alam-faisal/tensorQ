import numpy as np

def rand_dm(n): 
    a = np.random.rand(2**n,2**n) + 1.j * np.random.rand(2**n,2**n)
    a = a@a.conj().T
    return a/np.trace(a)
