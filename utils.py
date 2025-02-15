import numpy as np
import scipy.io
from .circuits import *

def base(b,n,length):
    """ base b representation of number n assuming there are length dits """
    size = int(np.ceil(np.log(max(1,n))/np.log(b)))
    listy = [place
        for i in range(size,-1,-1)
        if (place := n//b**i%b) or i<size] or [0]
    for _ in range(length-len(listy)):
        listy.insert(0, 0)
    return listy

def string_to_sv(string, local_dim): 
    n = len(string)
    sv = np.zeros(local_dim**n)
    sv[int(string, local_dim)] = 1.0
    return sv

def pretty_print_dm(dmat, local_dim, threshold=1e-3):
    """ prints density matrix as a mixture of quantum states """
    evals, evecs = np.linalg.eigh(dmat)
    n = dmat.shape[0]
    length = int(np.log(n) / np.log(local_dim))

    # Sort eigenvalues and eigenvectors by absolute value of eigenvalues in descending order
    idx = np.argsort(np.abs(evals))[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    result = []
    for i, (eval, evec) in enumerate(zip(evals, evecs.T)):
        if abs(eval) > threshold:
            non_zero_indices = np.where(np.abs(evec) > threshold)[0]
            terms = []
            for idx in non_zero_indices:
                base_repr = ''.join(map(str, base(local_dim, idx, length)))
                term = f"{evec[idx]:.4f}|{base_repr}⟩"
                terms.append(term)
            state_repr = " + ".join(terms)
            result.append(f"{eval.real:.4f} * ({state_repr})")
    
    print("\n ".join(result))
    
def pretty_print_sv(sv, local_dim, threshold=1e-3):
    """ prints statevector as a linear combination of computational basis states """
    n = sv.shape[0]
    length = round(np.log(n) / np.log(local_dim))

    non_zero_indices = np.where(np.abs(sv) > threshold)[0]
    terms = []
    for idx in non_zero_indices:
        base_repr = ''.join(map(str, base(local_dim, idx, length)))
        term = f"{sv[idx]:.4f}|{base_repr}⟩"
        terms.append(term)
    state_repr = " + ".join(terms)
    
    print("".join(state_repr))
    
#################################
##### Fidelities ################
#################################

from scipy.linalg import ishermitian
def sqrtm(hmat): 
    if ishermitian(hmat, atol=1e-8):
        evals, evecs = np.linalg.eigh(hmat)
        evals[evals < 0] = 0.0
        return evecs @ np.diag(np.sqrt(evals.real)) @ evecs.conj().T
    else: 
        raise ValueError("provided matrix is not Hermitian to tolerance 1e-12")

def fidelity(rho1, rho2): 
    rho1_sq = sqrtm(rho1)
    arg = rho1_sq @ rho2 @ rho1_sq
    arg_sq = sqrtm(arg)
    return np.trace(arg_sq).real**2

def hellinger_fidelity(rho1, rho2): 
    return sum(np.sqrt(np.diag(rho1) * np.diag(rho2))).real**2

def two_norm_fidelity(rho1, rho2): 
    if type(rho1) == np.ndarray:
        return np.trace(rho1.conj().T @ rho2).real
    elif isinstance(rho1, MPO): 
        return (rho1.conj() @ rho2).trace(scaled=False).real
    else: 
        raise TypeError(f"provided density operator is not of valid type; must be {np.ndarray} or {MPO} not {type(rho1)}")