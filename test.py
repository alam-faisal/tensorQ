in_strings = ['00', '01', '10']
local_dim = 2
unitary = random_unitary(2)
out_states = [unitary @ string_to_sv(in_str, local_dim) for in_str in in_strings]
Q = construct_unitary(in_strings, out_states, local_dim)

print(np.allclose(Q.conj().T @ Q, np.eye(4)), np.allclose(Q @ Q.conj().T, np.eye(4))) 

for st in in_strings: 
    print(np.allclose(Q @ string_to_sv(st, local_dim), unitary @ string_to_sv(st, local_dim))) 