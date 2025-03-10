from .tn import *
from .noise import *
from scipy.linalg import block_diag

class Gate:
    def __init__(self, name, indices, matrix):
        self.name = name
        self.indices = [indices] if type(indices) == int else indices
        self.span = len(self.indices)
        self.matrix = matrix

    def __str__(self):
        return f"{self.name} gate on site(s) {self.indices}"

    def to_superop(self, nm=None): 
        if self.name != 'ID' or nm is None: 
            return np.kron(self.matrix.conj(), self.matrix).reshape([self.get_local_dim()]*4*self.span)
        else: 
            return nm.get_superop(self.time).reshape([self.get_local_dim()]*4*self.span)
        
    def get_local_dim(self):
        return int(self.matrix.shape[0]**(1/self.span))
    
    def dag(self): 
        g = copy.deepcopy(self)
        g.matrix = g.matrix.conj().T
        return g
    
class SuperOp:
    def __init__(self, name, span, matrix, start_idx=0, time=0.0):
        self.name = name
        
        self.start_idx = start_idx
        self.indices = [start_idx+i for i in range(span)]
        self.span = span
        
        self.matrix = matrix
        self.shape = matrix.shape 
        self.time = time 
        
    def shift(self, new_start_idx): 
        return SuperOp(self.name, self.span, self.matrix, start_idx=new_start_idx, time=self.time)
    
    def __str__(self):
        return f"{self.name} superoperator on site(s) {self.indices}"

###########################################
############# SUBCLASSES ##################
###########################################

class ID(Gate): 
    def __init__(self, local_dim, time, indices): 
        super().__init__("ID", indices, np.eye(local_dim))
        self.time = time
        
    def __str__(self):
        return f"Idling on site(s) {self.indices} for time {self.time}"
    
# n is needed for PBC where *non-local* CUs are allowed across the boundary
# indices = (targets, control)
# note target indices HAVE to be in ascending order and nearest neighbors
# applies subgate when control wire is at 1 and applies identity otherwise

class CUGate(Gate):
    def __init__(self, indices, subgate, n):
        d = subgate.get_local_dim()
        submat = subgate.matrix
        s = submat.shape[0]
        indices = np.array([int(i) for i in indices])
        r = np.min(np.abs(indices[:-1] - indices[-1]))
        
        if r > 1: # 'long'-range gate across boundary
            raise NotImplementedError("Long range CUGate has not been implemented yet")
            
        else: # nearest neighbor gate 
            mat_list = [np.eye(s), submat] + [np.eye(s)] * (d-2)
            mat = block_diag(*mat_list)
            
            if indices[-2] < indices[-1]: 
                indices = indices
                mat = mat.reshape(d,s,d,s).transpose(1,0,3,2).reshape(d*s,d*s)
                bottom_heavy = False
            elif indices[-1] < indices[0]: 
                indices = np.concatenate(([indices[-1]], indices[0:-1]))
                bottom_heavy = True
            else: 
                raise ValueError("indices does not have allowed ordering")
        
        name = f"CU{d}"
        super().__init__(name, indices, mat)
        self.bottom_heavy = bottom_heavy
        
    def __str__(self):
        title = "bottom heavy" if self.bottom_heavy else "top heavy"
        return f"{title} {self.name} gate on site(s) {self.indices}"
    
# n is needed for PBC where *non-local* CNOTs are allowed across the boundary
# indices = (target, control)
class CNOTGate(Gate):
    def __init__(self, indices, matrix, n):
        d = int(np.sqrt(matrix.shape[0]))
        indices = [int(i) for i in indices]
        
        if np.abs(indices[0] - indices[1]) > 1: # 'long'-range gate across boundary
            temp = np.array(indices) + 1
            if temp[0]%n < temp[1]%n: 
                indices = indices
                mat = matrix
                bottom_heavy = False
            else: 
                indices = indices[::-1]
                mat = matrix.reshape(d,d,d,d).transpose(1,0,3,2).reshape(d*d,d*d)
                bottom_heavy = True
        else: # nearest neighbor gate 
            if indices[0] < indices[1]: 
                indices = indices
                mat = matrix
                bottom_heavy = False
            else: 
                indices = indices[::-1]
                mat = matrix.reshape(d,d,d,d).transpose(1,0,3,2).reshape(d*d,d*d)
                bottom_heavy = True
        
        name = f"CNOT{d}"
        super().__init__(name, indices, mat)
        self.bottom_heavy = bottom_heavy
        
    def __str__(self):
        title = "bottom heavy" if self.bottom_heavy else "top heavy"
        return f"{title} {self.name} gate on site(s) {self.indices}"
    
class ParamGate(Gate):
    def __init__(self, name, indices, *args): 
        super().__init__(name, indices, None) 
        
        if type(args[0]) == np.ndarray: 
            mat = args[0]
            self.update_matrix(mat)
            self.angles = self.solve_angles()
        
        else:
            self.angles = [args] if type(args) == float else args
            mat = self.construct_matrix()
            self.update_matrix(mat)
            
    def __str__(self):
        return f"{self.name} gate on site(s) {self.indices} with angle(s) {self.angles}"

    def update_matrix(self, mat): 
        self.matrix = mat
        
    def construct_matrix(self): 
        raise NotImplementedError("Subclasses must implement this method")
        
    def solve_angles(self): 
        return "unsolved" # if subclass doesn't bother implementing we probably don't need angles
    
########################################
############ QUBIT GATES ###############
########################################

def SX(indices): return Gate("SX", 
                              indices,
                              (1/np.sqrt(2)) * np.array([[1,-1j],[-1j,1]]))

def X(indices): return Gate("X", 
                            indices, 
                            np.array([[0,1],[1,0]]))

def H(indices): return Gate("H", 
                            indices, 
                            np.array([[1,1],[1,-1]])/np.sqrt(2))

def SDG(indices): return Gate("SDG", 
                              indices, 
                              np.array([[1,0],[0,-1j]]))

def CNOT(indices, n): return CNOTGate(indices, 
                                      np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]]), 
                                      n)  

def SWAP(indices): return Gate("SWAP", indices, 
                                np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]))

class RZ(ParamGate): 
    def __init__(self, indices, *args):
        super().__init__("RZ", indices, *args)
    
    def construct_matrix(self): 
        return np.array([[1,0],[0,np.exp(1.j*self.angles[0])]])

class U(ParamGate):
    def __init__(self, indices, *args):
        super().__init__("U", indices, *args)
    
    def construct_matrix(self): 
        t,p,l = self.angles
        return np.array([[np.cos(t/2), -np.exp(1j*l)*np.sin(t/2)],
                           [np.exp(1j*p)*np.sin(t/2), np.exp(1j*(p+l))*np.cos(t/2)]])
    
    def solve_angles(self): 
        phase = np.angle(self.matrix[0,0]) 
        self.matrix = self.matrix / np.exp(1.j*phase) # removes phase for top left entry
        t = np.arccos(self.matrix[0,0])*2
        if np.abs(np.sin(t/2)) > 1e-12:
            l = np.angle(-self.matrix[0,1]/np.sin(t/2))
            p = np.angle(self.matrix[1,1]/self.matrix[0,0])-l
        else: 
            l_plus_p = np.angle(self.matrix[1,1])
            l,p = l_plus_p/2, l_plus_p/2
        
        return t,p,l        
        
    def decompose(self, basis='ZSX'): 
        if basis == 'ZSX': 
            t,p,l = self.solve_angles()#self.angles
            i = self.indices
            
            return list(reversed([
                RZ(i,p+np.pi),
                SX(i),
                RZ(i,t+np.pi),
                SX(i), 
                RZ(i,l)
            ]))   # If A = BC, we must apply C first and then B
            
        else: 
            raise NotImplementedError(f"{basis} basis decomposition has not been implemented")

#########################################
############# QUTRIT GATES ##############
#########################################

def X01(indices): return Gate("X01", 
                              indices, 
                              np.array([[0,-1j,0],[-1j,0,0],[0,0,1]]))

def SX01(indices): return Gate("SX01", 
                               indices, 
                               (1/np.sqrt(2)) * np.array([[1,-1j,0],[-1j,1,0],[0,0,np.sqrt(2)]]))

def X12(indices): return Gate("X12", 
                              indices, 
                              np.array([[1,0,0],[0,0,-1j],[0,-1j,0]]))

def SX12(indices): return Gate("SX12", 
                               indices, 
                               (1/np.sqrt(2)) * np.array([[np.sqrt(2),0,0],[0,1,-1j],[0,-1j,1]]))

def H01(indices): return Gate("H01", 
                              indices, 
                              np.array([[1/np.sqrt(2),1/np.sqrt(2),0],[1/np.sqrt(2),-1/np.sqrt(2),0],[0,0,1]]))

def SDG01(indices): return Gate("SDG01", 
                              indices, 
                              np.array([[1,0,0],[0,-1j,0],[0,0,1]]))

def CNOT3(indices, n): return CNOTGate(indices, 
                                      np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                       [0., 0., 0., 0., 1, 0., 0., 0., 0.],
                                       [0., 0., 1/np.sqrt(2), 0., 0., 1/np.sqrt(2), 0., 0., 0.],
                                       [0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                       [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                       [0., 0.,  -1/np.sqrt(2), 0., 0., 1/np.sqrt(2), 0., 0., 0.],
                                       [0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                       [0., 0., 0., 0., 0., 0., 0., 1j, 0.],
                                       [0., 0., 0., 0., 0., 0., 0., 0., 1.]]).T, 
                                      n)   

def SWAP3(indices): return Gate("SWAP3", indices, 
                                np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                       [0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                       [0., 0., 0, 0., 0., 0, 1., 0., 0.],
                                       [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                       [0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                       [0., 0.,  0, 0., 0., 0, 0., 1., 0.],
                                       [0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                       [0., 0., 0., 0., 0., 1., 0., 0, 0.],
                                       [0., 0., 0., 0., 0., 0., 0., 0., 1.]]))

class RZ01(ParamGate): 
    def __init__(self, indices, *args):
        super().__init__("RZ01", indices, *args)
    
    def construct_matrix(self): 
        return np.array([[1,0,0],[0,np.exp(1.j*self.angles[0]),0],[0,0,1]])
    
class RZ12(ParamGate): 
    def __init__(self, indices, *args):
        super().__init__("RZ12", indices, *args)
    
    def construct_matrix(self): 
        return np.array([[1,0,0],[0,1,0],[0,0,np.exp(1.j*self.angles[0])]])

class U01(ParamGate):
    def __init__(self, indices, *args):
        super().__init__("U01", indices, *args)
    
    def construct_matrix(self): 
        t,p,l = self.angles
        return np.array([[np.cos(t/2), -np.exp(1j*l)*np.sin(t/2), 0],
                           [np.exp(1j*p)*np.sin(t/2), np.exp(1j*(p+l))*np.cos(t/2), 0],
                           [0, 0, 1]])
    
    def solve_angles(self): 
        return None # TODO implement solver 
    
    def decompose(self, basis='ZSX'): 
        if basis == 'ZSX': 
            t,p,l = self.angles
            i = self.indices
            return list(reversed([
                RZ01(i,p+np.pi),
                SX01(i),
                RZ01(i,t+np.pi),
                SX01(i), 
                RZ01(i,l)
            ]))
        else: 
            raise NotImplementedError(f"{basis} basis decomposition has not been implemented")