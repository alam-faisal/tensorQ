import numpy as np
from functools import reduce

class ThermalNoise: 
    def __init__(self, t1, t2, one_site_time, two_site_time, local_dim, coupling=1.0):
        self.t1 = t1
        self.t2 = t2
        self.one_site_time = one_site_time
        self.two_site_time = two_site_time
        self.local_dim = local_dim
        self.coupling = coupling
        
    def get_kraus(self, time): 
        raise NotImplementedError("Subclasses must implement this method")
        
    def get_superop(self, time): 
        kraus_list = self.get_kraus(time)       
        return sum([np.kron(k.conj(), k) for k in kraus_list])
    
class QubitNoise(ThermalNoise): 
    def __init__(self, t1, t2, one_qubit_time, two_qubit_time, coupling=1.0):
        super().__init__(t1, t2, one_qubit_time, two_qubit_time, 2, coupling)
        
    def get_kraus(self, time): 
        t1, t2 = self.t1, self.t2
        p1 = 1-np.exp(-time/t1)
        p2 = 1-np.exp(-time/t2)
        p1c = 1-p1
        p2c = 1-p2
        
        kraus_list = [np.zeros((2,2)) for _ in range(4)]
        
        kraus_list[0][0,0] = np.sqrt(p2)
        kraus_list[0][1,1] = -np.sqrt(p2*p1c)
        
        kraus_list[1][0,1] = -np.sqrt(p1*p2)
        
        kraus_list[2][0,0] = np.sqrt(p2c)
        kraus_list[2][1,1] = np.sqrt(p1c*p2c)
        
        kraus_list[3][0,1] = np.sqrt(p1*p2c)
        
        return kraus_list
        
        
class QutritNoise(ThermalNoise): 
    def __init__(self, t1, t2, one_qutrit_time, two_qutrit_time, coupling=1.0):
        super().__init__(t1, t2, one_qutrit_time, two_qutrit_time, 3, coupling)
    
    def get_kraus(self, time): 
        t1, t2 = self.t1, self.t2
        tr = t1*t2/(t1+t2)
        p1 = 1-np.exp(-time/t1)
        p2 = 1-np.exp(-time/t2)
        pr = 1-np.exp(-time/tr)
        p1c = 1-p1
        p2c = 1-p2
        prc = 1-pr

        kraus_list = [np.zeros((3,3)) for _ in range(9)]

        kraus_list[0][0,0] = np.sqrt(p2c)
        kraus_list[0][1,1] = np.sqrt(prc)
        kraus_list[0][2,2] = np.sqrt(prc)

        kraus_list[1][0,1] = np.sqrt(p2c*p1)
        kraus_list[2][1,2] = np.sqrt(p2c*p1)

        kraus_list[3][0,0] = np.sqrt(p2) * (1/np.sqrt(2))
        kraus_list[3][1,1] = -np.sqrt(p1c*p2) * (1/np.sqrt(2))
        kraus_list[3][2,2] = np.sqrt(p1c*p2) * (1/np.sqrt(2))

        kraus_list[4][0,1] = np.sqrt(p1*p2) * (1/np.sqrt(2))
        kraus_list[5][1,2] = -np.sqrt(p1*p2) * (1/np.sqrt(2))

        kraus_list[6][0,0] = np.sqrt(p2) * (1/np.sqrt(2))
        kraus_list[6][1,1] = np.sqrt(p1c*p2) * (1/np.sqrt(2))
        kraus_list[6][2,2] = -np.sqrt(p1c*p2) * (1/np.sqrt(2))

        kraus_list[7][0,1] = np.sqrt(p1*p2) * (1/np.sqrt(2))
        kraus_list[8][1,2] = np.sqrt(p1*p2) * (1/np.sqrt(2))
        return kraus_list        