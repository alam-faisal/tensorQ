from .gates import *

class Circuit:
    def __init__(self, gate_list, n=None):
        self.gate_list = gate_list
        sites = np.concatenate([gate.indices for gate in gate_list])
        self.num_sites = max(sites) + 1 if n is None else n
        self.local_dim = gate_list[0].get_local_dim()
        self.layers = None
        self.built = False
        self.meas_qubits = None
        
    def update_gate(self, gate_idx, new_array): 
        self.gate_list[gate_idx].matrix = new_array
        if self.layers is not None:
            self.construct_layers()
        
    def decompose(self, basis='ZSX'): 
        gate_list = []
        for gate in self.gate_list:
            if gate.name == "U" and gate.angles is not None:
                decomposed_gates = gate.decompose(basis)
                gate_list.extend(decomposed_gates)
            else:
                gate_list.append(gate)

        self.gate_list = gate_list
        
    def construct_layers(self): 
        layers = []
        for gate in self.gate_list:
            place_available = True
            layer_ind = len(layers) - 1

            # first we find the oldest layer that fails to commute with current gate
            while place_available and layer_ind >= 0:
                g_indices = set(gate.indices)
                existing_indices = {ind for gate in layers[layer_ind] for ind in gate.indices}
                if g_indices.intersection(existing_indices): 
                    place_available = False
                else: 
                    layer_ind -= 1

            # then we add current gate to the following layer
            if layer_ind < len(layers) - 1: 
                layers[layer_ind + 1].append(gate)
            else: 
                layers.append([gate])

        self.layers = layers
    
    def add_noise(self, nm): 
        layers = []
        for layer in self.layers: 
            if not any(isinstance(obj, SuperOp) for obj in layer):
                layer_time = max([gate_time(gate, nm) for gate in layer])
                gate_times = [0.0 for i in range(self.num_sites)]
                for i in range(self.num_sites): 
                    for gate in layer: 
                        if i in gate.indices:
                            gate_times[i] = gate_time(gate, nm)

                noise_layer = [ID(nm.local_dim, max(nm.coupling*layer_time, gate_times[i]), i) 
                               for i in range(self.num_sites)]
                layer.extend(noise_layer)
                layers.append(layer)
            else: 
                layers.append(layer)
        
        self.layers = layers
        
    def build(self, nm=None): 
        """ note that this returns a new circuit instance built with provided noise model
        and leaves the current circuit instance unchanged """
        new_circ = copy.deepcopy(self)
        new_circ.decompose()
        new_circ.construct_layers()
        if nm is not None: 
            new_circ.add_noise(nm)
        
        new_circ.built = True    
        return new_circ
    
    def add_meas(self, meas_qubits): 
        self.meas_qubits = meas_qubits
        
    def dag(self): 
        gate_list = [g.dag() for g in self.gate_list[::-1]]
        return Circuit(gate_list, self.num_sites)
        
    def to_matrix(self): 
        return circ_to_mat(self)
        
    def __str__(self): 
        if self.layers is not None: 
            full_str = []
            for layer in self.layers:
                layer_str = " \n".join([str(gate) for gate in layer])
                full_str.append(layer_str)
            full_str = " \n\n".join(full_str)
            return full_str
        
        else: 
            return " \n".join([str(gate) for gate in self.gate_list])  
        
    def draw(self): # TODO 
        """ use svgwrite to visualize the circuit """
        return None
            
def gate_time(gate, nm):     
    if gate.name[0:4] == "CNOT": 
        return nm.two_site_time
    elif gate.name[0:2] == "RZ":
        return 0.0
    else: 
        return nm.one_site_time
    
def circ_to_mat(circ, n=None): 
    if circ.layers == None: 
        circ.construct_layers()
    n = circ.num_sites if n is None else n
    
    layer_mats = []    
    for layer in circ.layers: 
        missing_indices = set(range(n)) - set(np.concatenate([gate.indices for gate in layer]))
        for index in missing_indices: 
            layer.append(ID(circ.local_dim, 0.0, [index]))
        
        sorted_layer = sorted(layer, key=lambda gate: gate.indices[0])
        mat_list = [gate.matrix for gate in sorted_layer]
        layer_mats.append(reduce(np.kron, mat_list))
        
    return np.linalg.multi_dot(layer_mats[::-1]) if len(layer_mats) > 1 else layer_mats[0]

def shift_gate_list(circ, start_site): 
    """ return gate_list of circ with all indices shifted by start_site """
    gate_list = copy.deepcopy(circ.gate_list)
    for gate in gate_list: 
        gate.indices = np.array(gate.indices) + start_site

    return gate_list