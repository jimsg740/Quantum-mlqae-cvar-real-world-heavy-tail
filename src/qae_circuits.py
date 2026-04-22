import numpy as np
import scipy.optimize as opt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RYGate

def get_marginal_prob(probs, prefix, n):
    if not prefix: return np.sum(probs)
    k = len(prefix)
    start_idx = int(prefix, 2) << (n - k)
    end_idx = start_idx + (1 << (n - k))
    return np.sum(probs[start_idx:end_idx])

def build_custom_state_prep(probs, n):
    qc = QuantumCircuit(n, name="Tree_Prep")
    for k in range(n):
        target_qubit = n - 1 - k
        control_qubits = [n - 1 - i for i in range(k)]
        for i in range(2**k):
            prefix = format(i, f'0{k}b') if k > 0 else ""
            p_prefix = get_marginal_prob(probs, prefix, n)
            p_prefix_0 = get_marginal_prob(probs, prefix + '0', n)
            
            if p_prefix > 0:
                theta = 2 * np.arccos(np.sqrt(np.clip(p_prefix_0 / p_prefix, 0.0, 1.0)))
            else:
                theta = 0.0
                
            if theta > 1e-10:
                zero_indices = [n - 1 - j for j, bit in enumerate(prefix) if bit == '0']
                if zero_indices: qc.x(zero_indices)
                if k == 0:
                    qc.ry(theta, target_qubit)
                else:
                    qc.append(RYGate(theta).control(len(control_qubits)), control_qubits + [target_qubit])
                if zero_indices: qc.x(zero_indices)
    return qc

def build_pricing_oracle(payoffs, max_payoff, n):
    qr_state = QuantumRegister(n, 'state')
    qr_obj = QuantumRegister(1, 'obj')
    qc = QuantumCircuit(qr_state, qr_obj, name="Pricing_Oracle")
    
    for i in range(2**n):
        if payoffs[i] > 0:
            theta_i = 2 * np.arcsin(np.sqrt(payoffs[i] / max_payoff))
            bin_str = format(i, f'0{n}b')
            zero_indices = [n - 1 - j for j, bit in enumerate(bin_str) if bit == '0']
            
            if zero_indices: qc.x(zero_indices)
            qc.append(RYGate(theta_i).control(n), qr_state[:] + [qr_obj[0]])
            if zero_indices: qc.x(zero_indices)
            
    return qc

def build_qae_circuit(n, probs, payoffs, max_payoff, k):
    """
    Assemble the full QAE circuit with State Prep (A), Oracle, and Grover iterations.
    """
    qr_state = QuantumRegister(n, 'state')
    qr_obj = QuantumRegister(1, 'obj')
    cr = ClassicalRegister(1, 'meas')
    
    qc_A = QuantumCircuit(qr_state, qr_obj, name='A')
    qc_A.append(build_custom_state_prep(probs, n).to_gate(), qr_state)
    qc_A.append(build_pricing_oracle(payoffs, max_payoff, n).to_gate(), qr_state[:] + qr_obj[:])
    gate_A = qc_A.to_gate()
    gate_A_dg = qc_A.inverse().to_gate(label='A_dg')
    
    qc_S_chi = QuantumCircuit(qr_state, qr_obj, name='S_chi')
    qc_S_chi.z(qr_obj[0])
    gate_S_chi = qc_S_chi.to_gate()
    
    qc_S_0 = QuantumCircuit(qr_state, qr_obj, name='S_0')
    qc_S_0.x(qr_state); qc_S_0.x(qr_obj)
    qc_S_0.h(qr_obj[0])
    qc_S_0.mcx(qr_state, qr_obj[0])
    qc_S_0.h(qr_obj[0])
    qc_S_0.x(qr_state); qc_S_0.x(qr_obj)
    gate_S_0 = qc_S_0.to_gate()
    
    main_qc = QuantumCircuit(qr_state, qr_obj, cr)
    main_qc.append(gate_A, qr_state[:] + qr_obj[:])
    
    for _ in range(k):
        main_qc.append(gate_S_chi, qr_state[:] + qr_obj[:])
        main_qc.append(gate_A_dg, qr_state[:] + qr_obj[:])
        main_qc.append(gate_S_0, qr_state[:] + qr_obj[:])
        main_qc.append(gate_A, qr_state[:] + qr_obj[:])
        
    main_qc.measure(qr_obj[0], cr[0])
    return main_qc

class MLQAEOptimizer:
    """
    Maximum Likelihood estimation logic without deep QPE.
    """
    def __init__(self, k_list, hits_list, shots):
        self.k_list = k_list
        self.hits_list = hits_list
        self.shots = shots

    def negative_log_likelihood(self, theta):
        ll = 0.0
        for k, hits in zip(self.k_list, self.hits_list):
            p_1 = np.sin((2 * k + 1) * theta)**2
            p_1 = np.clip(p_1, 1e-10, 1.0 - 1e-10)
            ll += hits * np.log(p_1) + (self.shots - hits) * np.log(1 - p_1)
        return -ll

    def run_optimization(self):
        # 1. Grid search for stable initial point
        theta_grid = np.linspace(0, np.pi/4, 1000)
        ll_values = [self.negative_log_likelihood(t) for t in theta_grid]
        best_initial_theta = theta_grid[np.argmin(ll_values)]
        
        # 2. Local L-BFGS-B optimization
        result = opt.minimize(
            self.negative_log_likelihood, 
            x0=best_initial_theta, 
            bounds=[(0.0, np.pi/2)],
            method='L-BFGS-B'
        )
        return result.x[0]