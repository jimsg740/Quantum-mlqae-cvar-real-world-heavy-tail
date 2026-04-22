import numpy as np

def get_log_spaced_grid(cdf_func, x_min, x_max, n_qubits):
    """Implements the log-spaced binning to capture heavy-tail topology."""
    num_bins = 2 ** n_qubits
    log_edges = np.linspace(np.log(x_min), np.log(x_max), num_bins + 1)
    edges = np.exp(log_edges)
    
    # Midpoints for payoff calculation
    x_i = np.sqrt(edges[:-1] * edges[1:])
    
    # Probability masses
    p_raw = cdf_func(edges[1:]) - cdf_func(edges[:-1])
    p_normalized = p_raw / np.sum(p_raw)
    
    return x_i, p_normalized