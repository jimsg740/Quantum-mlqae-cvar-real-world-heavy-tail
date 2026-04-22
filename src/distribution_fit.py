"""
Extreme Value Theory (EVT) and Distribution Fitting Module.
Implements the Conditional A-D distance optimization for threshold selection
and the Spliced Lognormal-GPD model for heavy-tailed risk data.
"""

import numpy as np
from scipy.stats import lognorm, genpareto

def optimize_gpd_threshold_ad(data: np.ndarray, q_start=0.90, q_end=0.98, step=0.005):
    """
    Finds the optimal Peak-Over-Threshold (POT) using the Conditional Tail Anderson-Darling Statistic.
    
    Args:
        data: Array of raw empirical losses.
        q_start, q_end: The quantile search grid boundaries.
        step: Search grid step size.
        
    Returns:
        best_u: The absolute threshold value minimizing the A-D statistic.
        best_q: The corresponding quantile.
        quantiles: Array of quantiles tested.
        ad_scores: Array of A-D statistics for each tested quantile.
    """
    quantiles = np.arange(q_start, q_end + step, step)
    ad_scores = []
    
    best_ad = np.inf
    best_u = None
    best_q = None

    for q in quantiles:
        u = np.percentile(data, q * 100)
        excesses = data[data > u] - u
        N_u = len(excesses)
        
        # If too few samples in the tail, skip to avoid extreme variance
        if N_u < 5: 
            ad_scores.append(np.inf)
            continue

        # Fit GPD on current excesses
        c_gpd, _, scale_gpd = genpareto.fit(excesses, floc=0)
        
        # Calculate Conditional A-D Statistic
        Y_sorted = np.sort(excesses)
        G_Y = genpareto.cdf(Y_sorted, c=c_gpd, scale=scale_gpd)
        
        # Clip to prevent log(0) issues in extreme cases
        G_Y = np.clip(G_Y, 1e-10, 1.0 - 1e-10)
        
        i = np.arange(1, N_u + 1)
        term = (2 * i - 1) / N_u * (np.log(G_Y) + np.log(1 - G_Y[::-1]))
        ad_stat = -N_u - np.sum(term)
        
        ad_scores.append(ad_stat)
        
        # Update optimal threshold
        if ad_stat < best_ad:
            best_ad = ad_stat
            best_u = u
            best_q = q

    return best_u, best_q, quantiles, np.array(ad_scores)

def fit_spliced_distribution(data, threshold):
    """
    Fits Lognormal body and GPD tail using an explicit threshold u.
    
    Args:
        data: Empirical loss data.
        threshold: The absolute threshold u (e.g., computed from optimize_gpd_threshold_ad).
    """
    body_data = data[data <= threshold]
    tail_data = data[data > threshold]
    
    w = len(body_data) / len(data)
    
    # Fit Body (Lognormal)
    shape_ln, loc_ln, scale_ln = lognorm.fit(body_data, floc=0)
    
    # Fit Tail (GPD) on excesses
    excesses = tail_data - threshold
    c_gpd, loc_gpd, scale_gpd = genpareto.fit(excesses, floc=0)
    
    return w, threshold, (shape_ln, scale_ln), (c_gpd, scale_gpd)

def spliced_cdf(x, w, threshold, ln_params, gpd_params):
    """Custom Spliced Cumulative Distribution Function (CDF)."""
    shape_ln, scale_ln = ln_params
    c_gpd, scale_gpd = gpd_params
    
    x = np.asarray(x)
    cdf_values = np.zeros_like(x, dtype=float)
    
    norm_factor = lognorm.cdf(threshold, s=shape_ln, scale=scale_ln)
    
    mask_body = x <= threshold
    if np.any(mask_body):
        cdf_values[mask_body] = w * (lognorm.cdf(x[mask_body], s=shape_ln, scale=scale_ln) / norm_factor)
        
    mask_tail = x > threshold
    if np.any(mask_tail):
        cdf_values[mask_tail] = w + (1 - w) * genpareto.cdf(x[mask_tail] - threshold, c=c_gpd, scale=scale_gpd)
        
    return cdf_values