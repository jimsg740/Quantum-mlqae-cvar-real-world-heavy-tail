# Quantum CVaR Estimation for Catastrophe Insurance Pricing

A Quantum Amplitude Estimation (QAE) framework for computing Conditional Value-at-Risk (CVaR) on real-world heavy-tailed catastrophe loss data, demonstrating quadratic speedup over classical Monte Carlo methods.

## Highlights

- **Real data, not synthetic**: Built on 110,000+ FEMA Public Assistance loss records (2019–2023)
- **Spliced Lognormal-GPD model**: Threshold optimized via conditional Anderson-Darling distance
- **Log-spaced quantum binning**: Minimizes discretization error in the heavy tail
- **MLQAE architecture**: QPE-free amplitude estimation with shallow circuits
- **Empirical quadratic speedup**: $O(N^{-1})$ vs classical $O(N^{-1/2})$, validated across multiple VaR thresholds

## Project Structure

```
├── data/
│   └── fema_sample_data.csv          # Simplified FEMA dataset (5 key columns)
├── notebooks/
│   ├── 01_Threshold_Optimization.ipynb       # A-D statistic threshold search
│   ├── 02_Distribution_Comparison.ipynb      # Spliced vs single-model benchmarking
│   ├── 03_Quantum_Grid_Log_Binning.ipynb     # Log vs linear discretization proof
│   └── 04_MLQAE_Convergence_Benchmarking.ipynb  # QAE vs Monte Carlo convergence
├── src/
│   ├── distribution_fit.py           # EVT fitting & spliced CDF
│   ├── discretization.py             # Log-spaced grid generation
│   └── qae_circuits.py               # Quantum circuits & ML optimizer
├── requirements.txt
└── README.md
```

## Quick Start

```bash
git clone https://github.com/<your-username>/quantum-cvar-catastrophe-pricing.git
cd quantum-cvar-catastrophe-pricing
pip install -r requirements.txt
```

Then open `notebooks/01_Threshold_Optimization.ipynb` and run sequentially through all four notebooks.

## Methodology

**1. Threshold Optimization** — The optimal Peak-over-Threshold for the GPD tail is found by minimizing the conditional Anderson-Darling statistic across a quantile grid (90%–98%).

**2. Spliced Distribution** — A Lognormal body and GPD tail are joined at the optimized threshold, with a probability anchor $W$ ensuring CDF continuity.

**3. CVaR Decomposition** — CVaR is split into a classically computed VaR threshold plus a quantum-estimated Expected Excess Loss (EEL), following Rockafellar-Uryasev.

**4. Quantum Encoding** — The spliced distribution is discretized onto a $2^n$-state log-spaced grid and loaded via tree-based controlled-RY rotations. A pricing oracle encodes the payoff function into an ancilla qubit.

**5. MLQAE Convergence** — Multiple circuits with increasing Grover depth are executed; maximum likelihood estimation recovers the target amplitude without QPE overhead.

## Data Source

The dataset is derived from the [FEMA Public Assistance Funded Projects](https://www.fema.gov/openfema-data-page/public-assistance-funded-projects-details-v1) open data portal, filtered to natural disaster declarations from 2019–2023.

## Requirements

- Python ≥ 3.10
- Qiskit ≥ 1.0
- qiskit-aer ≥ 0.13
- NumPy, SciPy, Pandas, Matplotlib

## License

MIT
