# Stochastic-Maintenance-Optimisation
Poisson-process damage modelling with Monte Carlo optimisation of inspection intervals.

This project models damage arrivals on a highway bridge as a **Poisson process**, estimates the arrival rate via **maximum likelihood**, and optimises the **inspection interval** to minimise expected cost using **Monte Carlo simulation**.

## Method (Quant Summary)

1) **Damage arrival model (count data)**
- Monthly damage counts \(k_t\) with exposure \(T_t\) (days/month)
- Assume \(k_t \sim \text{Poisson}(\lambda T_t)\)
- Estimate \(\lambda\) via **MLE** on log-likelihood

2) **Severity model**
- Empirical severity distribution over discrete levels (minor / moderate / major)
- Used as categorical distribution for simulation

3) **Cost functional**
Expected cost per inspection cycle includes:
- inspection cost
- repair cost (severity-dependent)
- penalty if total severity score exceeds a threshold

4) **Policy optimisation**
- Evaluate inspection interval \(\tau\) (months)
- Estimate \( \mathbb{E}[C(\tau)] / \tau \) via Monte Carlo
- Choose \(\tau^\*\) that minimises expected monthly cost

## Outputs
- MLE estimate of damage arrival rate \(\hat{\lambda}\)
- Estimated severity probabilities
- Estimated expected monthly cost under baseline policy
- Optimal inspection interval \(\tau^\*\) (grid search)

## Data
The repository includes anonymised monthly damage count data used for Poisson rate estimation and policy optimisation.


## How to run
```bash
pip install -r requirements.txt
python src/highway_maintenance_optim.py

