
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import pandas as pd
from utils import save_results_to_csv, visualize_results
# -----------------------
# Sampled Parameters with Increased Dispersion
# -----------------------
np.random.seed(33)

# Sampled using normal distributions
delta_P = np.random.normal(loc=300, scale=100)         # Pressure drop (Pa)
L = np.random.normal(loc=0.02, scale=0.01)             # Length of artery (m)
true_r = np.random.normal(loc=0.0002, scale=0.0001)    # Radius (m)
true_mu = np.random.normal(loc=0.0035, scale=0.002)    # Dynamic viscosity (Pa·s)

# -----------------------
# Core Functions
# -----------------------

def poiseuille_flow_vec(r, mu):
    """Vectorized Poiseuille's equation"""
    return (np.pi * np.power(r, 4) * delta_P) / (8 * mu * L)

def simulate_noisy_observations(Q_true, num_samples=30, noise_percent=0.2):
    """Generate noisy observations"""
    return np.random.normal(loc=Q_true, scale=Q_true * noise_percent, size=num_samples)

def build_and_run_model(Q_obs, Q_true):
    """Bayesian model using PyMC and Numpyro backend for speed"""
    with pm.Model() as model:
        r = pm.HalfNormal("r", sigma=0.0005)
        mu = pm.Normal("mu", mu=0.0035, sigma=0.003)
        Q_est = (np.pi * r**4 * delta_P) / (8 * mu * L)

        pm.Normal("Q_obs", mu=Q_est, sigma=Q_true * 0.2, observed=Q_obs)

        trace = pm.sample(
            draws=1000,
            tune=500,
            chains=4,
            cores=4,
            init="adapt_diag",
            target_accept=0.95,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": False},
            random_seed=42,
            sampler_kwargs={"nuts_sampler": "numpyro"}  # <-- Faster backend
        )
    return trace

def analyze_posteriors(trace, Q_true):
    """Analyze posterior samples and detect abnormalities"""
    posterior_r = trace.posterior["r"].stack(samples=("chain", "draw")).values
    posterior_mu = trace.posterior["mu"].stack(samples=("chain", "draw")).values
    posterior_Q = poiseuille_flow_vec(posterior_r, posterior_mu)

    normal_lower = Q_true * 0.9
    normal_upper = Q_true * 1.1

    alerts = np.where(
        posterior_Q < normal_lower * 0.3,
        "Blockage",
        np.where(
            posterior_Q < normal_lower,
            "Vasoconstriction",
            np.where(
                posterior_Q > normal_upper,
                "Vasodilation",
                "Normal"
            )
        )
    )
    return posterior_r, posterior_mu, posterior_Q, alerts




if __name__ == "__main__":
    folder_path = "blood_flow_analysis_results"
    Q_true = poiseuille_flow_vec(true_r, true_mu)
    Q_obs = simulate_noisy_observations(Q_true)

    trace = build_and_run_model(Q_obs, Q_true)
    posterior_r, posterior_mu, posterior_Q, alerts = analyze_posteriors(trace, Q_true)

    save_results_to_csv(posterior_r, posterior_mu, posterior_Q, alerts, folder_path)
    #visualize_results(posterior_r, posterior_mu, posterior_Q, Q_true, alerts, folder_path)
    visualize_results(posterior_r, posterior_mu, posterior_Q, Q_true, true_r, true_mu, alerts, folder_path)

