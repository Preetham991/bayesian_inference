
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import math

# -----------------------
# Sampled Parameters
# -----------------------
np.random.seed(42)

# Default physical parameters
delta_P = np.random.normal(loc=300, scale=300)         # Pressure drop (Pa)
L = np.random.normal(loc=0.02, scale=0.03)             # Length of artery (m)
true_r = np.random.normal(loc=0.0002, scale=0.0003)    # Radius (m)
true_mu = np.random.normal(loc=0.0035, scale=0.005)    # Viscosity (Pa·s)

# -----------------------
# Core Functions
# -----------------------

def poiseuille_flow_vec(r, mu):
    """Vectorized Poiseuille's equation"""
    return (np.pi * np.power(r, 4) * delta_P) / (8 * mu * L)

def simulate_noisy_observations(Q_true, num_samples=30, noise_percent=1.0):
    """Generate noisy observations with validation"""
    if Q_true <= 0 or noise_percent < 0:
        raise ValueError(f"Invalid parameters: Q_true={Q_true}, noise_percent={noise_percent}")
    
    return np.random.normal(loc=Q_true, scale=Q_true * noise_percent, size=num_samples)

def validate_physical_parameters(r, mu):
    """Validate that radius and viscosity are within realistic bounds"""
    if r <= 0 or mu <= 0:
        raise ValueError(f"Invalid physical parameters: radius={r}, viscosity={mu}")
    
    # Check that the radius and viscosity are not too large
    if r > 0.1 or mu > 1000:
        raise ValueError(f"Unrealistic physical parameters: radius={r}, viscosity={mu}")
    
    return True

def build_and_run_model(Q_obs, Q_true):
    """Bayesian model using PyMC with high dispersion priors and error handling"""
    try:
        with pm.Model() as model:
            r = pm.HalfNormal("r", sigma=0.0035)            # Wide HalfNormal → allows near-zero
            mu = pm.Normal("mu", mu=0.0035, sigma=0.01)     # Wide Normal → high resistance

            Q_est = (np.pi * r**4 * delta_P) / (8 * mu * L)

            pm.Normal("Q_obs", mu=Q_est, sigma=Q_true * 1.0, observed=Q_obs)

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
                sampler_kwargs={"nuts_sampler": "numpyro"}  # Faster backend
            )
        return trace
    except Exception as e:
        print(f"Error during model building and sampling: {e}")
        return None

def analyze_posteriors(trace, Q_true):
    """Analyze posterior samples and detect abnormalities"""
    if trace is None:
        raise ValueError("Trace is None. The model did not run successfully.")
    
    posterior_r = trace.posterior["r"].stack(samples=("chain", "draw")).values
    posterior_mu = trace.posterior["mu"].stack(samples=("chain", "draw")).values
    posterior_Q = poiseuille_flow_vec(posterior_r, posterior_mu)

    normal_lower = Q_true * 0.9
    normal_upper = Q_true * 1.1

    # Alert classification based on posterior flow rate (Q)
    alerts = np.where(
        posterior_Q < Q_true * 0.05,  # Extreme Blockage
        "Extreme Blockage",
        np.where(
            posterior_Q < normal_lower * 0.7,  # Blockage
            "Blockage",
            np.where(
                posterior_Q < normal_lower,  # Reduced Flow
                "Reduced Flow",
                np.where(
                    posterior_Q > normal_upper,  # Near Normal Flow
                    "Near Normal Flow",
                    "Near Normal Flow"
                )
            )
        )
    )
    return posterior_r, posterior_mu, posterior_Q, alerts

def handle_outliers(data, threshold=0.1):
    """Handle extreme outliers by setting them to a threshold"""
    data = np.clip(data, None, threshold)  # Set upper limit
    return data

def save_results_to_folder(posterior_r, posterior_mu, posterior_Q, alerts, folder_name="extremeconditions_results"):
    """Save plots and alerts to a folder"""
    # Create the directory if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Plotting and saving results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot Posterior for Radius
    axes[0].hist(posterior_r, bins=30, density=True, color='skyblue', alpha=0.7)
    axes[0].set_title("Posterior of Radius (r)")
    axes[0].set_xlabel("Radius (m)")
    axes[0].set_ylabel("Density")

    # Plotting mu ± n * sigma for Radius
    mu_r = np.mean(posterior_r)
    sigma_r = np.std(posterior_r)
    
    for i in range(1, 4):
        axes[0].axvline(mu_r + i * sigma_r, color='red', linestyle='--', label=f'mu + {i} sigma')
        axes[0].axvline(mu_r - i * sigma_r, color='blue', linestyle='--', label=f'mu - {i} sigma')
    
    # Plot Posterior for Viscosity
    axes[1].hist(posterior_mu, bins=30, density=True, color='lightgreen', alpha=0.7)
    axes[1].set_title("Posterior of Viscosity (mu)")
    axes[1].set_xlabel("Viscosity (Pa·s)")
    axes[1].set_ylabel("Density")

    # Plotting mu ± n * sigma for Viscosity
    mu_mu = np.mean(posterior_mu)
    sigma_mu = np.std(posterior_mu)
    
    for i in range(1, 4):
        axes[1].axvline(mu_mu + i * sigma_mu, color='red', linestyle='--', label=f'mu + {i} sigma')
        axes[1].axvline(mu_mu - i * sigma_mu, color='blue', linestyle='--', label=f'mu - {i} sigma')
    
    # Plot Posterior for Flow Rate
    axes[2].hist(posterior_Q, bins=30, density=True, color='lightcoral', alpha=0.7)
    axes[2].set_title("Posterior of Flow Rate (Q)")
    axes[2].set_xlabel("Flow Rate (m³/s)")
    axes[2].set_ylabel("Density")

    # Plotting mu ± n * sigma for Flow Rate
    mu_Q = np.mean(posterior_Q)
    sigma_Q = np.std(posterior_Q)
    
    for i in range(1, 4):
        axes[2].axvline(mu_Q + i * sigma_Q, color='red', linestyle='--', label=f'mu + {i} sigma')
        axes[2].axvline(mu_Q - i * sigma_Q, color='blue', linestyle='--', label=f'mu - {i} sigma')
    
    plt.tight_layout()
    plt.savefig(f"{folder_name}/posterior_plots.png")
    plt.close()

    # Save alerts to a CSV file
    with open(f"{folder_name}/alerts.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Alert"])  # Header row
        for alert in alerts:
            writer.writerow([alert])

    print(f"Results saved to folder: {folder_name}")

# -----------------------
# Main Pipeline for All Conditions
# -----------------------
if __name__ == "__main__":

    # Define Q_true
    Q_true = poiseuille_flow_vec(true_r, true_mu)

    # Increase the difference in Q_obs for each condition to make differences very visible
    try:
        Q_obs_extreme_blockage = simulate_noisy_observations(Q_true * 0.005, noise_percent=0.1)  # Extreme Blockage
        Q_obs_blockage = simulate_noisy_observations(Q_true * 0.15, noise_percent=0.15)         # Blockage
        Q_obs_reduced_flow = simulate_noisy_observations(Q_true * 0.4, noise_percent=0.2)      # Reduced Flow
        Q_obs_near_normal_flow = simulate_noisy_observations(Q_true * 1.5, noise_percent=0.1)   # Near Normal Flow
    except ValueError as e:
        print(f"Error in simulation setup: {e}")
        exit()

    # Run the model for each case
    cases = [
        ("extreme_blockage", Q_obs_extreme_blockage),
        ("blockage", Q_obs_blockage),
        ("reduced_flow", Q_obs_reduced_flow),
        ("near_normal_flow", Q_obs_near_normal_flow),
    ]

    # Iterate through each case
    for case_name, Q_obs in cases:
        print(f"Running case: {case_name}")
        trace = build_and_run_model(Q_obs, Q_true)
        if trace:
            posterior_r, posterior_mu, posterior_Q, alerts = analyze_posteriors(trace, Q_true)
            save_results_to_folder(posterior_r, posterior_mu, posterior_Q, alerts, folder_name=f"extremeconditions/{case_name}")
 