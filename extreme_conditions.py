
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import pandas as pd

# -----------------------
# Sampled Parameters with Adjusted Distributions
# -----------------------
np.random.seed(33)

# Widening the standard deviations to increase variability
delta_P = np.random.normal(loc=300, scale=100)         # Pressure drop (Pa)
L = np.random.normal(loc=0.02, scale=0.01)             # Length of artery (m)
true_r = np.random.normal(loc=0.0002, scale=0.00015)   # Radius (m) with higher spread
true_mu = np.random.normal(loc=0.0035, scale=0.0015)   # Dynamic viscosity (Pa·s) with higher spread

# -----------------------
# Core Functions (No changes)
# -----------------------

def poiseuille_flow_vec(r, mu):
    """Vectorized Poiseuille's equation"""
    return (np.pi * np.power(r, 4) * delta_P) / (8 * mu * L)

def simulate_noisy_observations(Q_true, num_samples=30, noise_percent=0.4):
    """Generate noisy observations with increased noise percentage"""
    return np.random.normal(loc=Q_true, scale=Q_true * noise_percent, size=num_samples)

def build_and_run_model(Q_obs, Q_true):
    """Bayesian model using PyMC"""
    with pm.Model() as model:
        r = pm.HalfNormal("r", sigma=0.0005)
        mu = pm.Normal("mu", mu=0.0035, sigma=0.003)
        Q_est = (np.pi * r**4 * delta_P) / (8 * mu * L)

        pm.Normal("Q_obs", mu=Q_est, sigma=Q_true * 0.4, observed=Q_obs)  # Increased noise in observed Q

        trace = pm.sample(
            draws=1000,
            tune=500,
            chains=4,
            cores=4,
            init="adapt_diag",
            target_accept=0.95,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": False},
            random_seed=42
        )
    return trace

def analyze_posteriors(trace, Q_true):
    """Analyze posterior samples and categorize estimations"""
    posterior_r = trace.posterior["r"].stack(samples=("chain", "draw")).values
    posterior_mu = trace.posterior["mu"].stack(samples=("chain", "draw")).values
    posterior_Q = poiseuille_flow_vec(posterior_r, posterior_mu)

    # Calculate extreme values based on mu and sigma
    mu_plus_sigma = true_mu + true_mu * 0.003
    two_mu_plus_sigma = 2 * true_mu + true_mu * 0.003
    extreme_mu_values = [true_mu * 10]  # Extreme values for mu
    extreme_r_values = [true_r * 1.2, true_r * 1.5, true_r * 2]

    extreme_Q = [poiseuille_flow_vec(np.array([r]), true_mu)[0] for r in extreme_r_values]
    extreme_mu_Q = [poiseuille_flow_vec(true_r, np.array([mu]))[0] for mu in extreme_mu_values]

    # Categorize the results
    def categorize(values, mu, sigma):
        """Categorize values based on their deviation from mu"""
        within_mu_sigma = []
        within_two_mu_sigma = []
        extreme = []
        
        for value in values:
            if mu - sigma <= value <= mu + sigma:
                within_mu_sigma.append(value)
            elif mu - 2*sigma <= value <= mu + 2*sigma:
                within_two_mu_sigma.append(value)
            else:
                extreme.append(value)
        
        return within_mu_sigma, within_two_mu_sigma, extreme

    # Categorize posterior samples
    r_within_mu_sigma, r_within_two_mu_sigma, r_extreme = categorize(posterior_r, true_r, true_r * 0.5)
    mu_within_mu_sigma, mu_within_two_mu_sigma, mu_extreme = categorize(posterior_mu, true_mu, true_mu * 0.003)

    return posterior_r, posterior_mu, posterior_Q, r_within_mu_sigma, r_within_two_mu_sigma, r_extreme, mu_within_mu_sigma, mu_within_two_mu_sigma, mu_extreme, extreme_r_values, extreme_mu_values, extreme_Q, extreme_mu_Q

def save_results_to_csv(posterior_r, posterior_mu, posterior_Q, folder_path, r_within_mu_sigma, r_within_two_mu_sigma, r_extreme, mu_within_mu_sigma, mu_within_two_mu_sigma, mu_extreme, extreme_r_values, extreme_mu_values, extreme_Q, extreme_mu_Q):
    """Save posterior results with categorized values"""
    df = pd.DataFrame({
        "posterior_r": posterior_r,
        "posterior_mu": posterior_mu,
        "posterior_Q": posterior_Q,
    })
    
    # Mark categories for r and mu
    df["r_within_mu_sigma"] = df["posterior_r"].isin(r_within_mu_sigma)
    df["r_within_two_mu_sigma"] = df["posterior_r"].isin(r_within_two_mu_sigma)
    df["r_extreme"] = df["posterior_r"].isin(r_extreme)
    
    df["mu_within_mu_sigma"] = df["posterior_mu"].isin(mu_within_mu_sigma)
    df["mu_within_two_mu_sigma"] = df["posterior_mu"].isin(mu_within_two_mu_sigma)
    df["mu_extreme"] = df["posterior_mu"].isin(mu_extreme)

    os.makedirs(folder_path, exist_ok=True)
    df.to_csv(os.path.join(folder_path, "posterior_results_with_categories.csv"), index=False)

def visualize_results(posterior_r, posterior_mu, posterior_Q, Q_true, folder_path, r_within_mu_sigma, r_within_two_mu_sigma, r_extreme, mu_within_mu_sigma, mu_within_two_mu_sigma, mu_extreme, extreme_r_values, extreme_mu_values, extreme_Q, extreme_mu_Q):
    """Plot posterior histograms with categories"""
    plt.figure(figsize=(15, 5))

    # Radius Posterior
    plt.subplot(1, 3, 1)
    plt.hist(posterior_r, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(true_r, color='red', linestyle='--', label='True r')
    for r in extreme_r_values:
        plt.axvline(r, color='green', linestyle='--', label=f'Extreme r: {r}')
    plt.title("Posterior of Radius")
    plt.xlabel("Radius (m)")
    plt.ylabel("Count")
    plt.legend()

    # Viscosity Posterior
    plt.subplot(1, 3, 2)
    plt.hist(posterior_mu, bins=30, color='lightgreen', edgecolor='black')
    plt.axvline(true_mu, color='red', linestyle='--', label='True mu')
    for mu in extreme_mu_values:
        plt.axvline(mu, color='green', linestyle='--', label=f'Extreme mu: {mu}')
    plt.title("Posterior of Viscosity")
    plt.xlabel("Viscosity (Pa·s)")
    plt.ylabel("Count")
    plt.legend()

    # Flow Rate Posterior
    plt.subplot(1, 3, 3)
    plt.hist(posterior_Q, bins=30, color='orange', edgecolor='black')
    plt.axvline(Q_true, color='black', linestyle='--', label='True Q')
    for Q in extreme_Q:
        plt.axvline(Q, color='green', linestyle='--', label=f'Extreme Q: {Q}')
    plt.title("Posterior of Flow Rate")
    plt.xlabel("Flow Rate (m³/s)")
    plt.ylabel("Count")
    plt.legend()

    plt.tight_layout()
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(os.path.join(folder_path, "posterior_analysis_with_categories.png"), dpi=300)
    #plt.show()

# -----------------------
# Execution
# -----------------------
if __name__ == "__main__":
    folder_path = "blood_flow_analysis_results"
    Q_true = poiseuille_flow_vec(true_r, true_mu)
    Q_obs = simulate_noisy_observations(Q_true)

    trace = build_and_run_model(Q_obs, Q_true)
    posterior_r, posterior_mu, posterior_Q, r_within_mu_sigma, r_within_two_mu_sigma, r_extreme, mu_within_mu_sigma, mu_within_two_mu_sigma, mu_extreme, extreme_r_values, extreme_mu_values, extreme_Q, extreme_mu_Q = analyze_posteriors(trace, Q_true)

    save_results_to_csv(posterior_r, posterior_mu, posterior_Q, folder_path, r_within_mu_sigma, r_within_two_mu_sigma, r_extreme, mu_within_mu_sigma, mu_within_two_mu_sigma, mu_extreme, extreme_r_values, extreme_mu_values, extreme_Q, extreme_mu_Q)
    visualize_results(posterior_r, posterior_mu, posterior_Q, Q_true, folder_path, r_within_mu_sigma, r_within_two_mu_sigma, r_extreme, mu_within_mu_sigma, mu_within_two_mu_sigma, mu_extreme, extreme_r_values, extreme_mu_values, extreme_Q, extreme_mu_Q)





