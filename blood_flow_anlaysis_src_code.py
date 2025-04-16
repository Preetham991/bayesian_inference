
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import pandas as pd

# -----------------------
# Sampled Parameters with Increased Dispersion
# -----------------------
np.random.seed(42)

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

def save_results_to_csv(posterior_r, posterior_mu, posterior_Q, alerts, folder_path):
    """Save posterior results"""
    df = pd.DataFrame({
        "posterior_r": posterior_r,
        "posterior_mu": posterior_mu,
        "posterior_Q": posterior_Q,
        "alert": alerts
    })
    os.makedirs(folder_path, exist_ok=True)
    df.to_csv(os.path.join(folder_path, "posterior_results.csv"), index=False)

def visualize_results(posterior_r, posterior_mu, posterior_Q, Q_true, alerts, folder_path):
    """Plot posterior histograms and flow states with sigma binning for r, mu, and Q"""
    alert_counts = Counter(alerts)
    normal_lower = Q_true * 0.9
    normal_upper = Q_true * 1.1

    # ---------------------------
    # Radius Stats
    # ----------------------------
    r_mean = np.mean(posterior_r)
    r_std = np.std(posterior_r)
    r_minus_3sigma, r_plus_3sigma = r_mean - 3 * r_std, r_mean + 3 * r_std
    r_minus_2sigma, r_plus_2sigma = r_mean - 2 * r_std, r_mean + 2 * r_std
    r_minus_1sigma, r_plus_1sigma = r_mean - r_std, r_mean + r_std

    # ----------------------------
    # Viscosity Stats
    # ----------------------------
    mu_mean = np.mean(posterior_mu)
    mu_std = np.std(posterior_mu)
    mu_minus_3sigma, mu_plus_3sigma = mu_mean - 3 * mu_std, mu_mean + 3 * mu_std
    mu_minus_2sigma, mu_plus_2sigma = mu_mean - 2 * mu_std, mu_mean + 2 * mu_std
    mu_minus_1sigma, mu_plus_1sigma = mu_mean - mu_std, mu_mean + mu_std

    # ----------------------------
    # Flow Stats
    # ----------------------------
    Q_mean = np.mean(posterior_Q)
    Q_std = np.std(posterior_Q)
    Q_minus_3sigma, Q_plus_3sigma = Q_mean - 3 * Q_std, Q_mean + 3 * Q_std
    Q_minus_2sigma, Q_plus_2sigma = Q_mean - 2 * Q_std, Q_mean + 2 * Q_std
    Q_minus_1sigma, Q_plus_1sigma = Q_mean - Q_std, Q_mean + Q_std

    plt.figure(figsize=(18, 5))

    # ----------------------------
    # Radius Plot (r)
    # ----------------------------
    plt.subplot(1, 3, 1)
    plt.hist(posterior_r, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(true_r, color='red', linestyle='--', label='True r')
    plt.axvspan(r_minus_3sigma, r_plus_3sigma, color='yellow', alpha=0.3, label='r ± 3σ')
    plt.axvspan(r_minus_2sigma, r_plus_2sigma, color='orange', alpha=0.3, label='r ± 2σ')
    plt.axvspan(r_minus_1sigma, r_plus_1sigma, color='green', alpha=0.3, label='r ± σ')
    plt.title("Posterior of Radius")
    plt.xlabel("Radius (m)")
    plt.ylabel("Count")
    plt.legend()

    # ----------------------------
    # Viscosity Plot (mu)
    # ----------------------------
    plt.subplot(1, 3, 2)
    plt.hist(posterior_mu, bins=30, color='lightgreen', edgecolor='black')
    plt.axvline(true_mu, color='red', linestyle='--', label='True μ')
    plt.axvspan(mu_minus_3sigma, mu_plus_3sigma, color='yellow', alpha=0.3, label='μ ± 3σ')
    plt.axvspan(mu_minus_2sigma, mu_plus_2sigma, color='orange', alpha=0.3, label='μ ± 2σ')
    plt.axvspan(mu_minus_1sigma, mu_plus_1sigma, color='green', alpha=0.3, label='μ ± σ')
    plt.title("Posterior of Viscosity")
    plt.xlabel("Viscosity (Pa·s)")
    plt.ylabel("Count")
    plt.legend()

    # ----------------------------
    # Flow Plot (Q)
    # ----------------------------
    plt.subplot(1, 3, 3)
    plt.hist(posterior_Q, bins=30, color='orange', edgecolor='black')
    plt.axvline(Q_true, color='black', linestyle='--', label='True Q')
    plt.axvspan(Q_minus_3sigma, Q_plus_3sigma, color='yellow', alpha=0.3, label='Q ± 3σ')
    plt.axvspan(Q_minus_2sigma, Q_plus_2sigma, color='orange', alpha=0.3, label='Q ± 2σ')
    plt.axvspan(Q_minus_1sigma, Q_plus_1sigma, color='green', alpha=0.3, label='Q ± σ')
    plt.axvspan(0, normal_lower * 0.3, color='black', alpha=0.2, label='Blockage')
    plt.axvspan(normal_lower * 0.3, normal_lower, color='red', alpha=0.2, label='Vasoconstriction(contracted  blood vessel)')
    plt.axvspan(normal_upper, np.max(posterior_Q), color='blue', alpha=0.2, label='Vasodilation(Enlarged blood vessel)')
    plt.title("Posterior of Flow")
    plt.xlabel("Flow Rate (m³/s)")
    plt.ylabel("Count")
    plt.legend()

    plt.tight_layout()
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(os.path.join(folder_path, "blood_flow_analysis.png"), dpi=300)
    plt.show()

    print("\nFlow State Summary:")
    for state, count in alert_counts.items():
        percent = (count / len(alerts)) * 100
        print(f"  {state}: {count} samples ({percent:.1f}%)")


# -----------------------
# Execution
# -----------------------
if __name__ == "__main__":
    folder_path = "blood_flow_analysis_results"
    Q_true = poiseuille_flow_vec(true_r, true_mu)
    Q_obs = simulate_noisy_observations(Q_true)

    trace = build_and_run_model(Q_obs, Q_true)
    posterior_r, posterior_mu, posterior_Q, alerts = analyze_posteriors(trace, Q_true)

    save_results_to_csv(posterior_r, posterior_mu, posterior_Q, alerts, folder_path)
    visualize_results(posterior_r, posterior_mu, posterior_Q, Q_true, alerts, folder_path)
