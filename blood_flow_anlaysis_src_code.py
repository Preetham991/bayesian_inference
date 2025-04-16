import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# -----------------------
# Sampled Parameters from Normal Distributions
# -----------------------
np.random.seed(42)  # For reproducibility

delta_P = np.random.normal(loc=300, scale=10)         # Pressure drop (Pa)
L = np.random.normal(loc=0.02, scale=0.002)           # Length of artery (m)
true_r = np.random.normal(loc=0.0002, scale=1e-5)     # Radius (m)
true_mu = np.random.normal(loc=0.0035, scale=2e-4)    # Dynamic viscosity (Pa·s)

# -----------------------
# Core Functions
# -----------------------
def poiseuille_flow(r, mu):
    """Poiseuille's equation for volumetric flow rate"""
    return (np.pi * r**4 * delta_P) / (8 * mu * L)

def simulate_noisy_observations(Q_true, num_samples=30, noise_percent=0.05):
    """Generate noisy observations of flow rate"""
    return np.random.normal(loc=Q_true, scale=Q_true * noise_percent, size=num_samples)

def build_and_run_model(Q_obs, Q_true):
    """Bayesian model using PyMC with Gaussian priors"""
    with pm.Model() as model:
        r = pm.HalfNormal("r", mu=0.0015, sigma=0.0003)
        mu = pm.Normal("mu", mu=0.0035, sigma=0.001)

        Q_est = (np.pi * r**4 * delta_P) / (8 * mu * L)
        Q_like = pm.Normal("Q_obs", mu=Q_est, sigma=Q_true * 0.05, observed=Q_obs)

        trace = pm.sample(1000, tune=500, chains=2, target_accept=0.95, return_inferencedata=True)
    return trace

def analyze_posteriors(trace, Q_true):
    """Analyze posterior samples and detect flow abnormalities"""
    posterior_r = trace.posterior["r"].stack(samples=("chain", "draw")).values
    posterior_mu = trace.posterior["mu"].stack(samples=("chain", "draw")).values
    posterior_Q = poiseuille_flow(posterior_r, posterior_mu)

    normal_lower = Q_true * 0.9
    normal_upper = Q_true * 1.1

    alerts = []
    for q in posterior_Q:
        if q < normal_lower * 0.3:
            alerts.append("Blockage")
        elif q < normal_lower:
            alerts.append("Vasoconstriction")
        elif q > normal_upper:
            alerts.append("Vasodilation")
        else:
            alerts.append("Normal")
    return posterior_r, posterior_mu, posterior_Q, alerts

def visualize_results(posterior_r, posterior_mu, posterior_Q, Q_true, alerts):
    """Plot posteriors and abnormal flow regions"""
    alert_counts = Counter(alerts)
    normal_lower = Q_true * 0.9
    normal_upper = Q_true * 1.1

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.hist(posterior_r, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(true_r, color='red', linestyle='--', label='True radius')
    plt.title("Posterior of Radius")
    plt.xlabel("Radius (m)")
    plt.ylabel("Count")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.hist(posterior_mu, bins=30, color='lightgreen', edgecolor='black')
    plt.axvline(true_mu, color='red', linestyle='--', label='True viscosity')
    plt.title("Posterior of Viscosity")
    plt.xlabel("Viscosity (Pa·s)")
    plt.ylabel("Count")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.hist(posterior_Q, bins=30, color='orange', edgecolor='black')
    plt.axvline(Q_true, color='black', linestyle='--', label='True Q')
    plt.axvspan(0, normal_lower * 0.3, color='black', alpha=0.3, label='Blockage')
    plt.axvspan(normal_lower * 0.3, normal_lower, color='red', alpha=0.2, label='Vasoconstriction')
    plt.axvspan(normal_upper, max(posterior_Q), color='blue', alpha=0.2, label='Vasodilation')
    plt.title("Posterior of Flow")
    plt.xlabel("Flow Rate (m³/s)")
    plt.ylabel("Count")
    plt.legend()

    plt.tight_layout()
    plt.savefig("gaussian_blood_flow_analysis.png", dpi=300)
    plt.show()

    print("\nFlow State Summary:")
    for state, count in alert_counts.items():
        percent = (count / len(alerts)) * 100
        print(f"  {state}: {count} samples ({percent:.1f}%)")

# -----------------------
# Execution
# -----------------------
if __name__ == "__main__":
    Q_true = poiseuille_flow(true_r, true_mu)
    Q_obs = simulate_noisy_observations(Q_true)

    trace = build_and_run_model(Q_obs, Q_true)
    posterior_r, posterior_mu, posterior_Q, alerts = analyze_posteriors(trace, Q_true)
    visualize_results(posterior_r, posterior_mu, posterior_Q, Q_true, alerts)
