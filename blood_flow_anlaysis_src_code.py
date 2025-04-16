
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
    """Plot posterior histograms and flow states"""
    alert_counts = Counter(alerts)
    normal_lower = Q_true * 0.9
    normal_upper = Q_true * 1.1

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.hist(posterior_r, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(true_r, color='red', linestyle='--', label='True r')
    plt.title("Posterior of Radius")
    plt.xlabel("Radius (m)")
    plt.ylabel("Count")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.hist(posterior_mu, bins=30, color='lightgreen', edgecolor='black')
    plt.axvline(true_mu, color='red', linestyle='--', label='True mu')
    plt.title("Posterior of Viscosity")
    plt.xlabel("Viscosity (Pa·s)")
    plt.ylabel("Count")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.hist(posterior_Q, bins=30, color='orange', edgecolor='black')
    plt.axvline(Q_true, color='black', linestyle='--', label='True Q')
    plt.axvspan(0, normal_lower * 0.3, color='black', alpha=0.3, label='Blockage')
    plt.axvspan(normal_lower * 0.3, normal_lower, color='red', alpha=0.2, label='Vasoconstriction')
    plt.axvspan(normal_upper, np.max(posterior_Q), color='blue', alpha=0.2, label='Vasodilation')
    plt.title("Posterior of Flow")
    plt.xlabel("Flow Rate (m³/s)")
    plt.ylabel("Count")
    plt.legend()

    plt.tight_layout()
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(os.path.join(folder_path, "gaussian_blood_flow_analysis.png"), dpi=300)
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





# import pymc as pm
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import Counter
# import os
# import pandas as pd

# # -----------------------
# # Sampled Parameters with Increased Dispersion
# # -----------------------
# np.random.seed(42)

# # Sampled using normal distributions
# delta_P = np.random.normal(loc=300, scale=100)         # Pressure drop (Pa)
# L = np.random.normal(loc=0.02, scale=0.01)             # Length of artery (m)
# true_r = np.random.normal(loc=0.0002, scale=0.0001)    # Radius (m)
# true_mu = np.random.normal(loc=0.0035, scale=0.002)    # Dynamic viscosity (Pa·s)

# # -----------------------
# # Core Functions
# # -----------------------

# def poiseuille_flow_vec(r, mu):
#     """Vectorized Poiseuille's equation"""
#     return (np.pi * np.power(r, 4) * delta_P) / (8 * mu * L)

# def simulate_noisy_observations(Q_true, num_samples=30, noise_percent=0.2):
#     """Generate noisy observations"""
#     return np.random.normal(loc=Q_true, scale=Q_true * noise_percent, size=num_samples)

# def build_and_run_model(Q_obs, Q_true):
#     """Bayesian model using PyMC"""
#     with pm.Model() as model:
#         r = pm.HalfNormal("r", sigma=0.0005)
#         mu = pm.Normal("mu", mu=0.0035, sigma=0.003)
#         Q_est = (np.pi * r**4 * delta_P) / (8 * mu * L)

#         pm.Normal("Q_obs", mu=Q_est, sigma=Q_true * 0.2, observed=Q_obs)

#         trace = pm.sample(
#             draws=1000,
#             tune=500,
#             chains=4,
#             cores=4,
#             init="adapt_diag",
#             target_accept=0.95,
#             return_inferencedata=True,
#             idata_kwargs={"log_likelihood": False},
#             random_seed=42
#         )
#     return trace

# def analyze_posteriors(trace, Q_true):
#     """Analyze posterior samples and detect abnormalities"""
#     posterior_r = trace.posterior["r"].stack(samples=("chain", "draw")).values
#     posterior_mu = trace.posterior["mu"].stack(samples=("chain", "draw")).values
#     posterior_Q = poiseuille_flow_vec(posterior_r, posterior_mu)

#     extreme_mu_values = [true_mu + true_mu * 0.003, 2 * true_mu + true_mu * 0.003, true_mu * 10]
#     extreme_r_values = [true_r * 1.2, true_r * 1.5, true_r * 2]
#     extreme_Q = [poiseuille_flow_vec(np.array([r]), true_mu)[0] for r in extreme_r_values]
#     extreme_mu_Q = [poiseuille_flow_vec(true_r, np.array([mu]))[0] for mu in extreme_mu_values]

#     return posterior_r, posterior_mu, posterior_Q, extreme_r_values, extreme_mu_values, extreme_Q, extreme_mu_Q

# def save_results_to_csv(posterior_r, posterior_mu, posterior_Q, folder_path, extreme_r_values, extreme_mu_values, extreme_Q, extreme_mu_Q):
#     """Save posterior results with highlighted bins for extreme values"""
#     df = pd.DataFrame({
#         "posterior_r": posterior_r,
#         "posterior_mu": posterior_mu,
#         "posterior_Q": posterior_Q,
#     })
    
#     # Binning for r, mu, Q
#     bins_r = np.histogram(posterior_r, bins=30)[1]
#     bins_mu = np.histogram(posterior_mu, bins=30)[1]
#     bins_Q = np.histogram(posterior_Q, bins=30)[1]

#     # Highlight the bins containing extreme values for r and mu
#     df["highlight_r"] = np.digitize(df["posterior_r"], bins_r, right=True)
#     df["highlight_mu"] = np.digitize(df["posterior_mu"], bins_mu, right=True)
#     df["highlight_Q"] = np.digitize(df["posterior_Q"], bins_Q, right=True)

#     # Mark bins containing extreme values
#     df["extreme_r"] = df["highlight_r"].isin([np.digitize(r, bins_r, right=True) for r in extreme_r_values])
#     df["extreme_mu"] = df["highlight_mu"].isin([np.digitize(mu, bins_mu, right=True) for mu in extreme_mu_values])
#     df["extreme_Q"] = df["highlight_Q"].isin([np.digitize(Q, bins_Q, right=True) for Q in extreme_Q])

#     os.makedirs(folder_path, exist_ok=True)
#     df.to_csv(os.path.join(folder_path, "posterior_results_with_highlights.csv"), index=False)

# def visualize_results(posterior_r, posterior_mu, posterior_Q, Q_true, folder_path, extreme_r_values, extreme_mu_values, extreme_Q, extreme_mu_Q):
#     """Plot posterior histograms and flow states"""
#     plt.figure(figsize=(15, 5))

#     # Radius Posterior
#     plt.subplot(1, 3, 1)
#     plt.hist(posterior_r, bins=30, color='skyblue', edgecolor='black')
#     plt.axvline(true_r, color='red', linestyle='--', label='True r')
#     for r in extreme_r_values:
#         plt.axvline(r, color='green', linestyle='--', label=f'Extreme r: {r}')
#     plt.title("Posterior of Radius")
#     plt.xlabel("Radius (m)")
#     plt.ylabel("Count")
#     plt.legend()

#     # Viscosity Posterior
#     plt.subplot(1, 3, 2)
#     plt.hist(posterior_mu, bins=30, color='lightgreen', edgecolor='black')
#     plt.axvline(true_mu, color='red', linestyle='--', label='True mu')
#     for mu in extreme_mu_values:
#         plt.axvline(mu, color='green', linestyle='--', label=f'Extreme mu: {mu}')
#     plt.title("Posterior of Viscosity")
#     plt.xlabel("Viscosity (Pa·s)")
#     plt.ylabel("Count")
#     plt.legend()

#     # Flow Rate Posterior
#     plt.subplot(1, 3, 3)
#     plt.hist(posterior_Q, bins=30, color='orange', edgecolor='black')
#     plt.axvline(Q_true, color='black', linestyle='--', label='True Q')
#     for Q in extreme_Q:
#         plt.axvline(Q, color='green', linestyle='--', label=f'Extreme Q: {Q}')
#     plt.title("Posterior of Flow Rate")
#     plt.xlabel("Flow Rate (m³/s)")
#     plt.ylabel("Count")
#     plt.legend()

#     plt.tight_layout()
#     os.makedirs(folder_path, exist_ok=True)
#     plt.savefig(os.path.join(folder_path, "posterior_analysis_with_extremes.png"), dpi=300)
#     plt.show()

# # -----------------------
# # Execution
# # -----------------------
# if __name__ == "__main__":
#     folder_path = "blood_flow_analysis_results"
#     Q_true = poiseuille_flow_vec(true_r, true_mu)
#     Q_obs = simulate_noisy_observations(Q_true)

#     trace = build_and_run_model(Q_obs, Q_true)
#     posterior_r, posterior_mu, posterior_Q, extreme_r_values, extreme_mu_values, extreme_Q, extreme_mu_Q = analyze_posteriors(trace, Q_true)

#     save_results_to_csv(posterior_r, posterior_mu, posterior_Q, folder_path, extreme_r_values, extreme_mu_values, extreme_Q, extreme_mu_Q)
#     visualize_results(posterior_r, posterior_mu, posterior_Q, Q_true, folder_path, extreme_r_values, extreme_mu_values, extreme_Q, extreme_mu_Q)
