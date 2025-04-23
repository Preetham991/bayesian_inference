
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Simulate basic parameters
np.random.seed(42)

# Adjusted parameters for faster execution
delta_P = np.random.normal(loc=300, scale=100)
L = np.random.normal(loc=0.02, scale=0.005)
true_r = np.random.normal(loc=0.0002, scale=0.0001)
true_mu = np.random.normal(loc=0.0035, scale=0.001)

# Poiseuille's equation for flowrate
def poiseuille_flow(r, mu, delta_P, L):  # Pass delta_P and L
    return (np.pi * r**4 * delta_P) / (8 * mu * L)

# Simulate noisy observations
def simulate_noisy_observations(Q_true, num_samples=10, noise_percent=0.2):
    return np.random.normal(loc=Q_true, scale=Q_true * noise_percent, size=num_samples)

# Standardization (Z-score normalization)
scaler_standard = StandardScaler()
scaler_normalize = MinMaxScaler()

# Generate true flowrate and noisy observations
Q_true = poiseuille_flow(true_r, true_mu, delta_P, L) # Pass delta_P and L
Q_obs = simulate_noisy_observations(Q_true)

# Apply Standardization and Normalization
Q_obs_standardized = scaler_standard.fit_transform(Q_obs.reshape(-1, 1)).flatten()
Q_obs_normalized = scaler_normalize.fit_transform(Q_obs.reshape(-1, 1)).flatten()

# Function to run Bayesian model with optimized likelihood
def build_and_run_model_optimized(Q_obs_transformed, data_type="original"):
    with pm.Model() as model:
        # Define priors for radius and viscosity
        r = pm.HalfNormal("r", sigma=0.0035)
        mu = pm.Normal("mu", mu=0.0035, sigma=0.01)

        # Flowrate model (using the original equation)
        Q_est = poiseuille_flow(r, mu, delta_P, L) # Pass delta_P and L

        # Likelihood with noisy observations, simplified for scaled data
        if data_type == "original":
            sigma_likelihood = Q_true * 0.2  # Original scale
        elif data_type == "standardized":
            sigma_likelihood = 0.5  # Example:  Adjust based on your standardized data's noise
        elif data_type == "normalized":
            sigma_likelihood = 0.1  # Example: Adjust based on your normalized data's noise
        else:
            raise ValueError("Invalid data_type. Must be 'original', 'standardized', or 'normalized'")

        pm.Normal("Q_obs", mu=Q_est, sigma=sigma_likelihood, observed=Q_obs_transformed)

        # Sampling using MCMC (increased target_accept)
        trace = pm.sample(draws=1000, tune=500, chains=4, cores=4, target_accept=0.95,init="adapt_diag",sampler_kwargs={"nuts_sampler": "numpyro"}, return_inferencedata=True)
    return trace

# Main function
def main():
    # Run the model for original, standardized, and normalized observations
    trace_original = build_and_run_model_optimized(Q_obs, data_type="original")
    trace_standardized = build_and_run_model_optimized(Q_obs_standardized, data_type="standardized")
    trace_normalized = build_and_run_model_optimized(Q_obs_normalized, data_type="normalized")

    # Extract posterior samples
    posterior_r_original = trace_original.posterior["r"].stack(samples=("chain", "draw")).values
    posterior_mu_original = trace_original.posterior["mu"].stack(samples=("chain", "draw")).values
    posterior_Q_original = poiseuille_flow(posterior_r_original, posterior_mu_original, delta_P, L)

    posterior_r_standardized = trace_standardized.posterior["r"].stack(samples=("chain", "draw")).values
    posterior_mu_standardized = trace_standardized.posterior["mu"].stack(samples=("chain", "draw")).values
    posterior_Q_standardized = poiseuille_flow(posterior_r_standardized, posterior_mu_standardized, delta_P, L)

    posterior_r_normalized = trace_normalized.posterior["r"].stack(samples=("chain", "draw")).values
    posterior_mu_normalized = trace_normalized.posterior["mu"].stack(samples=("chain", "draw")).values
    posterior_Q_normalized = poiseuille_flow(posterior_r_normalized, posterior_mu_normalized, delta_P, L)
    # Set folder structure for saving results and plots
    folder_path = "standardization_normalization_effects"
    os.makedirs(folder_path, exist_ok=True)
    results_folder = os.path.join(folder_path, "results")
    os.makedirs(results_folder, exist_ok=True)
    plots_folder = os.path.join(folder_path, "plots")
    os.makedirs(plots_folder, exist_ok=True)

    # Save results to CSV (Including flow rates)
    results_df = pd.DataFrame({
        "Scaling Method": ["Original", "Standardized", "Normalized"],
        "Mean Radius": [
            np.mean(posterior_r_original),
            np.mean(posterior_r_standardized),
            np.mean(posterior_r_normalized)
        ],
        "Mean Viscosity": [
            np.mean(posterior_mu_original),
            np.mean(posterior_mu_standardized),
            np.mean(posterior_mu_normalized)
        ],
        "Mean Flow Rate": [
            np.mean(posterior_Q_original),
            np.mean(posterior_Q_standardized),
            np.mean(posterior_Q_normalized)
        ]
    })
    csv_filename = os.path.join(results_folder, "standardization_normalization_effects.csv")
    results_df.to_csv(csv_filename, index=False)

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    axs[0].hist(posterior_r_original, bins=20, alpha=0.5, label="Original", color='blue')
    axs[0].hist(posterior_r_standardized, bins=20, alpha=0.5, label="Standardized", color='green')
    axs[0].hist(posterior_r_normalized, bins=20, alpha=0.5, label="Normalized", color='red')
    axs[0].set_title("Posterior Distribution of Radius")
    axs[0].legend()

    axs[1].hist(posterior_mu_original, bins=20, alpha=0.5, label="Original", color='blue')
    axs[1].hist(posterior_mu_standardized, bins=20, alpha=0.5, label="Standardized", color='green')
    axs[1].hist(posterior_mu_normalized, bins=20, alpha=0.5, label="Normalized", color='red')
    axs[1].set_title("Posterior Distribution of Viscosity")
    axs[1].legend()

    axs[2].hist(posterior_Q_original, bins=20, alpha=0.5, label="Original", color='blue')
    axs[2].hist(posterior_Q_standardized, bins=20, alpha=0.5, label="Standardized", color='green')
    axs[2].hist(posterior_Q_normalized, bins=20, alpha=0.5, label="Normalized", color='red')
    axs[2].set_title("Posterior Distribution of Flow Rate")
    axs[2].legend()

    plot_filename = os.path.join(plots_folder, "standardization_normalization_effects.png")
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()

    print("Results saved to CSV and plots saved successfully.")

if __name__ == "__main__":
    main()
