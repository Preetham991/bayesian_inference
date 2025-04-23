# # # # # # # import numpy as np
# # # # # # # import matplotlib.pyplot as plt
# # # # # # # import pymc as pm
# # # # # # # import theano.tensor as tt
# # # # # # # from sklearn.preprocessing import StandardScaler, MinMaxScaler
# # # # # # # import pickle
# # # # # # # import os

# # # # # # # # -------------------------
# # # # # # # # Preprocessing Functions
# # # # # # # # -------------------------

# # # # # # # def standardize_inputs(*inputs):
# # # # # # #     """Standardize inputs to have zero mean and unit variance."""
# # # # # # #     scaler = StandardScaler()
# # # # # # #     standardized_inputs = [scaler.fit_transform(input.reshape(-1, 1)).flatten() for input in inputs]
# # # # # # #     return standardized_inputs

# # # # # # # def normalize_inputs(*inputs):
# # # # # # #     """Normalize inputs to a [0, 1] range."""
# # # # # # #     scaler = MinMaxScaler()
# # # # # # #     normalized_inputs = [scaler.fit_transform(input.reshape(-1, 1)).flatten() for input in inputs]
# # # # # # #     return normalized_inputs

# # # # # # # # -----------------------
# # # # # # # # Model Definition and Running
# # # # # # # # -----------------------

# # # # # # # def poiseuille_flow_vec(radius, viscosity):
# # # # # # #     """Poiseuille flow equation for a given radius and viscosity."""
# # # # # # #     # Assuming laminar flow in a pipe, calculate flow rate (Q)
# # # # # # #     # Q = (pi * r^4 * delta_P) / (8 * mu * L)
# # # # # # #     # In this case, we're using a simplified relation for demonstration
# # # # # # #     return np.pi * radius**4 / (8 * viscosity)

# # # # # # # def simulate_noisy_observations(true_flow_rate, noise_level=0.05):
# # # # # # #     """Simulate noisy observations of the flow rate."""
# # # # # # #     noise = np.random.normal(0, noise_level, len(true_flow_rate))
# # # # # # #     return true_flow_rate + noise

# # # # # # # def build_and_run_model(Q_obs, Q_true):
# # # # # # #     """Build the Bayesian model and run inference."""
# # # # # # #     with pm.Model() as model:
# # # # # # #         # Priors for model parameters
# # # # # # #         delta_P = pm.Normal('delta_P', mu=10, sigma=5)
# # # # # # #         L = pm.Normal('L', mu=1.0, sigma=0.5)
# # # # # # #         true_r = pm.Normal('true_r', mu=0.05, sigma=0.01)
# # # # # # #         true_mu = pm.Normal('true_mu', mu=0.001, sigma=0.0005)

# # # # # # #         # Poiseuille flow equation as the likelihood function
# # # # # # #         Q_model = poiseuille_flow_vec(true_r, true_mu)

# # # # # # #         # Likelihood (observed data)
# # # # # # #         obs = pm.Normal('obs', mu=Q_model, sigma=0.05, observed=Q_obs)

# # # # # # #         # Inference: sampling from the posterior
# # # # # # #         trace = pm.sample(1000, return_inferencedata=False)
        
# # # # # # #     return trace

# # # # # # # def analyze_posteriors(trace, Q_true):
# # # # # # #     """Analyze the posterior distributions."""
# # # # # # #     # Extracting samples
# # # # # # #     posterior_r = trace['true_r']
# # # # # # #     posterior_mu = trace['true_mu']
# # # # # # #     posterior_Q = poiseuille_flow_vec(posterior_r, posterior_mu)
    
# # # # # # #     # Calculate the deviation from true values
# # # # # # #     alerts = {
# # # # # # #         'r_alert': np.abs(np.mean(posterior_r) - np.mean(Q_true)),
# # # # # # #         'mu_alert': np.abs(np.mean(posterior_mu) - np.mean(Q_true))
# # # # # # #     }

# # # # # # #     return posterior_r, posterior_mu, posterior_Q, alerts

# # # # # # # def save_results_to_csv_and_pickle(r, mu, Q, alerts, folder_path):
# # # # # # #     """Save the results to CSV and pickle format."""
# # # # # # #     if not os.path.exists(folder_path):
# # # # # # #         os.makedirs(folder_path)

# # # # # # #     # Saving to pickle
# # # # # # #     with open(f"{folder_path}/results.pkl", "wb") as f:
# # # # # # #         pickle.dump((r, mu, Q, alerts), f)

# # # # # # #     # Saving to CSV
# # # # # # #     results = np.column_stack((r, mu, Q))
# # # # # # #     np.savetxt(f"{folder_path}/results.csv", results, delimiter=",")

# # # # # # # # -----------------------
# # # # # # # # Main Pipeline
# # # # # # # # -----------------------
# # # # # # # if __name__ == "__main__":
# # # # # # #     # True values for parameters
# # # # # # #     true_r = np.array([0.05])  # True radius in meters
# # # # # # #     true_mu = np.array([0.001])  # True viscosity in Pascal-seconds
# # # # # # #     delta_P = np.array([10])  # Pressure drop in Pascal
# # # # # # #     L = np.array([1.0])  # Length in meters
    
# # # # # # #     # True flow rate with original values
# # # # # # #     Q_true = poiseuille_flow_vec(true_r, true_mu)
    
# # # # # # #     # Simulate noisy observations
# # # # # # #     Q_obs = simulate_noisy_observations(Q_true)

# # # # # # #     # Standardized inputs
# # # # # # #     standardized_delta_P, standardized_L, standardized_r, standardized_mu = standardize_inputs(delta_P, L, true_r, true_mu)
    
# # # # # # #     # Normalized inputs
# # # # # # #     normalized_delta_P, normalized_L, normalized_r, normalized_mu = normalize_inputs(delta_P, L, true_r, true_mu)

# # # # # # #     # Run model with original inputs
# # # # # # #     trace_original = build_and_run_model(Q_obs, Q_true)

# # # # # # #     # Run model with standardized inputs
# # # # # # #     trace_standardized = build_and_run_model(Q_obs, Q_true)

# # # # # # #     # Run model with normalized inputs
# # # # # # #     trace_normalized = build_and_run_model(Q_obs, Q_true)

# # # # # # #     # Posterior analysis for original inputs
# # # # # # #     posterior_r_original, posterior_mu_original, posterior_Q_original, alerts_original = analyze_posteriors(trace_original, Q_true)

# # # # # # #     # Posterior analysis for standardized inputs
# # # # # # #     posterior_r_standardized, posterior_mu_standardized, posterior_Q_standardized, alerts_standardized = analyze_posteriors(trace_standardized, Q_true)

# # # # # # #     # Posterior analysis for normalized inputs
# # # # # # #     posterior_r_normalized, posterior_mu_normalized, posterior_Q_normalized, alerts_normalized = analyze_posteriors(trace_normalized, Q_true)

# # # # # # #     # Compare the effects of standardization and normalization on posterior distributions
# # # # # # #     plt.figure(figsize=(12, 8))

# # # # # # #     # Plot posterior for radius
# # # # # # #     plt.subplot(2, 2, 1)
# # # # # # #     plt.hist(posterior_r_original, bins=30, alpha=0.5, label='Original')
# # # # # # #     plt.hist(posterior_r_standardized, bins=30, alpha=0.5, label='Standardized')
# # # # # # #     plt.hist(posterior_r_normalized, bins=30, alpha=0.5, label='Normalized')
# # # # # # #     plt.title("Posterior Distribution of Radius")
# # # # # # #     plt.legend()

# # # # # # #     # Plot posterior for viscosity
# # # # # # #     plt.subplot(2, 2, 2)
# # # # # # #     plt.hist(posterior_mu_original, bins=30, alpha=0.5, label='Original')
# # # # # # #     plt.hist(posterior_mu_standardized, bins=30, alpha=0.5, label='Standardized')
# # # # # # #     plt.hist(posterior_mu_normalized, bins=30, alpha=0.5, label='Normalized')
# # # # # # #     plt.title("Posterior Distribution of Viscosity")
# # # # # # #     plt.legend()

# # # # # # #     # Plot posterior for flow rate
# # # # # # #     plt.subplot(2, 2, 3)
# # # # # # #     plt.hist(posterior_Q_original, bins=30, alpha=0.5, label='Original')
# # # # # # #     plt.hist(posterior_Q_standardized, bins=30, alpha=0.5, label='Standardized')
# # # # # # #     plt.hist(posterior_Q_normalized, bins=30, alpha=0.5, label='Normalized')
# # # # # # #     plt.title("Posterior Distribution of Flow Rate")
# # # # # # #     plt.legend()

# # # # # # #     # Show the plots
# # # # # # #     plt.tight_layout()
# # # # # # #     plt.show()
# # # # # # #     # Ensure plots directory exists
# # # # # # #     folder_path = "blood_flow_analysis_results"
# # # # # # #     plot_folder = os.path.join(folder_path, "plots")
# # # # # # #     os.makedirs(plot_folder, exist_ok=True)

# # # # # # #     # Plot and save posterior for radius
# # # # # # #     plt.figure(figsize=(6, 4))
# # # # # # #     plt.hist(posterior_r_original, bins=30, alpha=0.5, label='Original')
# # # # # # #     plt.hist(posterior_r_standardized, bins=30, alpha=0.5, label='Standardized')
# # # # # # #     plt.hist(posterior_r_normalized, bins=30, alpha=0.5, label='Normalized')
# # # # # # #     plt.title("Posterior Distribution of Radius")
# # # # # # #     plt.xlabel("Radius (m)")
# # # # # # #     plt.ylabel("Frequency")
# # # # # # #     plt.legend()
# # # # # # #     plt.tight_layout()
# # # # # # #     plt.savefig(f"{plot_folder}/posterior_radius.png", dpi=300)
# # # # # # #     plt.close()

# # # # # # #     # Plot and save posterior for viscosity
# # # # # # #     plt.figure(figsize=(6, 4))
# # # # # # #     plt.hist(posterior_mu_original, bins=30, alpha=0.5, label='Original')
# # # # # # #     plt.hist(posterior_mu_standardized, bins=30, alpha=0.5, label='Standardized')
# # # # # # #     plt.hist(posterior_mu_normalized, bins=30, alpha=0.5, label='Normalized')
# # # # # # #     plt.title("Posterior Distribution of Viscosity")
# # # # # # #     plt.xlabel("Viscosity (Pa·s)")
# # # # # # #     plt.ylabel("Frequency")
# # # # # # #     plt.legend()
# # # # # # #     plt.tight_layout()
# # # # # # #     plt.savefig(f"{plot_folder}/posterior_viscosity.png", dpi=300)
# # # # # # #     plt.close()

# # # # # # #     # Plot and save posterior for flow rate
# # # # # # #     plt.figure(figsize=(6, 4))
# # # # # # #     plt.hist(posterior_Q_original, bins=30, alpha=0.5, label='Original')
# # # # # # #     plt.hist(posterior_Q_standardized, bins=30, alpha=0.5, label='Standardized')
# # # # # # #     plt.hist(posterior_Q_normalized, bins=30, alpha=0.5, label='Normalized')
# # # # # # #     plt.title("Posterior Distribution of Flow Rate")
# # # # # # #     plt.xlabel("Flow Rate (m³/s)")
# # # # # # #     plt.ylabel("Frequency")
# # # # # # #     plt.legend()
# # # # # # #     plt.tight_layout()
# # # # # # #     plt.savefig(f"{plot_folder}/posterior_flow_rate.png", dpi=300)
# # # # # # #     plt.close()


# # # # # # #     # Save results to CSV and pickle
# # # # # # #     folder_path = "blood_flow_analysis_results"
# # # # # # #     save_results_to_csv_and_pickle(posterior_r_original, posterior_mu_original, posterior_Q_original, alerts_original, folder_path)
# # # # # # #     save_results_to_csv_and_pickle(posterior_r_standardized, posterior_mu_standardized, posterior_Q_standardized, alerts_standardized, folder_path)
# # # # # # #     save_results_to_csv_and_pickle(posterior_r_normalized, posterior_mu_normalized, posterior_Q_normalized, alerts_normalized, folder_path)





# # # # # # import pymc as pm
# # # # # # import numpy as np
# # # # # # import matplotlib.pyplot as plt 
# # # # # # import os
# # # # # # import pandas as pd
# # # # # # from sklearn.preprocessing import StandardScaler, MinMaxScaler
# # # # # # from scipy.stats import gaussian_kde
# # # # # # from scipy.integrate import quad
# # # # # # import pickle

# # # # # # # -----------------------
# # # # # # # Sampled Parameters with Aggressive Dispersion
# # # # # # # -----------------------
# # # # # # np.random.seed(42)

# # # # # # # Increase variability for pressure, length, radius, and viscosity
# # # # # # delta_P = np.random.normal(loc=300, scale=300)         # Pressure drop (Pa)
# # # # # # L = np.random.normal(loc=0.02, scale=0.03)             # Length of artery (m)
# # # # # # true_r = np.random.normal(loc=0.0002, scale=0.0003)    # Radius (m)
# # # # # # true_mu = np.random.normal(loc=0.0035, scale=0.005)    # Viscosity (Pa·s)

# # # # # # # -----------------------
# # # # # # # Standardization and Normalization
# # # # # # # -----------------------
# # # # # # # Prepare data for scaling
# # # # # # parameters = np.array([[delta_P, L, true_r, true_mu]])

# # # # # # # Standardization (Z-score scaling)
# # # # # # scaler_standard = StandardScaler()
# # # # # # parameters_standard = scaler_standard.fit_transform(parameters)

# # # # # # # Normalization (Min-Max scaling to range [0, 1])
# # # # # # scaler_normal = MinMaxScaler()
# # # # # # parameters_normalized = scaler_normal.fit_transform(parameters)

# # # # # # # Unpack the scaled parameters for use in the model
# # # # # # delta_P_standard, L_standard, true_r_standard, true_mu_standard = parameters_standard[0]
# # # # # # delta_P_normalized, L_normalized, true_r_normalized, true_mu_normalized = parameters_normalized[0]

# # # # # # # -----------------------
# # # # # # # Core Functions
# # # # # # # -----------------------

# # # # # # def poiseuille_flow_vec(r, mu):
# # # # # #     """Vectorized Poiseuille's equation"""
# # # # # #     return (np.pi * np.power(r, 4) * delta_P) / (8 * mu * L)

# # # # # # def simulate_noisy_observations(Q_true, num_samples=30, noise_percent=1.0):
# # # # # #     """Generate noisy observations"""
# # # # # #     return np.random.normal(loc=Q_true, scale=Q_true * noise_percent, size=num_samples)

# # # # # # def analyze_posteriors(trace, Q_true):
# # # # # #     """Analyze the posterior distributions of r, mu, and Q"""
# # # # # #     posterior_r = trace.posterior["r"].values.flatten()
# # # # # #     posterior_mu = trace.posterior["mu"].values.flatten()
# # # # # #     posterior_Q = poiseuille_flow_vec(posterior_r, posterior_mu)

# # # # # #     # Compute alert thresholds for errors
# # # # # #     error = np.abs(posterior_Q - Q_true)
# # # # # #     alert_threshold = 0.05 * Q_true  # 5% error threshold
# # # # # #     alerts = error > alert_threshold

# # # # # #     return posterior_r, posterior_mu, posterior_Q, alerts

# # # # # # def build_and_run_model(Q_obs, Q_true, delta_P, L, r, mu):
# # # # # #     """Bayesian model using PyMC with high dispersion priors"""
# # # # # #     with pm.Model() as model:
# # # # # #         r = pm.HalfNormal("r", sigma=0.0035)  # Wide HalfNormal → allows near-zero
# # # # # #         mu = pm.Normal("mu", mu=0.0035, sigma=0.01)  # Wide Normal → high resistance

# # # # # #         Q_est = (np.pi * r**4 * delta_P) / (8 * mu * L)

# # # # # #         pm.Normal("Q_obs", mu=Q_est, sigma=Q_true * 1.0, observed=Q_obs)

# # # # # #         trace = pm.sample(
# # # # # #             draws=1000,
# # # # # #             tune=500,
# # # # # #             chains=4,
# # # # # #             cores=4,
# # # # # #             init="adapt_diag",
# # # # # #             target_accept=0.95,
# # # # # #             return_inferencedata=True,
# # # # # #             idata_kwargs={"log_likelihood": False},
# # # # # #             random_seed=42,
# # # # # #             sampler_kwargs={"nuts_sampler": "numpyro"}  # Faster backend
# # # # # #         )
# # # # # #     return trace

# # # # # # def save_results_to_csv_and_pickle(posterior_r, posterior_mu, posterior_Q, alerts, folder_path):
# # # # # #     """Save the results to CSV and pickle"""
# # # # # #     results = {
# # # # # #         "posterior_r": posterior_r,
# # # # # #         "posterior_mu": posterior_mu,
# # # # # #         "posterior_Q": posterior_Q,
# # # # # #         "alerts": alerts
# # # # # #     }
# # # # # #     # Save as pickle file
# # # # # #     pickle_path = os.path.join(folder_path, "results.pkl")
# # # # # #     with open(pickle_path, "wb") as f:
# # # # # #         pickle.dump(results, f)

# # # # # #     # Save as CSV
# # # # # #     df = pd.DataFrame(results)
# # # # # #     csv_path = os.path.join(folder_path, "results.csv")
# # # # # #     df.to_csv(csv_path, index=False)

# # # # # # def visualize_results(posterior_r, posterior_mu, posterior_Q, Q_true, true_r, true_mu, alerts, folder_path):
# # # # # #     """Visualize and save the results as plots"""
# # # # # #     # Plot 1: Posterior distribution of radius
# # # # # #     plt.figure(figsize=(10, 6))
# # # # # #     plt.hist(posterior_r, bins=30, density=True, alpha=0.6, color='g')
# # # # # #     plt.title('Posterior Distribution of Radius')
# # # # # #     plt.xlabel('Radius (m)')
# # # # # #     plt.ylabel('Density')
# # # # # #     plt.axvline(true_r, color='r', linestyle='--', label=f'True Radius: {true_r:.4f}')
# # # # # #     plt.legend()
# # # # # #     plt.savefig(os.path.join(folder_path, 'posterior_radius.png'))
# # # # # #     plt.close()

# # # # # #     # Plot 2: Posterior distribution of viscosity
# # # # # #     plt.figure(figsize=(10, 6))
# # # # # #     plt.hist(posterior_mu, bins=30, density=True, alpha=0.6, color='b')
# # # # # #     plt.title('Posterior Distribution of Viscosity')
# # # # # #     plt.xlabel('Viscosity (Pa·s)')
# # # # # #     plt.ylabel('Density')
# # # # # #     plt.axvline(true_mu, color='r', linestyle='--', label=f'True Viscosity: {true_mu:.4f}')
# # # # # #     plt.legend()
# # # # # #     plt.savefig(os.path.join(folder_path, 'posterior_viscosity.png'))
# # # # # #     plt.close()

# # # # # #     # Plot 3: Posterior distribution of flow rate
# # # # # #     plt.figure(figsize=(10, 6))
# # # # # #     plt.hist(posterior_Q, bins=30, density=True, alpha=0.6, color='orange')
# # # # # #     plt.title('Posterior Distribution of Flow Rate')
# # # # # #     plt.xlabel('Flow Rate (m^3/s)')
# # # # # #     plt.ylabel('Density')
# # # # # #     plt.axvline(Q_true, color='r', linestyle='--', label=f'True Flow Rate: {Q_true:.6f}')
# # # # # #     plt.legend()
# # # # # #     plt.savefig(os.path.join(folder_path, 'posterior_flow_rate.png'))
# # # # # #     plt.close()

# # # # # #     # Plot 4: Alerts visualization
# # # # # #     plt.figure(figsize=(10, 6))
# # # # # #     plt.plot(posterior_Q, color='purple', alpha=0.6, label='Posterior Flow Rate')
# # # # # #     plt.scatter(np.where(alerts)[0], posterior_Q[alerts], color='red', label='Alerts (Error > 5%)')
# # # # # #     plt.title('Flow Rate Posterior with Alerts')
# # # # # #     plt.xlabel('Sample')
# # # # # #     plt.ylabel('Flow Rate (m^3/s)')
# # # # # #     plt.legend()
# # # # # #     plt.savefig(os.path.join(folder_path, 'flow_rate_alerts.png'))
# # # # # #     plt.close()

# # # # # # # -----------------------
# # # # # # # Main Pipeline for Standardization and Normalization Effects
# # # # # # # -----------------------
# # # # # # if __name__ == "__main__":
# # # # # #     folder_path = "standardization_normalization_effects"

# # # # # #     # Calculate true flow rate with original values
# # # # # #     Q_true = poiseuille_flow_vec(true_r, true_mu)

# # # # # #     # Simulate noisy observations for true Q
# # # # # #     Q_obs = simulate_noisy_observations(Q_true)

# # # # # #     # Run the model for standardization (using standardized parameters)
# # # # # #     trace_standard = build_and_run_model(Q_obs, Q_true, delta_P_standard, L_standard, true_r_standard, true_mu_standard)

# # # # # #     # Run the model for normalization (using normalized parameters)
# # # # # #     trace_normalized = build_and_run_model(Q_obs, Q_true, delta_P_normalized, L_normalized, true_r_normalized, true_mu_normalized)

# # # # # #     # Analyze the posterior for both standardization and normalization effects
# # # # # #     posterior_r_standard, posterior_mu_standard, posterior_Q_standard, alerts_standard = analyze_posteriors(trace_standard, Q_true)
# # # # # #     posterior_r_normalized, posterior_mu_normalized, posterior_Q_normalized, alerts_normalized = analyze_posteriors(trace_normalized, Q_true)

# # # # # #     # Save results to CSV for both cases
# # # # # #     if not os.path.exists(folder_path):
# # # # # #         os.makedirs(folder_path)

# # # # # #     # Save results for both cases to CSV
# # # # # #     save_results_to_csv_and_pickle(posterior_r_standard, posterior_mu_standard, posterior_Q_standard, alerts_standard, folder_path)
# # # # # #     save_results_to_csv_and_pickle(posterior_r_normalized, posterior_mu_normalized, posterior_Q_normalized, alerts_normalized, folder_path)

# # # # # #     # Visualize and save plots for both standardization and normalization effects
# # # # # #     visualize_results(posterior_r_standard, posterior_mu_standard, posterior_Q_standard, Q_true, true_r, true_mu, alerts_standard, folder_path)
# # # # # #     visualize_results(posterior_r_normalized, posterior_mu_normalized, posterior_Q_normalized, Q_true, true_r, true_mu, alerts_normalized, folder_path)

# # # # # #     print(f"\nResults and plots saved to {folder_path}")


# # # # # import pymc as pm
# # # # # import numpy as np
# # # # # import matplotlib.pyplot as plt
# # # # # import os
# # # # # import pandas as pd
# # # # # from sklearn.preprocessing import StandardScaler, MinMaxScaler
# # # # # from scipy.stats import gaussian_kde
# # # # # from scipy.integrate import quad
# # # # # import pickle

# # # # # # -----------------------
# # # # # # Sampled Parameters with Aggressive Dispersion
# # # # # # -----------------------
# # # # # np.random.seed(42)

# # # # # # Increase variability for pressure, length, radius, and viscosity
# # # # # delta_P = np.random.normal(loc=300, scale=300)         # Pressure drop (Pa)
# # # # # L = np.random.normal(loc=0.02, scale=0.03)             # Length of artery (m)
# # # # # true_r = np.random.normal(loc=0.0002, scale=0.0003)    # Radius (m)
# # # # # true_mu = np.random.normal(loc=0.0035, scale=0.005)    # Viscosity (Pa·s)

# # # # # # -----------------------
# # # # # # Standardization and Normalization
# # # # # # -----------------------
# # # # # # Prepare data for scaling
# # # # # parameters = np.array([[delta_P, L, true_r, true_mu]])

# # # # # # Standardization (Z-score scaling)
# # # # # scaler_standard = StandardScaler()
# # # # # parameters_standard = scaler_standard.fit_transform(parameters)

# # # # # # Normalization (Min-Max scaling to range [0, 1])
# # # # # scaler_normal = MinMaxScaler()
# # # # # parameters_normalized = scaler_normal.fit_transform(parameters)

# # # # # # Unpack the scaled parameters for use in the model
# # # # # delta_P_standard, L_standard, true_r_standard, true_mu_standard = parameters_standard[0]
# # # # # delta_P_normalized, L_normalized, true_r_normalized, true_mu_normalized = parameters_normalized[0]

# # # # # # -----------------------
# # # # # # Core Functions
# # # # # # -----------------------

# # # # # def poiseuille_flow_vec(r, mu):
# # # # #     """Vectorized Poiseuille's equation"""
# # # # #     return (np.pi * np.power(r, 4) * delta_P) / (8 * mu * L)

# # # # # def simulate_noisy_observations(Q_true, num_samples=30, noise_percent=1.0):
# # # # #     """Generate noisy observations"""
# # # # #     return np.random.normal(loc=Q_true, scale=Q_true * noise_percent, size=num_samples)

# # # # # def analyze_posteriors(trace, Q_true):
# # # # #     """Analyze the posterior distributions of r, mu, and Q"""
# # # # #     posterior_r = trace.posterior["r"].values.flatten()
# # # # #     posterior_mu = trace.posterior["mu"].values.flatten()
# # # # #     posterior_Q = poiseuille_flow_vec(posterior_r, posterior_mu)

# # # # #     # Compute alert thresholds for errors
# # # # #     error = np.abs(posterior_Q - Q_true)
# # # # #     alert_threshold = 0.05 * Q_true  # 5% error threshold
# # # # #     alerts = error > alert_threshold

# # # # #     return posterior_r, posterior_mu, posterior_Q, alerts

# # # # # def build_and_run_model(Q_obs, Q_true, delta_P, L, r, mu):
# # # # #     """Bayesian model using PyMC with high dispersion priors"""
# # # # #     with pm.Model() as model:
# # # # #         r = pm.HalfNormal("r", sigma=0.0035)  # Wide HalfNormal → allows near-zero
# # # # #         mu = pm.Normal("mu", mu=0.0035, sigma=0.01)  # Wide Normal → high resistance

# # # # #         Q_est = (np.pi * r**4 * delta_P) / (8 * mu * L)

# # # # #         pm.Normal("Q_obs", mu=Q_est, sigma=Q_true * 1.0, observed=Q_obs)

# # # # #         trace = pm.sample(
# # # # #             draws=1000,
# # # # #             tune=500,
# # # # #             chains=4,
# # # # #             cores=4,
# # # # #             init="adapt_diag",
# # # # #             target_accept=0.95,
# # # # #             return_inferencedata=True,
# # # # #             idata_kwargs={"log_likelihood": False},
# # # # #             random_seed=42,
# # # # #             sampler_kwargs={"nuts_sampler": "numpyro"}  # Faster backend
# # # # #         )
# # # # #     return trace

# # # # # def save_results_to_csv_and_pickle(posterior_r, posterior_mu, posterior_Q, alerts, folder_path):
# # # # #     """Save the results to CSV and pickle"""
# # # # #     results = {
# # # # #         "posterior_r": posterior_r,
# # # # #         "posterior_mu": posterior_mu,
# # # # #         "posterior_Q": posterior_Q,
# # # # #         "alerts": alerts
# # # # #     }
# # # # #     # Save as pickle file
# # # # #     pickle_path = os.path.join(folder_path, "results.pkl")
# # # # #     with open(pickle_path, "wb") as f:
# # # # #         pickle.dump(results, f)

# # # # #     # Save as CSV
# # # # #     df = pd.DataFrame(results)
# # # # #     csv_path = os.path.join(folder_path, "results.csv")
# # # # #     df.to_csv(csv_path, index=False)

# # # # # def visualize_results(posterior_r_regular, posterior_mu_regular, posterior_Q_regular,
# # # # #                       posterior_r_standard, posterior_mu_standard, posterior_Q_standard,
# # # # #                       posterior_r_normalized, posterior_mu_normalized, posterior_Q_normalized,
# # # # #                       Q_true, true_r, true_mu, alerts_regular, alerts_standard, alerts_normalized, folder_path):
# # # # #     """Visualize and save the results as plots"""
# # # # #     # Plot 1: Posterior distribution of radius (overlapping)
# # # # #     plt.figure(figsize=(10, 6))
# # # # #     plt.hist(posterior_r_regular, bins=30, density=True, alpha=0.6, label='Regular', color='g')
# # # # #     plt.hist(posterior_r_standard, bins=30, density=True, alpha=0.6, label='Standardized', color='b')
# # # # #     plt.hist(posterior_r_normalized, bins=30, density=True, alpha=0.6, label='Normalized', color='orange')
# # # # #     plt.title('Posterior Distribution of Radius')
# # # # #     plt.xlabel('Radius (m)')
# # # # #     plt.ylabel('Density')
# # # # #     plt.axvline(true_r, color='r', linestyle='--', label=f'True Radius: {true_r:.4f}')
# # # # #     plt.legend()
# # # # #     plt.savefig(os.path.join(folder_path, 'posterior_radius_comparison.png'))
# # # # #     plt.close()

# # # # #     # Plot 2: Posterior distribution of viscosity (overlapping)
# # # # #     plt.figure(figsize=(10, 6))
# # # # #     plt.hist(posterior_mu_regular, bins=30, density=True, alpha=0.6, label='Regular', color='g')
# # # # #     plt.hist(posterior_mu_standard, bins=30, density=True, alpha=0.6, label='Standardized', color='b')
# # # # #     plt.hist(posterior_mu_normalized, bins=30, density=True, alpha=0.6, label='Normalized', color='orange')
# # # # #     plt.title('Posterior Distribution of Viscosity')
# # # # #     plt.xlabel('Viscosity (Pa·s)')
# # # # #     plt.ylabel('Density')
# # # # #     plt.axvline(true_mu, color='r', linestyle='--', label=f'True Viscosity: {true_mu:.4f}')
# # # # #     plt.legend()
# # # # #     plt.savefig(os.path.join(folder_path, 'posterior_viscosity_comparison.png'))
# # # # #     plt.close()

# # # # #     # Plot 3: Posterior distribution of flow rate (overlapping)
# # # # #     plt.figure(figsize=(10, 6))
# # # # #     plt.hist(posterior_Q_regular, bins=30, density=True, alpha=0.6, label='Regular', color='g')
# # # # #     plt.hist(posterior_Q_standard, bins=30, density=True, alpha=0.6, label='Standardized', color='b')
# # # # #     plt.hist(posterior_Q_normalized, bins=30, density=True, alpha=0.6, label='Normalized', color='orange')
# # # # #     plt.title('Posterior Distribution of Flow Rate')
# # # # #     plt.xlabel('Flow Rate (m^3/s)')
# # # # #     plt.ylabel('Density')
# # # # #     plt.axvline(Q_true, color='r', linestyle='--', label=f'True Flow Rate: {Q_true:.6f}')
# # # # #     plt.legend()
# # # # #     plt.savefig(os.path.join(folder_path, 'posterior_flow_rate_comparison.png'))
# # # # #     plt.close()

# # # # #     # Plot 4: Alerts visualization (overlapping)
# # # # #     plt.figure(figsize=(10, 6))
# # # # #     plt.plot(posterior_Q_regular, color='g', alpha=0.6, label='Regular')
# # # # #     plt.scatter(np.where(alerts_regular)[0], posterior_Q_regular[alerts_regular], color='red', label='Regular Alerts')
# # # # #     plt.plot(posterior_Q_standard, color='b', alpha=0.6, label='Standardized')
# # # # #     plt.scatter(np.where(alerts_standard)[0], posterior_Q_standard[alerts_standard], color='blue', label='Standardized Alerts')
# # # # #     plt.plot(posterior_Q_normalized, color='orange', alpha=0.6, label='Normalized')
# # # # #     plt.scatter(np.where(alerts_normalized)[0], posterior_Q_normalized[alerts_normalized], color='orange', label='Normalized Alerts')
# # # # #     plt.title('Flow Rate Posterior with Alerts')
# # # # #     plt.xlabel('Sample')
# # # # #     plt.ylabel('Flow Rate (m^3/s)')
# # # # #     plt.legend()
# # # # #     plt.savefig(os.path.join(folder_path, 'flow_rate_alerts_comparison.png'))
# # # # #     plt.close()

# # # # # # -----------------------
# # # # # # Main Pipeline for Standardization and Normalization Effects
# # # # # # -----------------------
# # # # # if __name__ == "__main__":
# # # # #     folder_path = "standardization_normalization_effects"

# # # # #     # Calculate true flow rate with original values
# # # # #     Q_true = poiseuille_flow_vec(true_r, true_mu)

# # # # #     # Simulate noisy observations for true Q
# # # # #     Q_obs = simulate_noisy_observations(Q_true)

# # # # #     # Run the model for regular (no scaling) case
# # # # #     trace_regular = build_and_run_model(Q_obs, Q_true, delta_P, L, true_r, true_mu)

# # # # #     # Run the model for standardization (using standardized parameters)
# # # # #     trace_standard = build_and_run_model(Q_obs, Q_true, delta_P_standard, L_standard, true_r_standard, true_mu_standard)

# # # # #     # Run the model for normalization (using normalized parameters)
# # # # #     trace_normalized = build_and_run_model(Q_obs, Q_true, delta_P_normalized, L_normalized, true_r_normalized, true_mu_normalized)

# # # # #     # Analyze the posterior for all three cases
# # # # #     posterior_r_regular, posterior_mu_regular, posterior_Q_regular, alerts_regular = analyze_posteriors(trace_regular, Q_true)
# # # # #     posterior_r_standard, posterior_mu_standard, posterior_Q_standard, alerts_standard = analyze_posteriors(trace_standard, Q_true)
# # # # #     posterior_r_normalized, posterior_mu_normalized, posterior_Q_normalized, alerts_normalized = analyze_posteriors(trace_normalized, Q_true)

# # # # #     # Save results to CSV for all cases
# # # # #     if not os.path.exists(folder_path):
# # # # #         os.makedirs(folder_path)

# # # # #     # Save results for all cases to CSV
# # # # #     save_results_to_csv_and_pickle(posterior_r_regular, posterior_mu_regular, posterior_Q_regular, alerts_regular, folder_path)
# # # # #     save_results_to_csv_and_pickle(posterior_r_standard, posterior_mu_standard, posterior_Q_standard, alerts_standard, folder_path)
# # # # #     save_results_to_csv_and_pickle(posterior_r_normalized, posterior_mu_normalized, posterior_Q_normalized, alerts_normalized, folder_path)

# # # # #     # Visualize and save plots for all cases
# # # # #     visualize_results(posterior_r_regular, posterior_mu_regular, posterior_Q_regular,
# # # # #                       posterior_r_standard, posterior_mu_standard, posterior_Q_standard,
# # # # #                       posterior_r_normalized, posterior_mu_normalized, posterior_Q_normalized,
# # # # #                       Q_true, true_r, true_mu, alerts_regular, alerts_standard, alerts_normalized, folder_path)

# # # # #     print(f"\nResults and plots saved to {folder_path}")







# # # # # import pymc as pm
# # # # # import numpy as np
# # # # # import matplotlib.pyplot as plt 
# # # # # from scipy.stats import gaussian_kde
# # # # # from scipy.integrate import quad
# # # # # import os
# # # # # import pandas as pd
# # # # # from sklearn.preprocessing import StandardScaler, MinMaxScaler
# # # # # from utils import save_results_to_csv_and_pickle, visualize_results

# # # # # # -----------------------
# # # # # # Sampled Parameters with Aggressive Dispersion
# # # # # # -----------------------
# # # # # np.random.seed(42)

# # # # # # Increase variability for pressure, length, radius, and viscosity
# # # # # delta_P = np.random.normal(loc=300, scale=300)         # Pressure drop (Pa)
# # # # # L = np.random.normal(loc=0.02, scale=0.03)             # Length of artery (m)
# # # # # true_r = np.random.normal(loc=0.0002, scale=0.0003)    # Radius (m) - small radius possible
# # # # # true_mu = np.random.normal(loc=0.0035, scale=0.005)    # Viscosity (Pa·s) - high resistance possible


# # # # # # -----------------------
# # # # # # Core Functions
# # # # # # -----------------------

# # # # # def poiseuille_flow_vec(r, mu):
# # # # #     """Vectorized Poiseuille's equation"""
# # # # #     return (np.pi * np.power(r, 4) * delta_P) / (8 * mu * L)

# # # # # def simulate_noisy_observations(Q_true, num_samples=30, noise_percent=1.0):
# # # # #     """Generate noisy observations"""
# # # # #     return np.random.normal(loc=Q_true, scale=Q_true * noise_percent, size=num_samples)

# # # # # def build_and_run_model(Q_obs, Q_true):
# # # # #     """Bayesian model using PyMC with high dispersion priors"""
# # # # #     with pm.Model() as model:
# # # # #         r = pm.HalfNormal("r", sigma=0.0035)            # Wide HalfNormal → allows near-zero
# # # # #         mu = pm.Normal("mu", mu=0.0035, sigma=0.01)     # Wide Normal → high resistance

# # # # #         Q_est = (np.pi * r**4 * delta_P) / (8 * mu * L)

# # # # #         pm.Normal("Q_obs", mu=Q_est, sigma=Q_true * 1.0, observed=Q_obs)

# # # # #         trace = pm.sample(
# # # # #             draws=500,
# # # # #             tune=100,
# # # # #             chains=4,
# # # # #             cores=4,
# # # # #             init="adapt_diag",
# # # # #             target_accept=0.95,
# # # # #             return_inferencedata=True,
# # # # #             idata_kwargs={"log_likelihood": False},
# # # # #             random_seed=42,
# # # # #             sampler_kwargs={"nuts_sampler": "numpyro"}  # Faster backend
# # # # #         )
# # # # #     return trace

# # # # # def analyze_posteriors(trace, Q_true):
# # # # #     """Analyze posterior samples and detect abnormalities"""
# # # # #     posterior_r = trace.posterior["r"].stack(samples=("chain", "draw")).values
# # # # #     posterior_mu = trace.posterior["mu"].stack(samples=("chain", "draw")).values
# # # # #     posterior_Q = poiseuille_flow_vec(posterior_r, posterior_mu)

# # # # #     normal_lower = Q_true * 0.9
# # # # #     normal_upper = Q_true * 1.1

# # # # #     alerts = np.where(
# # # # #         posterior_Q < normal_lower * 0.3,
# # # # #         "Blockage",
# # # # #         np.where(
# # # # #             posterior_Q < normal_lower,
# # # # #             "Vasoconstriction",
# # # # #             np.where(
# # # # #                 posterior_Q > normal_upper,
# # # # #                 "Vasodilation",
# # # # #                 "Normal"
# # # # #             )
# # # # #         )
# # # # #     )
# # # # #     return posterior_r, posterior_mu, posterior_Q, alerts

# # # # # # -----------------------
# # # # # # Main Pipeline
# # # # # # -----------------------
# # # # # if __name__ == "__main__":
# # # # #     folder_path = "standardization_normalization_effects"

# # # # #     # Create the folder for results if it doesn't exist
# # # # #     os.makedirs(folder_path, exist_ok=True)

# # # # #     Q_true = poiseuille_flow_vec(true_r, true_mu)
# # # # #     Q_obs = simulate_noisy_observations(Q_true)

# # # # #     # Apply Standardization (Z-score)
# # # # #     scaler_standard = StandardScaler()
# # # # #     Q_obs_standardized = scaler_standard.fit_transform(Q_obs.reshape(-1, 1)).flatten()

# # # # #     # Apply Normalization (Min-Max)
# # # # #     scaler_normalize = MinMaxScaler()
# # # # #     Q_obs_normalized = scaler_normalize.fit_transform(Q_obs.reshape(-1, 1)).flatten()

# # # # #     # Run model for Regular, Standardized, and Normalized Observations
# # # # #     trace_regular = build_and_run_model(Q_obs, Q_true)
# # # # #     trace_standardized = build_and_run_model(Q_obs_standardized, Q_true)
# # # # #     trace_normalized = build_and_run_model(Q_obs_normalized, Q_true)

# # # # #     # Analyze Posteriors for all cases
# # # # #     posterior_r_regular, posterior_mu_regular, posterior_Q_regular, _ = analyze_posteriors(trace_regular, Q_true)
# # # # #     posterior_r_standardized, posterior_mu_standardized, posterior_Q_standardized, _ = analyze_posteriors(trace_standardized, Q_true)
# # # # #     posterior_r_normalized, posterior_mu_normalized, posterior_Q_normalized, _ = analyze_posteriors(trace_normalized, Q_true)

# # # # #     # Create Plots for Radius, Mu, and Flowrate Comparison
# # # # #     fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# # # # #     # Radius Plot
# # # # #     axs[0].hist(posterior_r_regular, bins=50, alpha=0.5, label="Regular")
# # # # #     axs[0].hist(posterior_r_standardized, bins=50, alpha=0.5, label="Standardized")
# # # # #     axs[0].hist(posterior_r_normalized, bins=50, alpha=0.5, label="Normalized")
# # # # #     axs[0].set_title("Posterior Distribution of Radius")
# # # # #     axs[0].legend()

# # # # #     # Viscosity Plot
# # # # #     axs[1].hist(posterior_mu_regular, bins=50, alpha=0.5, label="Regular")
# # # # #     axs[1].hist(posterior_mu_standardized, bins=50, alpha=0.5, label="Standardized")
# # # # #     axs[1].hist(posterior_mu_normalized, bins=50, alpha=0.5, label="Normalized")
# # # # #     axs[1].set_title("Posterior Distribution of Viscosity")
# # # # #     axs[1].legend()

# # # # #     # Flowrate Plot
# # # # #     axs[2].hist(posterior_Q_regular, bins=50, alpha=0.5, label="Regular")
# # # # #     axs[2].hist(posterior_Q_standardized, bins=50, alpha=0.5, label="Standardized")
# # # # #     axs[2].hist(posterior_Q_normalized, bins=50, alpha=0.5, label="Normalized")
# # # # #     axs[2].set_title("Posterior Distribution of Flowrate")
# # # # #     axs[2].legend()

# # # # #     # Save the plot
# # # # #     plot_filename = os.path.join(folder_path, "comparison_standardization_normalization.png")
# # # # #     plt.tight_layout()
# # # # #     plt.savefig(plot_filename)
# # # # #     plt.close()

# # # # #     # Save results to CSV
# # # # #     results_df = pd.DataFrame({
# # # # #         "Radius (Regular)": posterior_r_regular,
# # # # #         "Radius (Standardized)": posterior_r_standardized,
# # # # #         "Radius (Normalized)": posterior_r_normalized,
# # # # #         "Mu (Regular)": posterior_mu_regular,
# # # # #         "Mu (Standardized)": posterior_mu_standardized,
# # # # #         "Mu (Normalized)": posterior_mu_normalized,
# # # # #         "Flowrate (Regular)": posterior_Q_regular,
# # # # #         "Flowrate (Standardized)": posterior_Q_standardized,
# # # # #         "Flowrate (Normalized)": posterior_Q_normalized
# # # # #     })
# # # # #     results_df.to_csv(os.path.join(folder_path, "posterior_comparison_results.csv"), index=False)

# # # # #     # Save Pickle for Future Use
# # # # #     save_results_to_csv_and_pickle(posterior_r_regular, posterior_mu_regular, posterior_Q_regular, folder_path)
# # # # #     save_results_to_csv_and_pickle(posterior_r_standardized, posterior_mu_standardized, posterior_Q_standardized, folder_path)
# # # # #     save_results_to_csv_and_pickle(posterior_r_normalized, posterior_mu_normalized, posterior_Q_normalized, folder_path)

# # # # #     # Visualize the Results with the Utility
# # # # #     visualize_results(posterior_r_regular, posterior_mu_regular, posterior_Q_regular, Q_true, true_r, true_mu, folder_path)
# # # # #     visualize_results(posterior_r_standardized, posterior_mu_standardized, posterior_Q_standardized, Q_true, true_r, true_mu, folder_path)
# # # # #     visualize_results(posterior_r_normalized, posterior_mu_normalized, posterior_Q_normalized, Q_true, true_r, true_mu, folder_path)




# # # # import pymc as pm
# # # # import numpy as np
# # # # import matplotlib.pyplot as plt 
# # # # import os
# # # # import pandas as pd
# # # # from sklearn.preprocessing import StandardScaler, MinMaxScaler

# # # # # -----------------------
# # # # # Sampled Parameters with Aggressive Dispersion
# # # # # -----------------------
# # # # np.random.seed(42)

# # # # # Increase variability for pressure, length, radius, and viscosity
# # # # delta_P = np.random.normal(loc=300, scale=300)         # Pressure drop (Pa)
# # # # L = np.random.normal(loc=0.02, scale=0.03)             # Length of artery (m)
# # # # true_r = np.random.normal(loc=0.0002, scale=0.0003)    # Radius (m) - small radius possible
# # # # true_mu = np.random.normal(loc=0.0035, scale=0.005)    # Viscosity (Pa·s) - high resistance possible


# # # # # -----------------------
# # # # # Core Functions
# # # # # -----------------------

# # # # def poiseuille_flow_vec(r, mu):
# # # #     """Vectorized Poiseuille's equation"""
# # # #     return (np.pi * np.power(r, 4) * delta_P) / (8 * mu * L)

# # # # def simulate_noisy_observations(Q_true, num_samples=30, noise_percent=1.0):
# # # #     """Generate noisy observations"""
# # # #     return np.random.normal(loc=Q_true, scale=Q_true * noise_percent, size=num_samples)

# # # # def build_and_run_model(Q_obs, Q_true):
# # # #     """Simplified Bayesian model using PyMC with optimized priors"""
# # # #     with pm.Model() as model:
# # # #         # Priors for r and mu with tighter distributions for faster sampling
# # # #         r = pm.HalfNormal("r", sigma=0.001)            # Narrower prior
# # # #         mu = pm.Normal("mu", mu=0.0035, sigma=0.005)   # Normal prior with smaller sigma

# # # #         # Flowrate model
# # # #         Q_est = (np.pi * r**4 * delta_P) / (8 * mu * L)

# # # #         # Observation model
# # # #         pm.Normal("Q_obs", mu=Q_est, sigma=Q_true * 0.5, observed=Q_obs)  # Reduced noise for faster convergence

# # # #         # Sampling with fewer draws and chains for faster execution
# # # #         trace = pm.sample(
# # # #             draws=200,  # Fewer draws for faster results
# # # #             tune=50,    # Fewer tuning steps
# # # #             chains=2,   # Reduced number of chains
# # # #             cores=2,    # Use fewer cores for faster sampling
# # # #             init="adapt_diag",
# # # #             target_accept=0.95,
# # # #             return_inferencedata=True,
# # # #             random_seed=42,
# # # #         )
# # # #     return trace

# # # # def analyze_posteriors(trace, Q_true):
# # # #     """Analyze posterior samples and detect abnormalities"""
# # # #     posterior_r = trace.posterior["r"].stack(samples=("chain", "draw")).values
# # # #     posterior_mu = trace.posterior["mu"].stack(samples=("chain", "draw")).values
# # # #     posterior_Q = poiseuille_flow_vec(posterior_r, posterior_mu)

# # # #     normal_lower = Q_true * 0.9
# # # #     normal_upper = Q_true * 1.1

# # # #     # Anomaly detection
# # # #     alerts = np.where(
# # # #         posterior_Q < normal_lower * 0.3, "Blockage",
# # # #         np.where(posterior_Q < normal_lower, "Vasoconstriction",
# # # #                  np.where(posterior_Q > normal_upper, "Vasodilation", "Normal"))
# # # #     )
# # # #     return posterior_r, posterior_mu, posterior_Q, alerts

# # # # # -----------------------
# # # # # Main Pipeline
# # # # # -----------------------
# # # # if __name__ == "__main__":
# # # #     folder_path = "standardization_normalization_effects"

# # # #     # Create the folder for results if it doesn't exist
# # # #     os.makedirs(folder_path, exist_ok=True)

# # # #     Q_true = poiseuille_flow_vec(true_r, true_mu)
# # # #     Q_obs = simulate_noisy_observations(Q_true)

# # # #     # Apply Standardization (Z-score)
# # # #     scaler_standard = StandardScaler()
# # # #     Q_obs_standardized = scaler_standard.fit_transform(Q_obs.reshape(-1, 1)).flatten()

# # # #     # Apply Normalization (Min-Max)
# # # #     scaler_normalize = MinMaxScaler()
# # # #     Q_obs_normalized = scaler_normalize.fit_transform(Q_obs.reshape(-1, 1)).flatten()

# # # #     # Run model for Regular, Standardized, and Normalized Observations
# # # #     trace_regular = build_and_run_model(Q_obs, Q_true)
# # # #     trace_standardized = build_and_run_model(Q_obs_standardized, Q_true)
# # # #     trace_normalized = build_and_run_model(Q_obs_normalized, Q_true)

# # # #     # Analyze Posteriors for all cases
# # # #     posterior_r_regular, posterior_mu_regular, posterior_Q_regular, _ = analyze_posteriors(trace_regular, Q_true)
# # # #     posterior_r_standardized, posterior_mu_standardized, posterior_Q_standardized, _ = analyze_posteriors(trace_standardized, Q_true)
# # # #     posterior_r_normalized, posterior_mu_normalized, posterior_Q_normalized, _ = analyze_posteriors(trace_normalized, Q_true)

# # # #     # Create Plots for Radius, Mu, and Flowrate Comparison
# # # #     fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# # # #     # Radius Plot
# # # #     axs[0].hist(posterior_r_regular, bins=50, alpha=0.5, label="Regular")
# # # #     axs[0].hist(posterior_r_standardized, bins=50, alpha=0.5, label="Standardized")
# # # #     axs[0].hist(posterior_r_normalized, bins=50, alpha=0.5, label="Normalized")
# # # #     axs[0].set_title("Posterior Distribution of Radius")
# # # #     axs[0].legend()

# # # #     # Viscosity Plot
# # # #     axs[1].hist(posterior_mu_regular, bins=50, alpha=0.5, label="Regular")
# # # #     axs[1].hist(posterior_mu_standardized, bins=50, alpha=0.5, label="Standardized")
# # # #     axs[1].hist(posterior_mu_normalized, bins=50, alpha=0.5, label="Normalized")
# # # #     axs[1].set_title("Posterior Distribution of Viscosity")
# # # #     axs[1].legend()

# # # #     # Flowrate Plot
# # # #     axs[2].hist(posterior_Q_regular, bins=50, alpha=0.5, label="Regular")
# # # #     axs[2].hist(posterior_Q_standardized, bins=50, alpha=0.5, label="Standardized")
# # # #     axs[2].hist(posterior_Q_normalized, bins=50, alpha=0.5, label="Normalized")
# # # #     axs[2].set_title("Posterior Distribution of Flowrate")
# # # #     axs[2].legend()

# # # #     # Save the plot
# # # #     plot_filename = os.path.join(folder_path, "comparison_standardization_normalization.png")
# # # #     plt.tight_layout()
# # # #     plt.savefig(plot_filename)
# # # #     plt.close()

# # # #     # Save results to CSV
# # # #     results_df = pd.DataFrame({
# # # #         "Radius (Regular)": posterior_r_regular,
# # # #         "Radius (Standardized)": posterior_r_standardized,
# # # #         "Radius (Normalized)": posterior_r_normalized,
# # # #         "Mu (Regular)": posterior_mu_regular,
# # # #         "Mu (Standardized)": posterior_mu_standardized,
# # # #         "Mu (Normalized)": posterior_mu_normalized,
# # # #         "Flowrate (Regular)": posterior_Q_regular,
# # # #         "Flowrate (Standardized)": posterior_Q_standardized,
# # # #         "Flowrate (Normalized)": posterior_Q_normalized
# # # #     })
# # # #     results_df.to_csv(os.path.join(folder_path, "posterior_comparison_results.csv"), index=False)

# # # #     # Save Pickle for Future Use
# # # #     # Implement the save function if needed





# # # import pymc as pm
# # # import numpy as np
# # # import matplotlib.pyplot as plt 
# # # import os
# # # import pandas as pd
# # # from sklearn.preprocessing import StandardScaler, MinMaxScaler

# # # # -----------------------
# # # # Sampled Parameters with Aggressive Dispersion
# # # # -----------------------
# # # np.random.seed(42)

# # # # Increase variability for pressure, length, radius, and viscosity
# # # delta_P = np.random.normal(loc=300, scale=300)         # Pressure drop (Pa)
# # # L = np.random.normal(loc=0.02, scale=0.03)             # Length of artery (m)
# # # true_r = np.random.normal(loc=0.0002, scale=0.0003)    # Radius (m) - small radius possible
# # # true_mu = np.random.normal(loc=0.0035, scale=0.005)    # Viscosity (Pa·s) - high resistance possible


# # # # -----------------------
# # # # Core Functions
# # # # -----------------------

# # # def poiseuille_flow_vec(r, mu):
# # #     """Vectorized Poiseuille's equation"""
# # #     return (np.pi * np.power(r, 4) * delta_P) / (8 * mu * L)

# # # def simulate_noisy_observations(Q_true, num_samples=30, noise_percent=0.2):
# # #     """Generate noisy observations with reduced noise level"""
# # #     return np.random.normal(loc=Q_true, scale=Q_true * noise_percent, size=num_samples)

# # # def build_and_run_model(Q_obs, Q_true):
# # #     """Simplified Bayesian model using PyMC with optimized priors"""
# # #     with pm.Model() as model:
# # #         # Priors for r and mu with looser distributions for better exploration
# # #         r = pm.HalfNormal("r", sigma=0.005)           # Wider prior
# # #         mu = pm.Normal("mu", mu=0.0035, sigma=0.01)   # Wider prior with more flexibility

# # #         # Flowrate model
# # #         Q_est = (np.pi * r**4 * delta_P) / (8 * mu * L)

# # #         # Observation model
# # #         pm.Normal("Q_obs", mu=Q_est, sigma=Q_true * 0.2, observed=Q_obs)  # Reduced noise for faster convergence

# # #         # Sampling with increased draws and chains for better exploration
# # #         trace = pm.sample(
# # #             draws=1000,  # More draws for better precision
# # #             tune=500,    # Increased tuning steps
# # #             chains=4,    # More chains to ensure better convergence
# # #             cores=2,     # Use fewer cores for faster sampling
# # #             init="adapt_diag",
# # #             target_accept=0.95,
# # #             return_inferencedata=True,
# # #             random_seed=42,
# # #         )
# # #     return trace

# # # def analyze_posteriors(trace, Q_true):
# # #     """Analyze posterior samples and detect abnormalities"""
# # #     posterior_r = trace.posterior["r"].stack(samples=("chain", "draw")).values
# # #     posterior_mu = trace.posterior["mu"].stack(samples=("chain", "draw")).values
# # #     posterior_Q = poiseuille_flow_vec(posterior_r, posterior_mu)

# # #     normal_lower = Q_true * 0.9
# # #     normal_upper = Q_true * 1.1

# # #     # Anomaly detection
# # #     alerts = np.where(
# # #         posterior_Q < normal_lower * 0.3, "Blockage",
# # #         np.where(posterior_Q < normal_lower, "Vasoconstriction",
# # #                  np.where(posterior_Q > normal_upper, "Vasodilation", "Normal"))
# # #     )
# # #     return posterior_r, posterior_mu, posterior_Q, alerts

# # # # -----------------------
# # # # Main Pipeline
# # # # -----------------------
# # # if __name__ == "__main__":
# # #     folder_path = "standardization_normalization_effects"

# # #     # Create the folder for results if it doesn't exist
# # #     os.makedirs(folder_path, exist_ok=True)

# # #     # Create subfolders for results and plots
# # #     results_folder = os.path.join(folder_path, "results")
# # #     plots_folder = os.path.join(folder_path, "plots")
# # #     os.makedirs(results_folder, exist_ok=True)
# # #     os.makedirs(plots_folder, exist_ok=True)

# # #     Q_true = poiseuille_flow_vec(true_r, true_mu)
# # #     Q_obs = simulate_noisy_observations(Q_true)

# # #     # Apply Standardization (Z-score)
# # #     scaler_standard = StandardScaler()
# # #     Q_obs_standardized = scaler_standard.fit_transform(Q_obs.reshape(-1, 1)).flatten()

# # #     # Apply Normalization (Min-Max)
# # #     scaler_normalize = MinMaxScaler()
# # #     Q_obs_normalized = scaler_normalize.fit_transform(Q_obs.reshape(-1, 1)).flatten()

# # #     # Run model for Regular, Standardized, and Normalized Observations
# # #     trace_regular = build_and_run_model(Q_obs, Q_true)
# # #     trace_standardized = build_and_run_model(Q_obs_standardized, Q_true)
# # #     trace_normalized = build_and_run_model(Q_obs_normalized, Q_true)

# # #     # Analyze Posteriors for all cases
# # #     posterior_r_regular, posterior_mu_regular, posterior_Q_regular, _ = analyze_posteriors(trace_regular, Q_true)
# # #     posterior_r_standardized, posterior_mu_standardized, posterior_Q_standardized, _ = analyze_posteriors(trace_standardized, Q_true)
# # #     posterior_r_normalized, posterior_mu_normalized, posterior_Q_normalized, _ = analyze_posteriors(trace_normalized, Q_true)

# # #     # Create Plots for Radius, Mu, and Flowrate Comparison
# # #     fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# # #     # Radius Plot
# # #     axs[0].hist(posterior_r_regular, bins=50, alpha=0.5, label="Regular")
# # #     axs[0].hist(posterior_r_standardized, bins=50, alpha=0.5, label="Standardized")
# # #     axs[0].hist(posterior_r_normalized, bins=50, alpha=0.5, label="Normalized")
# # #     axs[0].set_title("Posterior Distribution of Radius")
# # #     axs[0].legend()

# # #     # Viscosity Plot
# # #     axs[1].hist(posterior_mu_regular, bins=50, alpha=0.5, label="Regular")
# # #     axs[1].hist(posterior_mu_standardized, bins=50, alpha=0.5, label="Standardized")
# # #     axs[1].hist(posterior_mu_normalized, bins=50, alpha=0.5, label="Normalized")
# # #     axs[1].set_title("Posterior Distribution of Viscosity")
# # #     axs[1].legend()

# # #     # Flowrate Plot
# # #     axs[2].hist(posterior_Q_regular, bins=50, alpha=0.5, label="Regular")
# # #     axs[2].hist(posterior_Q_standardized, bins=50, alpha=0.5, label="Standardized")
# # #     axs[2].hist(posterior_Q_normalized, bins=50, alpha=0.5, label="Normalized")
# # #     axs[2].set_title("Posterior Distribution of Flowrate")
# # #     axs[2].legend()

# # #     # Save the plot in the plots folder
# # #     plot_filename = os.path.join(plots_folder, "comparison_standardization_normalization.png")
# # #     plt.tight_layout()
# # #     plt.savefig(plot_filename)
# # #     plt.close()

# # #     # Save results to CSV in the results folder
# # #     results_df = pd.DataFrame({
# # #         "Radius (Regular)": posterior_r_regular,
# # #         "Radius (Standardized)": posterior_r_standardized,
# # #         "Radius (Normalized)": posterior_r_normalized,
# # #         "Mu (Regular)": posterior_mu_regular,
# # #         "Mu (Standardized)": posterior_mu_standardized,
# # #         "Mu (Normalized)": posterior_mu_normalized,
# # #         "Flowrate (Regular)": posterior_Q_regular,
# # #         "Flowrate (Standardized)": posterior_Q_standardized,
# # #         "Flowrate (Normalized)": posterior_Q_normalized
# # #     })
# # #     results_df.to_csv(os.path.join(results_folder, "posterior_comparison_results.csv"), index=False)








# # import pymc as pm
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import os
# # import pandas as pd
# # from sklearn.preprocessing import StandardScaler, MinMaxScaler

# # # Simulate basic parameters
# # np.random.seed(42)

# # # True parameters
# # delta_P = np.random.normal(loc=300, scale=300)  # Pressure drop (Pa)
# # L = np.random.normal(loc=0.02, scale=0.03)      # Length of artery (m)
# # true_r = np.random.normal(loc=0.0002, scale=0.0003)  # Radius (m)
# # true_mu = np.random.normal(loc=0.0035, scale=0.005)  # Viscosity (Pa·s)

# # # Poiseuille's equation for flowrate
# # def poiseuille_flow(r, mu):
# #     return (np.pi * r**4 * delta_P) / (8 * mu * L)

# # # Simulate noisy observations
# # def simulate_noisy_observations(Q_true, num_samples=30, noise_percent=0.2):
# #     return np.random.normal(loc=Q_true, scale=Q_true * noise_percent, size=num_samples)

# # # Standardization (Z-score normalization)
# # scaler_standard = StandardScaler()
# # scaler_normalize = MinMaxScaler()

# # # Generate true flowrate and noisy observations
# # Q_true = poiseuille_flow(true_r, true_mu)
# # Q_obs = simulate_noisy_observations(Q_true)

# # # Apply Standardization and Normalization
# # Q_obs_standardized = scaler_standard.fit_transform(Q_obs.reshape(-1, 1)).flatten()
# # Q_obs_normalized = scaler_normalize.fit_transform(Q_obs.reshape(-1, 1)).flatten()

# # # Function to run Bayesian model
# # def build_and_run_model(Q_obs_transformed, Q_true):
# #     with pm.Model() as model:
# #         # Define priors for radius and viscosity
# #         r = pm.HalfNormal("r", sigma=0.0035)
# #         mu = pm.Normal("mu", mu=0.0035, sigma=0.01)
        
# #         # Flowrate model
# #         Q_est = (np.pi * r**4 * delta_P) / (8 * mu * L)

# #         # Likelihood with noisy observations
# #         pm.Normal("Q_obs", mu=Q_est, sigma=Q_true * 0.2, observed=Q_obs_transformed)

# #         # Sampling using MCMC
# #         trace = pm.sample(
# #             draws=500,
# #             tune=100,
# #             chains=4,
# #             cores=4,
# #             init="adapt_diag",
# #             target_accept=0.95,
# #             return_inferencedata=True,
# #             random_seed=42
# #         )
# #     return trace

# # # Main function that will be called when the script runs
# # def main():
# #     # Run the model for original, standardized, and normalized observations
# #     trace_original = build_and_run_model(Q_obs, Q_true)
# #     trace_standardized = build_and_run_model(Q_obs_standardized, Q_true)
# #     trace_normalized = build_and_run_model(Q_obs_normalized, Q_true)

# #     # Extract posterior samples for radius and viscosity
# #     posterior_r_original = trace_original.posterior["r"].stack(samples=("chain", "draw")).values
# #     posterior_mu_original = trace_original.posterior["mu"].stack(samples=("chain", "draw")).values

# #     posterior_r_standardized = trace_standardized.posterior["r"].stack(samples=("chain", "draw")).values
# #     posterior_mu_standardized = trace_standardized.posterior["mu"].stack(samples=("chain", "draw")).values

# #     posterior_r_normalized = trace_normalized.posterior["r"].stack(samples=("chain", "draw")).values
# #     posterior_mu_normalized = trace_normalized.posterior["mu"].stack(samples=("chain", "draw")).values

# #     # Set folder structure for saving results and plots
# #     folder_path = "scaling_comparison_effects"
# #     os.makedirs(folder_path, exist_ok=True)
# #     results_folder = os.path.join(folder_path, "results")
# #     os.makedirs(results_folder, exist_ok=True)
# #     plots_folder = os.path.join(folder_path, "plots")
# #     os.makedirs(plots_folder, exist_ok=True)

# #     # Save results to CSV
# #     results_df = pd.DataFrame({
# #         "Scaling Method": ["Original", "Standardized", "Normalized"],
# #         "Mean Radius": [
# #             np.mean(posterior_r_original),
# #             np.mean(posterior_r_standardized),
# #             np.mean(posterior_r_normalized)
# #         ],
# #         "Mean Viscosity": [
# #             np.mean(posterior_mu_original),
# #             np.mean(posterior_mu_standardized),
# #             np.mean(posterior_mu_normalized)
# #         ]
# #     })

# #     # Save the results CSV file
# #     csv_filename = os.path.join(results_folder, "scaling_comparison_results.csv")
# #     results_df.to_csv(csv_filename, index=False)

# #     # Plotting Radius and Viscosity for Original, Standardized, and Normalized
# #     fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# #     # Radius Plot
# #     axs[0].hist(posterior_r_original, bins=50, alpha=0.5, label="Original", color='blue')
# #     axs[0].hist(posterior_r_standardized, bins=50, alpha=0.5, label="Standardized", color='green')
# #     axs[0].hist(posterior_r_normalized, bins=50, alpha=0.5, label="Normalized", color='red')
# #     axs[0].set_title("Posterior Distribution of Radius")
# #     axs[0].legend()

# #     # Viscosity Plot
# #     axs[1].hist(posterior_mu_original, bins=50, alpha=0.5, label="Original", color='blue')
# #     axs[1].hist(posterior_mu_standardized, bins=50, alpha=0.5, label="Standardized", color='green')
# #     axs[1].hist(posterior_mu_normalized, bins=50, alpha=0.5, label="Normalized", color='red')
# #     axs[1].set_title("Posterior Distribution of Viscosity")
# #     axs[1].legend()

# #     # Save the plot
# #     plot_filename = os.path.join(plots_folder, "comparison_all_runs.png")
# #     plt.tight_layout()
# #     plt.savefig(plot_filename)
# #     plt.close()

# #     print("Results saved to CSV and plots saved successfully.")

# # # Call the main function only when the script is run directly
# # if __name__ == "__main__":
# #     main()




# import pymc as pm
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import pandas as pd
# from sklearn.preprocessing import StandardScaler, MinMaxScaler

# # Simulate basic parameters
# np.random.seed(42)

# # True parameters
# delta_P = np.random.normal(loc=300, scale=300)  # Pressure drop (Pa)
# L = np.random.normal(loc=0.02, scale=0.03)      # Length of artery (m)
# true_r = np.random.normal(loc=0.0002, scale=0.0003)  # Radius (m)
# true_mu = np.random.normal(loc=0.0035, scale=0.005)  # Viscosity (Pa·s)

# # Poiseuille's equation for flowrate
# def poiseuille_flow(r, mu):
#     return (np.pi * r**4 * delta_P) / (8 * mu * L)

# # Simulate noisy observations
# def simulate_noisy_observations(Q_true, num_samples=10, noise_percent=0.2):
#     return np.random.normal(loc=Q_true, scale=Q_true * noise_percent, size=num_samples)

# # Standardization (Z-score normalization)
# scaler_standard = StandardScaler()
# scaler_normalize = MinMaxScaler()

# # Generate true flowrate and noisy observations
# Q_true = poiseuille_flow(true_r, true_mu)
# Q_obs = simulate_noisy_observations(Q_true)

# # Apply Standardization and Normalization
# Q_obs_standardized = scaler_standard.fit_transform(Q_obs.reshape(-1, 1)).flatten()
# Q_obs_normalized = scaler_normalize.fit_transform(Q_obs.reshape(-1, 1)).flatten()

# # Function to run Bayesian model with reduced parameters
# def build_and_run_model(Q_obs_transformed):
#     with pm.Model() as model:
#         # Define priors for radius and viscosity
#         r = pm.HalfNormal("r", sigma=0.0035)
#         mu = pm.Normal("mu", mu=0.0035, sigma=0.01)
        
#         # Flowrate model
#         Q_est = (np.pi * r**4 * delta_P) / (8 * mu * L)

#         # Likelihood with noisy observations
#         pm.Normal("Q_obs", mu=Q_est, sigma=Q_true * 0.2, observed=Q_obs_transformed)

#         # Sampling using MCMC (lower samples for faster run)
#         trace = pm.sample(draws=200, tune=50, chains=2, cores=2, target_accept=0.95, return_inferencedata=True)
#     return trace

# # Main function that will be called when the script runs
# def main():
#     # Run the model for original, standardized, and normalized observations
#     trace_original = build_and_run_model(Q_obs)
#     trace_standardized = build_and_run_model(Q_obs_standardized)
#     trace_normalized = build_and_run_model(Q_obs_normalized)

#     # Extract posterior samples for radius and viscosity
#     posterior_r_original = trace_original.posterior["r"].stack(samples=("chain", "draw")).values
#     posterior_mu_original = trace_original.posterior["mu"].stack(samples=("chain", "draw")).values

#     posterior_r_standardized = trace_standardized.posterior["r"].stack(samples=("chain", "draw")).values
#     posterior_mu_standardized = trace_standardized.posterior["mu"].stack(samples=("chain", "draw")).values

#     posterior_r_normalized = trace_normalized.posterior["r"].stack(samples=("chain", "draw")).values
#     posterior_mu_normalized = trace_normalized.posterior["mu"].stack(samples=("chain", "draw")).values

#     # Set folder structure for saving results and plots
#     folder_path = "scaling_comparison_effects"
#     os.makedirs(folder_path, exist_ok=True)
#     results_folder = os.path.join(folder_path, "results")
#     os.makedirs(results_folder, exist_ok=True)
#     plots_folder = os.path.join(folder_path, "plots")
#     os.makedirs(plots_folder, exist_ok=True)

#     # Save results to CSV
#     results_df = pd.DataFrame({
#         "Scaling Method": ["Original", "Standardized", "Normalized"],
#         "Mean Radius": [
#             np.mean(posterior_r_original),
#             np.mean(posterior_r_standardized),
#             np.mean(posterior_r_normalized)
#         ],
#         "Mean Viscosity": [
#             np.mean(posterior_mu_original),
#             np.mean(posterior_mu_standardized),
#             np.mean(posterior_mu_normalized)
#         ]
#     })

#     # Save the results CSV file
#     csv_filename = os.path.join(results_folder, "scaling_comparison_results.csv")
#     results_df.to_csv(csv_filename, index=False)

#     # Plotting Radius and Viscosity for Original, Standardized, and Normalized
#     fig, axs = plt.subplots(2, 1, figsize=(10, 10))

#     # Radius Plot
#     axs[0].hist(posterior_r_original, bins=20, alpha=0.5, label="Original", color='blue')
#     axs[0].hist(posterior_r_standardized, bins=20, alpha=0.5, label="Standardized", color='green')
#     axs[0].hist(posterior_r_normalized, bins=20, alpha=0.5, label="Normalized", color='red')
#     axs[0].set_title("Posterior Distribution of Radius")
#     axs[0].legend()

#     # Viscosity Plot
#     axs[1].hist(posterior_mu_original, bins=20, alpha=0.5, label="Original", color='blue')
#     axs[1].hist(posterior_mu_standardized, bins=20, alpha=0.5, label="Standardized", color='green')
#     axs[1].hist(posterior_mu_normalized, bins=20, alpha=0.5, label="Normalized", color='red')
#     axs[1].set_title("Posterior Distribution of Viscosity")
#     axs[1].legend()

#     # Save the plot
#     plot_filename = os.path.join(plots_folder, "comparison_all_runs.png")
#     plt.tight_layout()
#     plt.savefig(plot_filename)
#     plt.close()

#     print("Results saved to CSV and plots saved successfully.")

# # Call the main function only when the script is run directly
# if __name__ == "__main__":
#     main()



# #standardization_normalization_effects




# import pymc as pm
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import pandas as pd
# from sklearn.preprocessing import StandardScaler, MinMaxScaler

# # Simulate basic parameters
# np.random.seed(42)

# # # True parameters
# # delta_P = np.random.normal(loc=300, scale=300)  # Pressure drop (Pa)
# # L = np.random.normal(loc=0.02, scale=0.03)      # Length of artery (m)
# # true_r = np.random.normal(loc=0.0002, scale=0.0003)  # Radius (m)
# # true_mu = np.random.normal(loc=0.0035, scale=0.005)  # Viscosity (Pa·s)



# # Adjusted parameters for faster execution (reduce standard deviation)
# delta_P = np.random.normal(loc=300, scale=100)  # Reduce variability for pressure drop
# L = np.random.normal(loc=0.02, scale=0.005)      # Reduce variability for length of artery
# true_r = np.random.normal(loc=0.0002, scale=0.0001)  # Less variability for radius
# true_mu = np.random.normal(loc=0.0035, scale=0.001)  # Less variability for viscosity






# # Poiseuille's equation for flowrate
# def poiseuille_flow(r, mu):
#     return (np.pi * r**4 * delta_P) / (8 * mu * L)

# # Simulate noisy observations
# def simulate_noisy_observations(Q_true, num_samples=10, noise_percent=0.2):
#     return np.random.normal(loc=Q_true, scale=Q_true * noise_percent, size=num_samples)

# # Standardization (Z-score normalization)
# scaler_standard = StandardScaler()
# scaler_normalize = MinMaxScaler()

# # Generate true flowrate and noisy observations
# Q_true = poiseuille_flow(true_r, true_mu)
# Q_obs = simulate_noisy_observations(Q_true)

# # Apply Standardization and Normalization
# Q_obs_standardized = scaler_standard.fit_transform(Q_obs.reshape(-1, 1)).flatten()
# Q_obs_normalized = scaler_normalize.fit_transform(Q_obs.reshape(-1, 1)).flatten()

# # Function to run Bayesian model with reduced parameters
# def build_and_run_model(Q_obs_transformed):
#     with pm.Model() as model:
#         # Define priors for radius and viscosity
#         r = pm.HalfNormal("r", sigma=0.0035)
#         mu = pm.Normal("mu", mu=0.0035, sigma=0.01)
        
#         # Flowrate model
#         Q_est = (np.pi * r**4 * delta_P) / (8 * mu * L)

#         # Likelihood with noisy observations
#         pm.Normal("Q_obs", mu=Q_est, sigma=Q_true * 0.2, observed=Q_obs_transformed)

#         # Sampling using MCMC 
#         trace = pm.sample(draws=1000, tune=500, chains=4, cores=4, target_accept=0.95, return_inferencedata=True)
#     return trace

# # Main function that will be called when the script runs
# def main():
#     # Run the model for original, standardized, and normalized observations
#     trace_original = build_and_run_model(Q_obs)
#     trace_standardized = build_and_run_model(Q_obs_standardized)
#     trace_normalized = build_and_run_model(Q_obs_normalized)

#     # Extract posterior samples for radius, viscosity, and flow rate
#     posterior_r_original = trace_original.posterior["r"].stack(samples=("chain", "draw")).values
#     posterior_mu_original = trace_original.posterior["mu"].stack(samples=("chain", "draw")).values
#     posterior_Q_original = (np.pi * posterior_r_original**4 * delta_P) / (8 * posterior_mu_original * L)

#     posterior_r_standardized = trace_standardized.posterior["r"].stack(samples=("chain", "draw")).values
#     posterior_mu_standardized = trace_standardized.posterior["mu"].stack(samples=("chain", "draw")).values
#     posterior_Q_standardized = (np.pi * posterior_r_standardized**4 * delta_P) / (8 * posterior_mu_standardized * L)

#     posterior_r_normalized = trace_normalized.posterior["r"].stack(samples=("chain", "draw")).values
#     posterior_mu_normalized = trace_normalized.posterior["mu"].stack(samples=("chain", "draw")).values
#     posterior_Q_normalized = (np.pi * posterior_r_normalized**4 * delta_P) / (8 * posterior_mu_normalized * L)

#     # Set folder structure for saving results and plots
#     folder_path = "standardization_normalization_effects"
#     os.makedirs(folder_path, exist_ok=True)
#     results_folder = os.path.join(folder_path, "results")
#     os.makedirs(results_folder, exist_ok=True)
#     plots_folder = os.path.join(folder_path, "plots")
#     os.makedirs(plots_folder, exist_ok=True)

#     # Save results to CSV (Including flow rates)
#     results_df = pd.DataFrame({
#         "Scaling Method": ["Original", "Standardized", "Normalized"],
#         "Mean Radius": [
#             np.mean(posterior_r_original),
#             np.mean(posterior_r_standardized),
#             np.mean(posterior_r_normalized)
#         ],
#         "Mean Viscosity": [
#             np.mean(posterior_mu_original),
#             np.mean(posterior_mu_standardized),
#             np.mean(posterior_mu_normalized)
#         ],
#         "Mean Flow Rate": [
#             np.mean(posterior_Q_original),
#             np.mean(posterior_Q_standardized),
#             np.mean(posterior_Q_normalized)
#         ]
#     })

#     # Save the results CSV file
#     csv_filename = os.path.join(results_folder, "standardization_normalization_effects.csv")
#     results_df.to_csv(csv_filename, index=False)

#     # Plotting Radius, Viscosity, and Flow Rate for Original, Standardized, and Normalized
#     fig, axs = plt.subplots(3, 1, figsize=(10, 15))

#     # Radius Plot
#     axs[0].hist(posterior_r_original, bins=20, alpha=0.5, label="Original", color='blue')
#     axs[0].hist(posterior_r_standardized, bins=20, alpha=0.5, label="Standardized", color='green')
#     axs[0].hist(posterior_r_normalized, bins=20, alpha=0.5, label="Normalized", color='red')
#     axs[0].set_title("Posterior Distribution of Radius")
#     axs[0].legend()

#     # Viscosity Plot
#     axs[1].hist(posterior_mu_original, bins=20, alpha=0.5, label="Original", color='blue')
#     axs[1].hist(posterior_mu_standardized, bins=20, alpha=0.5, label="Standardized", color='green')
#     axs[1].hist(posterior_mu_normalized, bins=20, alpha=0.5, label="Normalized", color='red')
#     axs[1].set_title("Posterior Distribution of Viscosity")
#     axs[1].legend()

#     # Flow Rate Plot
#     axs[2].hist(posterior_Q_original, bins=20, alpha=0.5, label="Original", color='blue')
#     axs[2].hist(posterior_Q_standardized, bins=20, alpha=0.5, label="Standardized", color='green')
#     axs[2].hist(posterior_Q_normalized, bins=20, alpha=0.5, label="Normalized", color='red')
#     axs[2].set_title("Posterior Distribution of Flow Rate")
#     axs[2].legend()

#     # Save the plot
#     plot_filename = os.path.join(plots_folder, "standardization_normalization_effects.png")
#     plt.tight_layout()
#     plt.savefig(plot_filename)
#     plt.close()

#     print("Results saved to CSV and plots saved successfully.")

# # Call the main function only when the script is run directly
# if __name__ == "__main__":
#     main()





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
