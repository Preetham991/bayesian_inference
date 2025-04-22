# import pymc as pm
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import gaussian_kde
# from scipy.integrate import quad
# import os
# from utils import save_results_to_csv_and_pickle, visualize_results

# # -----------------------
# # Sampled Parameters with Aggressive Dispersion
# # -----------------------
# np.random.seed(42)

# # Increase variability for pressure, length, radius, and viscosity
# delta_P = np.random.normal(loc=300, scale=300)         # Pressure drop (Pa)
# L = np.random.normal(loc=0.02, scale=0.03)             # Length of artery (m)
# true_r = np.random.normal(loc=0.0002, scale=0.0003)    # Radius (m) - small radius possible
# true_mu = np.random.normal(loc=0.0035, scale=0.005)    # Viscosity (Pa·s) - high resistance possible

# # -----------------------
# # Core Functions
# # -----------------------

# def poiseuille_flow_vec(r, mu):
#     """Vectorized Poiseuille's equation"""
#     return (np.pi * np.power(r, 4) * delta_P) / (8 * mu * L)

# def simulate_noisy_observations(Q_true, num_samples=30, noise_percent=1.0):
#     """Generate noisy observations"""
#     return np.random.normal(loc=Q_true, scale=Q_true * noise_percent, size=num_samples)

# def build_and_run_model(Q_obs, Q_true):
#     """Bayesian model using PyMC with high dispersion priors"""
#     with pm.Model() as model:
#         r = pm.HalfNormal("r", sigma=0.0035)            # Wide HalfNormal → allows near-zero
#         mu = pm.Normal("mu", mu=0.0035, sigma=0.01)     # Wide Normal → high resistance

#         Q_est = (np.pi * r**4 * delta_P) / (8 * mu * L)

#         pm.Normal("Q_obs", mu=Q_est, sigma=Q_true * 1.0, observed=Q_obs)

#         trace = pm.sample(
#             draws=1000,
#             tune=500,
#             chains=4,
#             cores=4,
#             init="adapt_diag",
#             target_accept=0.95,
#             return_inferencedata=True,
#             idata_kwargs={"log_likelihood": False},
#             random_seed=42,
#             sampler_kwargs={"nuts_sampler": "numpyro"}  # Faster backend
#         )
#     return trace

# def analyze_posteriors(trace, Q_true):
#     """Analyze posterior samples and detect abnormalities"""
#     posterior_r = trace.posterior["r"].stack(samples=("chain", "draw")).values
#     posterior_mu = trace.posterior["mu"].stack(samples=("chain", "draw")).values
#     posterior_Q = poiseuille_flow_vec(posterior_r, posterior_mu)

#     normal_lower = Q_true * 0.9
#     normal_upper = Q_true * 1.1

#     # New thresholds for "Blockage" and "Extreme Blockage"
#     extreme_blockage_threshold = 0.3 * normal_lower  # Extreme Blockage below 30% of the normal lower bound
#     blockage_threshold = 0.7 * normal_lower  # Blockage between 30% and 70% of the normal lower bound

#     alerts = np.where(
#         posterior_Q < extreme_blockage_threshold,
#         "Extreme Blockage",  # New extreme blockage regime
#         np.where(
#             posterior_Q < blockage_threshold,
#             "Blockage",  # New blockage regime
#             np.where(
#                 posterior_Q < normal_lower * 0.3,
#                 "Vasoconstriction",
#                 np.where(
#                     posterior_Q > normal_upper,
#                     "Vasodilation",
#                     "Normal"
#                 )
#             )
#         )
#     )
#     return posterior_r, posterior_mu, posterior_Q, alerts

# # -----------------------
# # Robust Edge Classification with KDE
# # -----------------------

# def classify_edge_point_robust(Q_edge, Q_post, Q_true, epsilon=0.01, verbose=True):
#     """
#     Robustly estimates the probability that a point near regime boundary belongs
#     to one of the neighboring regimes using KDE-based integration.
#     """
#     # Define regime boundaries
#     normal_lower = 0.9 * Q_true
#     normal_upper = 1.1 * Q_true
#     extreme_blockage_threshold = 0.3 * normal_lower  # Extreme Blockage below 30%
#     blockage_threshold = 0.7 * normal_lower  # Blockage between 30% and 70%

#     # Determine neighboring regimes
#     if Q_edge < extreme_blockage_threshold:
#         regime_low, regime_high = "Extreme Blockage", "Blockage"
#     elif extreme_blockage_threshold <= Q_edge < blockage_threshold:
#         regime_low, regime_high = "Blockage", "Vasoconstriction"
#     elif blockage_threshold <= Q_edge < normal_lower:
#         regime_low, regime_high = "Vasoconstriction", "Normal"
#     elif normal_lower <= Q_edge < normal_upper:
#         regime_low, regime_high = "Normal", "Vasodilation"
#     else:
#         regime_low, regime_high = "Vasodilation", "Extreme Vasodilation"

#     # Fit KDE to posterior samples
#     kde = gaussian_kde(Q_post)

#     # Integration range
#     lower_range = (Q_edge - epsilon, Q_edge)
#     upper_range = (Q_edge, Q_edge + epsilon)

#     # Compute probabilities in the epsilon neighborhoods
#     prob_low, _ = quad(kde, lower_range[0], lower_range[1])
#     prob_high, _ = quad(kde, upper_range[0], upper_range[1])

#     # Normalize
#     total_prob = prob_low + prob_high
#     prob_low /= total_prob
#     prob_high /= total_prob

#     if verbose:
#         print(f"\n Robust Edge Classification at Q = {Q_edge:.12f}")
#         print(f"Between: [{regime_low}] ←→ [{regime_high}]")
#         print(f"P({regime_low}) = {prob_low:.3f}")
#         print(f"P({regime_high}) = {prob_high:.3f}")

#     return {regime_low: prob_low, regime_high: prob_high}

# # -----------------------
# # Calculate the number of flow points in different regimes
# # -----------------------

# def count_regimes(posterior_Q, Q_true):
#     normal_lower = 0.9 * Q_true
#     normal_upper = 1.1 * Q_true
#     extreme_blockage_threshold = 0.3 * normal_lower  # Extreme Blockage below 30%
#     blockage_threshold = 0.7 * normal_lower  # Blockage between 30% and 70%

#     regimes = {
#         "Extreme Blockage": 0,
#         "Blockage": 0,
#         "Vasoconstriction": 0,
#         "Normal": 0,
#         "Vasodilation": 0,
#         "Extreme Vasodilation": 0
#     }

#     for Q in posterior_Q:
#         if Q < extreme_blockage_threshold:
#             regimes["Extreme Blockage"] += 1
#         elif extreme_blockage_threshold <= Q < blockage_threshold:
#             regimes["Blockage"] += 1
#         elif blockage_threshold <= Q < normal_lower:
#             regimes["Vasoconstriction"] += 1
#         elif normal_lower <= Q < normal_upper:
#             regimes["Normal"] += 1
#         elif Q >= normal_upper:
#             regimes["Vasodilation"] += 1
#         else:
#             regimes["Extreme Vasodilation"] += 1

#     return regimes

# # -----------------------
# # Main Pipeline
# # -----------------------
# if __name__ == "__main__":
#     folder_path = "blood_flow_analysis_results"

#     Q_true = poiseuille_flow_vec(true_r, true_mu)
#     Q_obs = simulate_noisy_observations(Q_true)

#     trace = build_and_run_model(Q_obs, Q_true)
#     posterior_r, posterior_mu, posterior_Q, alerts = analyze_posteriors(trace, Q_true)

#     save_results_to_csv_and_pickle(posterior_r, posterior_mu, posterior_Q, alerts, folder_path)
#     visualize_results(posterior_r, posterior_mu, posterior_Q, Q_true, true_r, true_mu, alerts, folder_path)

#     # Calculate the number of flow points in different regimes
#     regime_counts = count_regimes(posterior_Q, Q_true)

#     # Print the number of flow points in each regime
#     print("\n Flow Regime Counts:")
#     for regime, count in regime_counts.items():
#         print(f"{regime}: {count} flow points")







import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.integrate import quad
import os
from utils import save_results_to_csv_and_pickle, visualize_results

# -----------------------
# Sampled Parameters with Aggressive Dispersion
# -----------------------
np.random.seed(42)

# Increase variability for pressure, length, radius, and viscosity
delta_P = np.random.normal(loc=300, scale=300)         # Pressure drop (Pa)
L = np.random.normal(loc=0.02, scale=0.03)             # Length of artery (m)
true_r = np.random.normal(loc=0.0002, scale=0.0003)    # Radius (m) - small radius possible
true_mu = np.random.normal(loc=0.0035, scale=0.005)    # Viscosity (Pa·s) - high resistance possible

# -----------------------
# Core Functions
# -----------------------

def poiseuille_flow_vec(r, mu):
    """Vectorized Poiseuille's equation"""
    return (np.pi * np.power(r, 4) * delta_P) / (8 * mu * L)

def simulate_noisy_observations(Q_true, num_samples=30, noise_percent=1.0):
    """Generate noisy observations"""
    return np.random.normal(loc=Q_true, scale=Q_true * noise_percent, size=num_samples)

def build_and_run_model(Q_obs, Q_true):
    """Bayesian model using PyMC with high dispersion priors"""
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

# -----------------------
# Robust Edge Classification with KDE
# -----------------------

def classify_edge_point_robust(Q_edge, Q_post, Q_true, epsilon=0.01, verbose=True):
    """
    Robustly estimates the probability that a point near regime boundary belongs
    to one of the neighboring regimes using KDE-based integration.
    """
    # Define regime boundaries
    normal_lower = 0.9 * Q_true
    normal_upper = 1.1 * Q_true
    blockage_threshold = 0.3 * normal_lower

    # Determine neighboring regimes
    if Q_edge < blockage_threshold:
        regime_low, regime_high = "Below Blockage", "Blockage"
    elif blockage_threshold <= Q_edge < normal_lower:
        regime_low, regime_high = "Blockage", "Vasoconstriction"
    elif normal_lower <= Q_edge < normal_upper:
        regime_low, regime_high = "Normal", "Vasodilation"
    else:
        regime_low, regime_high = "Vasodilation", "Extreme Vasodilation"

    # Fit KDE to posterior samples
    kde = gaussian_kde(Q_post)

    # Integration range
    lower_range = (Q_edge - epsilon, Q_edge)
    upper_range = (Q_edge, Q_edge + epsilon)

    # Compute probabilities in the epsilon neighborhoods
    prob_low, _ = quad(kde, lower_range[0], lower_range[1])
    prob_high, _ = quad(kde, upper_range[0], upper_range[1])

    # Normalize
    total_prob = prob_low + prob_high
    prob_low /= total_prob
    prob_high /= total_prob

    if verbose:
        print(f"Robust Edge Classification at Q = {Q_edge:.12f}")
        print(f"Between: [{regime_low}] ←→ [{regime_high}]")
        print(f"P({regime_low}) = {prob_low:.3f}")
        print(f"P({regime_high}) = {prob_high:.3f}")

    return {regime_low: prob_low, regime_high: prob_high}

def classify_transition_probability(Q_edge, Q_post, Q_true, epsilon=0.01, verbose=False):
    """
    Calculate transition probability between regimes using KDE-based integration.
    """
    # Define regime boundaries
    normal_lower = 0.9 * Q_true
    normal_upper = 1.1 * Q_true
    blockage_threshold = 0.3 * normal_lower

    # Identify the starting regime based on Q_edge
    if Q_edge < blockage_threshold:
        start_regime = "Blockage"
        end_regime = "Vasoconstriction"
    elif blockage_threshold <= Q_edge < normal_lower:
        start_regime = "Vasoconstriction"
        end_regime = "Normal"
    elif normal_lower <= Q_edge < normal_upper:
        start_regime = "Normal"
        end_regime = "Vasodilation"
    else:
        start_regime = "Vasodilation"
        end_regime = "Extreme Vasodilation"

    # Fit KDE to posterior samples
    kde = gaussian_kde(Q_post)

    # Compute probabilities for transition from start to end regime
    lower_range = (Q_edge - epsilon, Q_edge)
    upper_range = (Q_edge, Q_edge + epsilon)
    prob_start, _ = quad(kde, lower_range[0], lower_range[1])
    prob_end, _ = quad(kde, upper_range[0], upper_range[1])

    total_prob = prob_start + prob_end
    prob_start /= total_prob
    prob_end /= total_prob

    # if verbose:
    #     print(f"\nTransition Probability from {start_regime} to {end_regime}:")
    #     print(f"P({start_regime}) = {prob_start:.3f}")
    #     print(f"P({end_regime}) = {prob_end:.3f}")

    return {start_regime: prob_start, end_regime: prob_end}

def find_transition_points(posterior_Q, Q_true, threshold=0.1):
    """
    Find edge points where there are significant changes in the flow regimes
    based on posterior flow rates.
    """
    transition_points = []
    normal_lower = 0.9 * Q_true
    normal_upper = 1.1 * Q_true

    # Identify points where regime transition occurs
    for i in range(1, len(posterior_Q)):
        prev_Q = posterior_Q[i-1]
        curr_Q = posterior_Q[i]
        if (prev_Q < normal_lower and curr_Q >= normal_lower) or (prev_Q > normal_upper and curr_Q <= normal_upper):
            # Transition detected: add edge points
            transition_points.append(curr_Q)

    return transition_points

# -----------------------
# Main Pipeline
# -----------------------
if __name__ == "__main__":
    folder_path = "blood_flow_analysis_results"

    Q_true = poiseuille_flow_vec(true_r, true_mu)
    Q_obs = simulate_noisy_observations(Q_true)

    trace = build_and_run_model(Q_obs, Q_true)
    posterior_r, posterior_mu, posterior_Q, alerts = analyze_posteriors(trace, Q_true)

    save_results_to_csv_and_pickle(posterior_r, posterior_mu, posterior_Q, alerts, folder_path)
    visualize_results(posterior_r, posterior_mu, posterior_Q, Q_true, true_r, true_mu, alerts, folder_path)

    # Find transition points based on posterior samples
    transition_points = find_transition_points(posterior_Q, Q_true)
    print(f"\nDetected Transition Points: {transition_points}")

    seen_transitions = set()

    # Classify transition probabilities for each unique transition point
    for edge_point in transition_points:
        result = classify_transition_probability(edge_point, posterior_Q, Q_true, epsilon=0.01 * Q_true, verbose=True)

        # Extract regimes from result dictionary
        regimes = tuple(sorted(result.keys()))  # Ensure consistent ordering

        if regimes not in seen_transitions:
            seen_transitions.add(regimes)
            print(f"\n Transition at Q={edge_point:.12f}:")
            for regime, prob in result.items():
                print(f"  P({regime}) = {prob:.3f}")
