# utils.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

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

def visualize_results(posterior_r, posterior_mu, posterior_Q, Q_true, true_r, true_mu, alerts, folder_path):
    """Plot posterior histograms and flow states"""
    alert_counts = Counter(alerts)
    normal_lower = Q_true * 0.9
    normal_upper = Q_true * 1.1

    # ------------------ Stats for plotting ------------------ #
    def stats(x): return np.mean(x), np.std(x)
    r_mean, r_std = stats(posterior_r)
    mu_mean, mu_std = stats(posterior_mu)
    Q_mean, Q_std = stats(posterior_Q)

    def sigma_bounds(mean, std): return [(mean - i*std, mean + i*std) for i in [1, 2, 3]]
    r_bounds = sigma_bounds(r_mean, r_std)
    mu_bounds = sigma_bounds(mu_mean, mu_std)
    Q_bounds = sigma_bounds(Q_mean, Q_std)

    plt.figure(figsize=(18, 5))

    # ------------------ Radius ------------------ #
    plt.subplot(1, 3, 1)
    plt.hist(posterior_r, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(true_r, color='red', linestyle='--', label='True r')
    for i, (l, h) in enumerate(r_bounds[::-1], start=1):
        plt.axvspan(l, h, alpha=0.3, label=f'r ± {4-i}σ', color=['yellow', 'orange', 'green'][3-i])
    plt.title("Posterior of Radius")
    plt.xlabel("Radius (m)")
    plt.ylabel("Count")
    plt.legend()

    # ------------------ Viscosity ------------------ #
    plt.subplot(1, 3, 2)
    plt.hist(posterior_mu, bins=30, color='lightgreen', edgecolor='black')
    plt.axvline(true_mu, color='red', linestyle='--', label='True μ')
    for i, (l, h) in enumerate(mu_bounds[::-1], start=1):
        plt.axvspan(l, h, alpha=0.3, label=f'μ ± {4-i}σ', color=['yellow', 'orange', 'green'][3-i])
    plt.title("Posterior of Viscosity")
    plt.xlabel("Viscosity (Pa·s)")
    plt.ylabel("Count")
    plt.legend()

    # ------------------ Flow ------------------ #
    plt.subplot(1, 3, 3)
    plt.hist(posterior_Q, bins=30, color='orange', edgecolor='black')
    plt.axvline(Q_true, color='black', linestyle='--', label='True Q')
    for i, (l, h) in enumerate(Q_bounds[::-1], start=1):
        plt.axvspan(l, h, alpha=0.3, label=f'Q ± {4-i}σ', color=['yellow', 'orange', 'green'][3-i])
    plt.axvspan(0, normal_lower * 0.3, color='black', alpha=0.2, label='Blockage')
    plt.axvspan(normal_lower * 0.3, normal_lower, color='red', alpha=0.2, label='Vasoconstriction')
    plt.axvspan(normal_upper, np.max(posterior_Q), color='blue', alpha=0.2, label='Vasodilation')
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
