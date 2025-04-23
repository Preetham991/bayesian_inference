# utils.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np



def save_results_to_csv_and_pickle(posterior_r, posterior_mu, posterior_Q, alerts, folder_path):
    """Save posterior results to CSV and Pickle file, saving both in the same location"""
    # Create a DataFrame with the posterior data
    df = pd.DataFrame({
        "posterior_r": posterior_r,
        "posterior_mu": posterior_mu,
        "posterior_Q": posterior_Q,
        "alert": alerts
    })
    
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Define the CSV file path
    csv_file_path = os.path.join(folder_path, "posterior_results.csv")
    
    # Save the DataFrame as a CSV file in the specified folder
    df.to_csv(csv_file_path, index=False)
    print(f"Results saved to {csv_file_path}")
    
    # Define the Pickle file path (same location as the CSV file)
    pickle_file_path = os.path.splitext(csv_file_path)[0] + ".pkl"
    
    # Save the DataFrame as a Pickle file in the same location as the CSV
    df.to_pickle(pickle_file_path)
    print(f"Pickle file saved to {pickle_file_path}")



def visualize_results(posterior_r, posterior_mu, posterior_Q, Q_true, true_r, true_mu, alerts, folder_path):
    """Plot posterior histograms using quantile-based probabilistic binning for radius, viscosity, and flow."""
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from collections import Counter
    from matplotlib.patches import Patch

    alert_counts = Counter(alerts)

    def quantile_bins(data, edges):
        return [np.percentile(data, q) for q in edges]

    # Quantile bin thresholds (existing binning logic)
    r_bins = quantile_bins(posterior_r, [5, 10, 90, 95])
    mu_bins = quantile_bins(posterior_mu, [5, 10, 90, 95])
    Q_bins = quantile_bins(posterior_Q, [5, 10, 90, 95])

    r_states = {
        'Too Narrow': (0, r_bins[0]),
        'Narrow': (r_bins[0], r_bins[1]),
        'Normal': (r_bins[1], r_bins[2]),
        'Wide': (r_bins[2], r_bins[3]),
        'Too Wide': (r_bins[3], np.max(posterior_r))
    }

    mu_states = {
        'Very Thin': (0, mu_bins[0]),
        'Thin': (mu_bins[0], mu_bins[1]),
        'Normal': (mu_bins[1], mu_bins[2]),
        'Thick': (mu_bins[2], mu_bins[3]),
        'Very Thick': (mu_bins[3], np.max(posterior_mu))
    }

    Q_states = {
        'Blockage': (0, Q_bins[0]),
        'Vasoconstriction': (Q_bins[0], Q_bins[1]),
        'Normal': (Q_bins[1], Q_bins[2]),
        'Vasodilation': (Q_bins[2], Q_bins[3]),
        'Severe Vasodilation': (Q_bins[3], np.max(posterior_Q))
    }

    def color_bin(patch, mid, bins):
        for label, (low, high) in bins.items():
            if low <= mid < high:
                color_map = {
                    'Too Narrow': 'black', 'Narrow': 'red', 'Normal': 'green', 'Wide': 'blue', 'Too Wide': 'purple',
                    'Very Thin': 'black', 'Thin': 'orange', 'Thick': 'blue', 'Very Thick': 'purple',
                    'Blockage': 'black', 'Vasoconstriction': 'red', 'Vasodilation': 'blue', 'Severe Vasodilation': 'purple'
                }
                patch.set_facecolor(color_map.get(label, 'gray'))
                patch.set_alpha(0.4)
                break

    # Calculate μ and σ for all parameters (to be used for μ ± 1σ, 2σ, 3σ)
    r_mu, r_sigma = np.mean(posterior_r), np.std(posterior_r)
    mu_mu, mu_sigma = np.mean(posterior_mu), np.std(posterior_mu)
    Q_mu, Q_sigma = np.mean(posterior_Q), np.std(posterior_Q)

    # ------------------- Plotting ------------------- #
    plt.figure(figsize=(18, 5))

    # -------- Posterior of Radius -------- #
    plt.subplot(1, 3, 1)
    # Plot μ ± 1σ, μ ± 2σ, μ ± 3σ lines in the background first
    plt.axvline(r_mu + r_sigma, linestyle=':', color='lightgreen', label='μ + 1σ')
    plt.axvline(r_mu - r_sigma, linestyle=':', color='lightgreen', label='μ - 1σ')
    plt.axvline(r_mu + 2 * r_sigma, linestyle=':', color='lightblue', label='μ + 2σ')
    plt.axvline(r_mu - 2 * r_sigma, linestyle=':', color='lightblue', label='μ - 2σ')
    plt.axvline(r_mu + 3 * r_sigma, linestyle=':', color='lightcoral', label='μ + 3σ')
    plt.axvline(r_mu - 3 * r_sigma, linestyle=':', color='lightcoral', label='μ - 3σ')

    n_r, bins_r, patches_r = plt.hist(posterior_r, bins=30, color='lightgray', edgecolor='black')
    for patch, left in zip(patches_r, bins_r[:-1]):
        mid = patch.get_x() + patch.get_width() / 2
        color_bin(patch, mid, r_states)
    for label, (pos, _) in r_states.items():
        plt.axvline(pos, linestyle='--', alpha=0.3, color='gray')
    plt.axvline(true_r, color='red', linestyle='-', label='True r')
    plt.title("Posterior of Radius")
    plt.xlabel("Radius (m)")
    plt.ylabel("Count")
    legend_elements_r = [
        Patch(facecolor='black', label='Too Narrow', alpha=0.4),
        Patch(facecolor='red', label='Narrow', alpha=0.4),
        Patch(facecolor='green', label='Normal', alpha=0.4),
        Patch(facecolor='blue', label='Wide', alpha=0.4),
        Patch(facecolor='purple', label='Too Wide', alpha=0.4),
        Patch(color='lightgreen', label='μ ± 1σ'),
        Patch(color='lightblue', label='μ ± 2σ'),
        Patch(color='lightcoral', label='μ ± 3σ'),
        Patch(color='red', label='True r')
    ]
    plt.legend(handles=legend_elements_r)

    # -------- Posterior of Viscosity -------- #
    plt.subplot(1, 3, 2)
    # Plot μ ± 1σ, μ ± 2σ, μ ± 3σ lines in the background first
    plt.axvline(mu_mu + mu_sigma, linestyle=':', color='lightgreen', label='μ + 1σ')
    plt.axvline(mu_mu - mu_sigma, linestyle=':', color='lightgreen', label='μ - 1σ')
    plt.axvline(mu_mu + 2 * mu_sigma, linestyle=':', color='lightblue', label='μ + 2σ')
    plt.axvline(mu_mu - 2 * mu_sigma, linestyle=':', color='lightblue', label='μ - 2σ')
    plt.axvline(mu_mu + 3 * mu_sigma, linestyle=':', color='lightcoral', label='μ + 3σ')
    plt.axvline(mu_mu - 3 * mu_sigma, linestyle=':', color='lightcoral', label='μ - 3σ')

    n_mu, bins_mu, patches_mu = plt.hist(posterior_mu, bins=30, color='lightgray', edgecolor='black')
    for patch, left in zip(patches_mu, bins_mu[:-1]):
        mid = patch.get_x() + patch.get_width() / 2
        color_bin(patch, mid, mu_states)
    for label, (pos, _) in mu_states.items():
        plt.axvline(pos, linestyle='--', alpha=0.3, color='gray')
    plt.axvline(true_mu, color='red', linestyle='-', label='True μ')
    plt.title("Posterior of Viscosity")
    plt.xlabel("Viscosity (Pa·s)")
    plt.ylabel("Count")
    legend_elements_mu = [
        Patch(facecolor='black', label='Very Thin', alpha=0.4),
        Patch(facecolor='orange', label='Thin', alpha=0.4),
        Patch(facecolor='green', label='Normal', alpha=0.4),
        Patch(facecolor='blue', label='Thick', alpha=0.4),
        Patch(facecolor='purple', label='Very Thick', alpha=0.4),
        Patch(color='lightgreen', label='μ ± 1σ'),
        Patch(color='lightblue', label='μ ± 2σ'),
        Patch(color='lightcoral', label='μ ± 3σ'),
        Patch(color='red', label='True μ')
    ]
    plt.legend(handles=legend_elements_mu)

    # -------- Posterior of Flow -------- #
    plt.subplot(1, 3, 3)
    # Plot μ ± 1σ, μ ± 2σ, μ ± 3σ lines in the background first
    plt.axvline(Q_mu + Q_sigma, linestyle=':', color='lightgreen', label='μ + 1σ')
    plt.axvline(Q_mu - Q_sigma, linestyle=':', color='lightgreen', label='μ - 1σ')
    plt.axvline(Q_mu + 2 * Q_sigma, linestyle=':', color='lightblue', label='μ + 2σ')
    plt.axvline(Q_mu - 2 * Q_sigma, linestyle=':', color='lightblue', label='μ - 2σ')
    plt.axvline(Q_mu + 3 * Q_sigma, linestyle=':', color='lightcoral', label='μ + 3σ')
    plt.axvline(Q_mu - 3 * Q_sigma, linestyle=':', color='lightcoral', label='μ - 3σ')

    n_q, bins_q, patches_q = plt.hist(posterior_Q, bins=30, color='lightgray', edgecolor='black')
    for patch, left in zip(patches_q, bins_q[:-1]):
        mid = patch.get_x() + patch.get_width() / 2
        color_bin(patch, mid, Q_states)
    for label, (pos, _) in Q_states.items():
        plt.axvline(pos, linestyle='--', alpha=0.3, color='gray')
    plt.axvline(Q_true, color='black', linestyle='-', label='True Q')
    plt.title("Posterior of Flow")
    plt.xlabel("Flow Rate (m³/s)")
    plt.ylabel("Count")
    legend_elements_q = [
        Patch(facecolor='black', label='Blockage', alpha=0.4),
        Patch(facecolor='red', label='Vasoconstriction', alpha=0.4),
        Patch(facecolor='green', label='Normal', alpha=0.4),
        Patch(facecolor='blue', label='Vasodilation', alpha=0.4),
        Patch(facecolor='purple', label='Severe Vasodilation', alpha=0.4),
        Patch(color='lightgreen', label='μ ± 1σ', alpha=0.6),
        Patch(color='lightblue', label='μ ± 2σ', alpha=0.6),
        Patch(color='lightcoral', label='μ ± 3σ', alpha=0.6),
        Patch(color='black', label='True Q', alpha=1.0)
    ]
    plt.legend(handles=legend_elements_q)

    plt.tight_layout()

    if folder_path:
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(os.path.join(folder_path, 'posterior_histograms.png'))

    plt.show()
    
    
    
    
    
# utils.py
import pandas as pd
import os

def excel_to_pickle(excel_file_path):
    """
    Converts an Excel file to a pickle file and saves it in the same directory.

    Parameters:
    - excel_file_path (str): The path to the input Excel file.
    """
    # Extract the directory of the Excel file
    dir_path = os.path.dirname(excel_file_path)
    
    # Define the pickle file path (same as Excel file, but with .pkl extension)
    pickle_file_path = os.path.join(dir_path, os.path.splitext(os.path.basename(excel_file_path))[0] + '.pkl')
    
    # Load the Excel file
    df = pd.read_excel(excel_file_path)
    
    # Save the DataFrame as a pickle file
    df.to_pickle(pickle_file_path)
    
    print(f"Excel file has been converted and saved as: {pickle_file_path}")


