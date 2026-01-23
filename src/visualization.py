import matplotlib.pyplot as plt
import numpy as np
import os

def time_series(atlas_labels, bold_ts, bold_ts_pd):
    roi_indices = [0, 1, 2, 3, 4]
    fig, axes = plt.subplots(nrows = 2, ncols = 1,figsize=(10, 5))
    for roi in roi_indices:
        axes[0].plot(bold_ts[:, roi], label=atlas_labels[roi])
        axes[1].plot(bold_ts_pd[:, roi], label=atlas_labels[roi])
    axes[0].set_xlabel("Time points")
    axes[1].set_xlabel("Time points")
    axes[0].set_ylabel("BOLD signal (standardized)")
    axes[1].set_ylabel("BOLD signal (standardized)")
    axes[0].set_title("ROI-wise BOLD Time Series (Healthy)")
    axes[1].set_title("ROI-wise BOLD Time Series (Parkinsons)")
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    plt.show()

    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/roi_time_series.png", dpi=300)
    print("\nROI plot saved to results/plots/roi_time_series.png")

def corr_heatmap(bold_ts, bold_ts_pd):
    corr_matrix = np.corrcoef(bold_ts.T)
    corr_matrix_pd = np.corrcoef(bold_ts_pd.T)
    print("\nConnectivity matrix shape (Healthy):", corr_matrix.shape)
    print("Connectivity matrix shape (Parkinsons):", corr_matrix_pd.shape)
    os.makedirs("results/matrices", exist_ok=True)
    np.save("results/matrices/pearson_connectivity.npy", corr_matrix) 
    print("Connectivity matrix saved to results/matrices/")

    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(8, 6))
    axes[0].imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_title("Functional Connectivity Matrix (Pearson) - Healthy")
    axes[0].set_xlabel("ROI Index")
    axes[0].set_ylabel("ROI Index")

    axes[1].imshow(corr_matrix_pd, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_title("Functional Connectivity Matrix (Pearson) - Parkinsons")
    axes[1].set_xlabel("ROI Index")
    axes[1].set_ylabel("ROI Index")

    os.makedirs("results/plots", exist_ok=True)
    plt.show()
    plt.savefig("results/plots/connectivity_matrix.png", dpi=300)
    plt.close()
    print("Connectivity matrix heatmap saved to results/plots/")