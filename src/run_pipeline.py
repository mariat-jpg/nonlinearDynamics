import numpy as np
import matplotlib.pyplot as plt
from src.load_data import load_fmri, preprocess_fmri, load_atlas
from src.extract_bold import extract_bold_timeseries
import os

print("RUNNING UPDATED run_pipeline.py")
#load and preprocess the fMRI
fmri_path = "dataset/sub-17017/func/sub-17017_task-rest_bold.nii.gz"
fmri_img = load_fmri(fmri_path)
fmri_clean = preprocess_fmri(fmri_img)
#load the atlas
atlas_img, atlas_labels = load_atlas()
#extract ROI time series
time_series = extract_bold_timeseries(fmri_clean, atlas_img)
#analysing the shape of the time series
print("Time series shape:", time_series.shape)

roi_indices = [0, 1, 2, 3, 4]
plt.figure(figsize=(10, 5))
for roi in roi_indices:
    plt.plot(time_series[:, roi], label=atlas_labels[roi])

plt.xlabel("Time points")
plt.ylabel("BOLD signal (standardized)")
plt.title("ROI-wise BOLD Time Series")
plt.legend()
plt.tight_layout()

os.makedirs("results/plots", exist_ok=True)
plt.savefig("results/plots/roi_time_series.png", dpi=300)
print("Saved ROI plot to results/plots/roi_time_series.png")

print("ENTERING CONNECTIVITY STEP")

#Pearsons Correlation Connectivity Matrix
corr_matrix = np.corrcoef(time_series.T)
print("Connectivity matrix shape:", corr_matrix.shape)
os.makedirs("results/matrices", exist_ok=True)
np.save("results/matrices/pearson_connectivity.npy", corr_matrix)
print("Connectivity matrix saved to results/matrices/")

plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Pearson Correlation")
plt.title("Functional Connectivity Matrix (Pearson)")
plt.xlabel("ROI Index")
plt.ylabel("ROI Index")

os.makedirs("results/plots", exist_ok=True)
plt.savefig("results/plots/connectivity_matrix.png", dpi=300)
plt.close()
print("Connectivity matrix heatmap saved to results/plots/")

