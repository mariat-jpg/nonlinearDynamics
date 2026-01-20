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
