from src.network import network
from src.visualization import time_series, corr_heatmap
from src.load_data import load_fmri, preprocess_fmri, load_atlas
from src.extract_bold import extract_bold_timeseries
from src.connectivity import compute_connectivity_matrix
from nilearn.image import mean_img, index_img
from nilearn.plotting import plot_epi, plot_roi, plot_matrix, plot_connectome, show, find_parcellation_cut_coords, find_xyz_cut_coords
import matplotlib.pyplot as plt
import numpy as np

pd_regions = [
        "Hippocampus_L", 
        "Hippocampus_R", 
        "Amygdala_L", 
        "Amygdala_R", 
        "Caudate_L", 
        "Caudate_R", 
        "Putamen_L", 
        "Putamen_R", 
        "Thalamus_L",
        "Thalamus_R", 
        "Olfactory_L",
        "Olfactory_R",
        "Pallidum_L",
        "Pallidum_R",
        "Precentral_L",
        "Precentral_R",
        "Postcentral_L",
        "Postcentral_R",
        "Cingulum_Ant_L",
        "Cingulum_Ant_R",
        "Frontal_Mid_L",
        "Frontal_Mid_R",
        "Supp_Motor_Area_L",
        "Supp_Motor_Area_R"
    ]

fmri_path = "dataset/healthy/func/sub-0222_ses-01_task-rest_bold.nii.gz"
fmri_path_pd = "dataset/parkinsons/func/sub-0231_ses-01_task-rest_bold.nii.gz"

#loading healthy fMRI data
fmri_img = load_fmri(fmri_path)
fmri_clean = preprocess_fmri(fmri_img)
#loading parkinsons fMRI data
fmri_img_pd = load_fmri(fmri_path_pd)
fmri_clean_pd = preprocess_fmri(fmri_img_pd)

atlas_maps, atlas_labels = load_atlas()

pd_indices = [atlas_labels.index(region) for region in pd_regions if region in atlas_labels]

#BOLD time series for healthy 
bold_ts = extract_bold_timeseries(fmri_clean, atlas_maps)
bold_ts = bold_ts[:, pd_indices]  #selecting only parkinsons relevant ROIs
np.save("outputs/bold_timeseries.npy", bold_ts)
print("BOLD time series shape (healthy): ", bold_ts.shape)
#BOLD time series for parkinsons
bold_ts_pd = extract_bold_timeseries(fmri_clean_pd, atlas_maps)
bold_ts_pd = bold_ts_pd[:, pd_indices]
np.save("outputs/bold_timeseries_parkinsons.npy", bold_ts_pd)
print("BOLD time series shape (parkinsons): ", bold_ts_pd.shape)

#plotting the time series of first 5 ROIs
time_series(atlas_labels, bold_ts, bold_ts_pd)

#pearson correlation heatmap 
corr_heatmap(bold_ts, bold_ts_pd)

#mean of all 3D fMRI images bc plot_epi only accepts 3D image
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))
healthy = plot_epi(mean_img(fmri_clean), axes = axes[0], title="Mean fMRI Image (Healthy)")
parkinsons = plot_epi(mean_img(fmri_clean_pd), axes = axes[1], title="Mean fMRI Image (Parkinsons)")
show()

correlation_matrix_healthy = compute_connectivity_matrix(bold_ts)
correlation_matrix_parkinsons = compute_connectivity_matrix(bold_ts_pd)

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))
healthy_matrix = plot_matrix(
    correlation_matrix_healthy,
    labels = pd_regions, #atlas_labels,
    vmax = 0.8, 
    vmin = -0.8, 
    colorbar = True, 
    title = "Functional Connectivity Matrix (Pearson) for Healthy Subject",
    axes = axes[0]
)
parkinsons_matrix = plot_matrix(
    correlation_matrix_parkinsons, 
    labels = pd_regions, #atlas_labels,
    vmax = 0.8, 
    vmin = -0.8, 
    colorbar = True, 
    title = "Functional Connectivity Matrix (Pearson) for Parkinsons Subject",
    axes = axes[1]
)
show()

network(correlation_matrix_healthy, correlation_matrix_parkinsons, atlas_labels)

#visualizing strongest 10% of the connections between PD ROIs
#basically visualization of the subgraph
coords_connectome = find_parcellation_cut_coords(atlas_maps)
coords_connectome_pd = coords_connectome[pd_indices]
n_rois = len(pd_indices)
conn_sub_h = correlation_matrix_healthy[:n_rois, :n_rois]
coords_sub_h = coords_connectome_pd[:n_rois]
labels_sub = pd_regions[:n_rois] #atlas_labels[:n_rois]
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(10, 8))
h_conn = plot_connectome(
    conn_sub_h, 
    coords_sub_h, 
    edge_threshold = "90%",
    node_size = 50,
    edge_cmap = "coolwarm", 
    axes = axes[0],
    title = "Correlation Between Top 20 ROIs (Healthy Subject)"
)
conn_sub_p = correlation_matrix_parkinsons[:n_rois, :n_rois]
coords_sub_p = coords_connectome_pd[:n_rois]
p_conn = plot_connectome(
    conn_sub_p, 
    coords_sub_p, 
    edge_threshold = "90%",
    node_size = 50,
    edge_cmap = "coolwarm", 
    axes = axes[1],
    title = "Correlation Between Top 20 ROIs (Parkinsons Subject)"
)
show()