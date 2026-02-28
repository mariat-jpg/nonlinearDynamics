from src.network import network
from src.visualization import time_series
from src.load_data import load_fmri, preprocess_fmri, load_atlas
from src.extract_bold import extract_bold_timeseries
from src.connectivity import compute_connectivity_matrix
from src.dynamic_analysis import sliding_window_connectivity, spectral_partition, spectral_clustering, compute_roi_switches
from nilearn.image import mean_img, index_img
from nilearn.plotting import plot_epi, plot_roi, plot_matrix, plot_connectome, show, find_parcellation_cut_coords, find_xyz_cut_coords
from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def epi_images():
    #mean of all 3D fMRI images bc plot_epi only accepts 3D image
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))
    healthy = plot_epi(mean_img(fmri_clean), axes = axes[0], title="Mean fMRI Image (Healthy)")
    parkinsons = plot_epi(mean_img(fmri_clean_pd), axes = axes[1], title="Mean fMRI Image (Parkinsons)")
    show()

def connectivity_visualization():
    global correlation_matrix_healthy
    global correlation_matrix_parkinsons

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
        labels = pd_regions, #atlas_labels
        vmax = 0.8, 
        vmin = -0.8, 
        colorbar = True, 
        title = "Functional Connectivity Matrix (Pearson) for Parkinsons Subject",
        axes = axes[1]
    )
    show()

def network_visualization():
    network(correlation_matrix_healthy, correlation_matrix_parkinsons, atlas_labels)

def connectome_visualization():
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

if __name__ == "__main__":

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

    fmri_path = "dataset/taowu/sub-control032057/func/sub-control032057_task-resting_bold.nii"
    fmri_path_pd = "dataset/taowu/sub-patient032077/func/sub-patient032077_task-resting_bold.nii"

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

    ch1 = ch2 = 0
    
    print("\nNonlinear Dynamics in Neural Networks\n")

    while (ch1!=3):
        print("\n1. Static Analysis\n2. Dynamic Analysis\n3. Exit")
        ch1 = int(input("Enter your choice: "))
        if (ch1==1):
            while (ch2!=5):
                print("\n1. View EPI images\n2. View Functional Connectivity Matrices\n3. View Functional Connectivity Subgraph\n4. View Top 20 Strongest Functional Connections\n5. Exit")
                ch2 = int(input("Enter your choice: "))
                if ch2==1:
                    epi_images()
                elif ch2==2:
                    connectivity_visualization()
                elif ch2==3:
                    network_visualization()
                elif ch2==4:    
                    connectome_visualization()
                else:
                    print("Exiting...")

        elif (ch1 == 2):
            #Healthy subjects
            healthy_roi_switches = []

            healthy_files = sorted(glob("dataset/taowu/sub-control*/func/*_task-resting_bold.nii"))
            print(f"\nFound {len(healthy_files)} healthy subjects")

            for fmri_path in healthy_files:
                subject_id = os.path.basename(os.path.dirname(os.path.dirname(fmri_path)))  # extracts e.g. sub-control032057

                fmri_img   = load_fmri(fmri_path)
                fmri_clean = preprocess_fmri(fmri_img)
                bold_ts    = extract_bold_timeseries(fmri_clean, atlas_maps)
                bold_ts    = bold_ts[:, pd_indices]

                connectivity_matrices = sliding_window_connectivity(bold_ts, window_size=20, step_size=1)
                all_clusters, all_fiedlers = spectral_clustering(connectivity_matrices)
                roi_switches = compute_roi_switches(all_clusters)

                healthy_roi_switches.append(roi_switches)

            #Parkinson's subjects
            parkinsons_roi_switches = []

            parkinsons_files = sorted(glob("dataset/taowu/sub-patient*/func/*_task-resting_bold.nii"))
            print(f"\nFound {len(parkinsons_files)} Parkinson's subjects")

            for fmri_path in parkinsons_files:
                subject_id = os.path.basename(os.path.dirname(os.path.dirname(fmri_path)))

                fmri_img   = load_fmri(fmri_path)
                fmri_clean = preprocess_fmri(fmri_img)
                bold_ts_pd = extract_bold_timeseries(fmri_clean, atlas_maps)
                bold_ts_pd = bold_ts_pd[:, pd_indices]

                connectivity_matrices = sliding_window_connectivity(bold_ts_pd, window_size=20, step_size=1)
                all_clusters, all_fiedlers = spectral_clustering(connectivity_matrices)
                roi_switches = compute_roi_switches(all_clusters)

                parkinsons_roi_switches.append(roi_switches)

            #Group averages
            avg_healthy     = np.mean(healthy_roi_switches, axis=0)      # shape (N,)
            avg_parkinsons  = np.mean(parkinsons_roi_switches, axis=0)

            print("\nGroup Average")
            print("\nAverage ROI switches (Healthy):")
            for region, val in zip(pd_regions, avg_healthy):
                print(f"  {region}: {val:.2f}")

            print("\nAverage ROI switches (Parkinson's):")
            for region, val in zip(pd_regions, avg_parkinsons):
                print(f"  {region}: {val:.2f}")

            np.save("outputs/avg_roi_switches_healthy.npy", avg_healthy)
            np.save("outputs/avg_roi_switches_parkinsons.npy", avg_parkinsons)
            np.save("outputs/all_roi_switches_healthy.npy", np.array(healthy_roi_switches))
            np.save("outputs/all_roi_switches_parkinsons.npy", np.array(parkinsons_roi_switches))
            print("\nResults saved to outputs/")

        else:
            print("Exiting...")