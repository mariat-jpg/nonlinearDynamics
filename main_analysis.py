from src.network import network
from src.visualization import time_series
from src.load_data import load_fmri, preprocess_fmri, load_atlas
from src.extract_bold import extract_bold_timeseries
from src.connectivity import compute_connectivity_matrix
from src.dynamic_analysis import sliding_window_connectivity, spectral_clustering, compute_roi_switches, compute_window_measures, average_measures
from src.dynamic_visualization import animate_connectivity, animate_network, plot_roi_switches, plot_recurrence, plot_measure_over_time
from src.generate_report import generate_report
from nilearn.image import mean_img
from nilearn.plotting import plot_epi, plot_matrix, show
from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def epi_images():
    plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_epi(mean_img(fmri_clean), axes=axes[0], title="Mean fMRI Image (Healthy)")
    plot_epi(mean_img(fmri_clean_pd), axes=axes[1], title="Mean fMRI Image (Parkinsons)")
    show()

def connectivity_visualization():
    global correlation_matrix_healthy
    global correlation_matrix_parkinsons

    correlation_matrix_healthy = compute_connectivity_matrix(bold_ts)
    correlation_matrix_parkinsons = compute_connectivity_matrix(bold_ts_pd)

    plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    plot_matrix(
        correlation_matrix_healthy,
        labels=pd_regions,
        vmax=0.8,
        vmin=-0.8,
        colorbar=True,
        title="Functional Connectivity (Healthy)",
        axes=axes[0]
    )

    plot_matrix(
        correlation_matrix_parkinsons,
        labels=pd_regions,
        vmax=0.8,
        vmin=-0.8,
        colorbar=True,
        title="Functional Connectivity (Parkinsons)",
        axes=axes[1]
    )

    show()

def network_visualization():
    if 'correlation_matrix_healthy' in globals():
        network(correlation_matrix_healthy, correlation_matrix_parkinsons, pd_regions)
    else:
        print("Please run Connectivity Visualization first.")

if __name__ == "__main__":

    pd_regions = [
        "Hippocampus_L", "Hippocampus_R",
        "Amygdala_L", "Amygdala_R",
        "Caudate_L", "Caudate_R",
        "Putamen_L", "Putamen_R",
        "Thalamus_L", "Thalamus_R",
        "Olfactory_L", "Olfactory_R",
        "Pallidum_L", "Pallidum_R",
        "Precentral_L", "Precentral_R",
        "Postcentral_L", "Postcentral_R",
        "Cingulum_Ant_L", "Cingulum_Ant_R",
        "Frontal_Mid_L", "Frontal_Mid_R",
        "Supp_Motor_Area_L", "Supp_Motor_Area_R"
    ]

    # Load Single Healthy + PD (for static view)
    fmri_path = "dataset/taowu/sub-control032057/func/sub-control032057_task-resting_bold.nii"
    fmri_path_pd = "dataset/taowu/sub-patient032077/func/sub-patient032077_task-resting_bold.nii"

    fmri_img = load_fmri(fmri_path)
    fmri_clean = preprocess_fmri(fmri_img)

    fmri_img_pd = load_fmri(fmri_path_pd)
    fmri_clean_pd = preprocess_fmri(fmri_img_pd)

    atlas_maps, atlas_labels = load_atlas()
    pd_indices = [atlas_labels.index(r) for r in pd_regions if r in atlas_labels]

    bold_ts = extract_bold_timeseries(fmri_clean, atlas_maps)[:, pd_indices]
    bold_ts_pd = extract_bold_timeseries(fmri_clean_pd, atlas_maps)[:, pd_indices]

    time_series(pd_regions, bold_ts, bold_ts_pd)

    print("\nNonlinear Dynamics in Neural Networks")

    while True:

        print("\n1. Static Analysis\n2. Dynamic Analysis\n3. Exit")
        ch1 = int(input("Enter your choice: "))

        if ch1 == 1:
            connectivity_visualization()
            network_visualization()
            epi_images()

        elif ch1 == 2:

            healthy_files = sorted(glob("dataset/taowu/sub-control*/func/*_task-resting_bold.nii"))
            parkinsons_files = sorted(glob("dataset/taowu/sub-patient*/func/*_task-resting_bold.nii"))

            if len(healthy_files) == 0 or len(parkinsons_files) == 0:
                print("No subject files found.")
                continue

            print(f"\nFound {len(healthy_files)} healthy subjects")
            print(f"Found {len(parkinsons_files)} Parkinson's subjects")

            healthy_roi_switches = []
            parkinsons_roi_switches = []

            healthy_measures = []
            parkinsons_measures = []

            all_con_healthy = []   
            all_con_parkinsons = []

            #Healthy 
            for file in healthy_files:
                fmri_img = load_fmri(file)
                fmri_clean = preprocess_fmri(fmri_img)
                bold_sub = extract_bold_timeseries(fmri_clean, atlas_maps)[:, pd_indices]

                conn = sliding_window_connectivity(bold_sub, 20, 1)
                all_con_healthy.append(conn)
                clusters, _ = spectral_clustering(conn)
                switches = compute_roi_switches(clusters)
                meaures = compute_window_measures(conn)

                healthy_roi_switches.append(switches)
                healthy_measures.append(meaures)

            #Parkinson
            for file in parkinsons_files:
                fmri_img = load_fmri(file)
                fmri_clean = preprocess_fmri(fmri_img)
                bold_sub = extract_bold_timeseries(fmri_clean, atlas_maps)[:, pd_indices]

                conn = sliding_window_connectivity(bold_sub, 20, 1)
                all_con_parkinsons.append(conn)
                clusters, _ = spectral_clustering(conn)
                switches = compute_roi_switches(clusters)
                meaures = compute_window_measures(conn)

                parkinsons_roi_switches.append(switches)
                parkinsons_measures.append(meaures)

            avg_healthy = np.mean(healthy_roi_switches, axis=0)
            avg_parkinsons = np.mean(parkinsons_roi_switches, axis=0)

            avg_healthy_measures = average_measures(healthy_measures)
            avg_parkinsons_measures = average_measures(parkinsons_measures)

            avg_con_healthy = np.mean(all_con_healthy, axis=0)
            avg_con_parkinsons = np.mean(all_con_parkinsons, axis=0)

            print("\nAverage ROI Switches (Healthy)")
            for r, v in zip(pd_regions, avg_healthy):
                print(f"{r}: {v:.2f}")

            print("\nAverage ROI Switches (Parkinsons)")
            for r, v in zip(pd_regions, avg_parkinsons):
                print(f"{r}: {v:.2f}")

            print("\nGraph Measures Comparison\n")
            for measure in ['clustering', 'path_length', 'global_efficiency', 'modularity']:
                h_val = avg_healthy_measures[measure].mean()
                pd_val = avg_parkinsons_measures[measure].mean()
                print(f"{measure:<20} Healthy: {h_val:.4f} | Parkinsons: {pd_val:.4f}")
            
            print("\nNode Strength Comparison (average per ROI)\n")
            print(f"{'ROI':<25} {'Healthy':>10} {'Parkinsons':>12}")
            print("-" * 50)
            for i, region in enumerate(pd_regions):
                h_strength = avg_healthy_measures['node_strength'][:, i].mean()
                pd_strength = avg_parkinsons_measures['node_strength'][:, i].mean()
                print(f"{region:<25} {h_strength:>10.4f} {pd_strength:>12.4f}")

            plot_measure_over_time(avg_healthy_measures, avg_parkinsons_measures, 'clustering')
            plot_measure_over_time(avg_healthy_measures, avg_parkinsons_measures, 'global_efficiency')
            plot_measure_over_time(avg_healthy_measures, avg_parkinsons_measures, 'modularity')
            plot_measure_over_time(avg_healthy_measures, avg_parkinsons_measures, 'path_length')

            # Healthy representative
            fmri_img = load_fmri(healthy_files[0])
            fmri_clean = preprocess_fmri(fmri_img)
            bold_rep_h = extract_bold_timeseries(fmri_clean, atlas_maps)[:, pd_indices]

            conn_h = sliding_window_connectivity(bold_rep_h, 20, 1)
            clusters_h, _ = spectral_clustering(conn_h)
            switches_h = compute_roi_switches(clusters_h)

            # Parkinson representative
            fmri_img = load_fmri(parkinsons_files[0])
            fmri_clean = preprocess_fmri(fmri_img)
            bold_rep_p = extract_bold_timeseries(fmri_clean, atlas_maps)[:, pd_indices]

            conn_p = sliding_window_connectivity(bold_rep_p, 20, 1)
            clusters_p, _ = spectral_clustering(conn_p)
            switches_p = compute_roi_switches(clusters_p)

            #Animations and Plots 
            print("\nGenerating animations...")
            animate_connectivity(conn_h, "Healthy Connectivity", "results/animations/healthy_heatmap.gif")
            animate_connectivity(conn_p, "Parkinsons Connectivity", "results/animations/pd_heatmap.gif")

            animate_network(conn_h, clusters_h, pd_regions, "Healthy Network", "results/animations/healthy_network.gif")
            animate_network(conn_p, clusters_p, pd_regions, "Parkinsons Network", "results/animations/pd_network.gif")

            print("avg_healthy mean:", np.mean(avg_healthy))
            print("avg_parkinsons mean:", np.mean(avg_parkinsons))

            plot_roi_switches(avg_healthy, avg_parkinsons, pd_regions)

            plot_recurrence(avg_con_healthy, avg_con_parkinsons)

            generate_report(avg_healthy, avg_parkinsons, avg_healthy_measures, avg_parkinsons_measures, pd_regions)

        elif ch1 == 3:
            print("Exiting...")
            break

        else:
            print("Invalid choice.")