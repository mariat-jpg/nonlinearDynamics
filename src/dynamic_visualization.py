import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import os

#Animated Connectivity Heatmap + Frame Images
def animate_connectivity(connectivity_matrices, title, save_path):

    os.makedirs("results/animations", exist_ok=True)
    os.makedirs("results/frames/heatmaps", exist_ok=True)

    fig, ax = plt.subplots(figsize=(6,6))

    def update(frame):

        ax.clear()

        matrix = connectivity_matrices[frame]

        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(f"{title} - Window {frame+1}")

        # Save individual frame image
        frame_path = f"results/frames/heatmaps/window_{frame+1:03d}.png"
        plt.savefig(frame_path, dpi=300)

        return [im]

    ani = FuncAnimation(fig, update, frames=len(connectivity_matrices))

    ani.save(save_path, writer="pillow", fps=2)

    plt.close()

    print(f"Saved animation to {save_path}")
    print("Saved individual heatmap frames to results/frames/heatmaps/")


#Animated Network Graph (WITH ROI LABELS)
def animate_network(connectivity_matrices, cluster_matrix, atlas_labels, title, save_path, threshold=0.5):

    os.makedirs("results/animations", exist_ok=True)
    os.makedirs("results/frames/networks", exist_ok=True)

    fig, ax = plt.subplots(figsize=(9,9))

    # Precompute layout using integer nodes first
    base_corr = connectivity_matrices[0].copy()
    base_corr[np.abs(base_corr) < threshold] = 0
    base_graph = nx.from_numpy_array(base_corr)
    int_pos = nx.spring_layout(base_graph, seed=42)

    # Relabel pos keys from integers to region names to match relabelled graph
    mapping = {i: atlas_labels[i] for i in range(len(atlas_labels))}
    pos = {mapping[i]: int_pos[i] for i in int_pos}  # <-- this is the fix

    def update(frame):
        ax.clear()

        corr = connectivity_matrices[frame].copy()
        corr[np.abs(corr) < threshold] = 0

        G = nx.from_numpy_array(corr)
        G = nx.relabel_nodes(G, mapping)

        node_colors = cluster_matrix[frame]

        nx.draw(
            G,
            pos,  # now has string keys matching the relabelled graph
            ax=ax,
            with_labels=True,
            node_color=node_colors,
            cmap=plt.cm.Set1,
            node_size=500,
            font_size=7,
            font_weight="bold",
            edge_color="gray"
        )

        ax.set_title(f"{title} - Window {frame+1}")

        frame_path = f"results/frames/networks/window_{frame+1:03d}.png"
        plt.savefig(frame_path, dpi=300)

    ani = FuncAnimation(fig, update, frames=len(connectivity_matrices))
    ani.save(save_path, writer="pillow", fps=2)
    plt.close()

    print(f"Saved animation to {save_path}")
    print("Saved individual network frames to results/frames/networks/")


#ROI Switching Bar Plot
def plot_roi_switches(switches_healthy, switches_pd, atlas_labels):

    os.makedirs("results/plots", exist_ok=True)

    x = np.arange(len(atlas_labels))

    plt.figure(figsize=(12,5))

    plt.bar(x - 0.2, switches_healthy, width=0.4, label="Healthy")
    plt.bar(x + 0.2, switches_pd, width=0.4, label="Parkinsons")

    plt.xticks(x, atlas_labels, rotation=90)

    plt.ylabel("Number of Community Switches")
    plt.title("ROI Flexibility Comparison")

    plt.legend()
    plt.tight_layout()

    all_vals = list(switches_healthy) + list(switches_pd)
    plt.ylim(min(all_vals) - 5, max(all_vals) + 5)

    save_path = "results/plots/roi_switching_comparison.png"

    plt.savefig(save_path, dpi=300)

    plt.show()

    print(f"ROI switching comparison saved to {save_path}")

#Recurrence Plot 
def plot_recurrence(connectivity_matrices_healthy, connectivity_matrices_pd):
    """
    connectivity_matrices: shape (num_windows, N, N)
    Flattens each window's connectivity matrix into a vector,
    then computes pairwise distance between all windows.
    """
    os.makedirs("results/plots", exist_ok=True)

    def compute_recurrence(connectivity_matrices, threshold_percentile=20):
        num_windows = connectivity_matrices.shape[0]
        states = connectivity_matrices.reshape(num_windows, -1)  # (num_windows, 576)

        # compute all pairwise distances
        distances = np.zeros((num_windows, num_windows))
        for i in range(num_windows):
            for j in range(num_windows):
                distances[i, j] = np.linalg.norm(states[i] - states[j])

        # set threshold as a percentile of all distances
        threshold = np.percentile(distances, threshold_percentile)
        recurrence = (distances < threshold).astype(int)

        return recurrence

    recurrence_healthy = compute_recurrence(connectivity_matrices_healthy)
    recurrence_pd      = compute_recurrence(connectivity_matrices_pd)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(recurrence_healthy, cmap="binary", origin="lower")
    axes[0].set_title("Recurrence Plot (Healthy)")
    axes[0].set_xlabel("Window")
    axes[0].set_ylabel("Window")

    axes[1].imshow(recurrence_pd, cmap="binary", origin="lower")
    axes[1].set_title("Recurrence Plot (Parkinson's)")
    axes[1].set_xlabel("Window")
    axes[1].set_ylabel("Window")

    plt.tight_layout()
    plt.savefig("results/plots/recurrence_plot.png", dpi=300)
    plt.show()
    print("Recurrence plot saved to results/plots/recurrence_plot.png")

def plot_measure_over_time(avg_healthy_measures, avg_parkinsons_measures, measure_name):
    plt.close('all')
    plt.figure(figsize=(12, 4))
    plt.plot(avg_healthy_measures[measure_name], label='Healthy', color='steelblue')
    plt.plot(avg_parkinsons_measures[measure_name], label="Parkinson's", color='tomato')
    plt.xlabel("Window")
    plt.ylabel(measure_name.replace('_', ' ').title())
    plt.title(f"{measure_name.replace('_', ' ').title()} Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/plots/{measure_name}_over_time.png", dpi=300)
    plt.show()