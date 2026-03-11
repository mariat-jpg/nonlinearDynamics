import scipy.io
import numpy as np
import networkx as nx

def sliding_window_connectivity(time_series, window_size, step_size, sigma=None):
    """
    time_series: shape (T, N)
    window_size: number of timepoints per window
    step_size: shift between windows
    sigma: std of Gaussian (default = window_size/3)
    """
    num_time_points, num_regions = time_series.shape
    num_windows = (num_time_points - window_size) // step_size + 1
    
    connectivity_matrices = []
    connectivity_matrices = np.zeros((num_windows, num_regions, num_regions))
    
    # Default sigma
    if sigma is None:
        sigma = window_size / 3
    
    # Create Gaussian window
    x = np.arange(window_size)
    center = (window_size - 1) / 2

    gaussian_window = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    # Normalize (optional but recommended)
    gaussian_window = gaussian_window / np.sum(gaussian_window)
    
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        
        window_ts = time_series[start:end]
        
        # Apply Gaussian taper
        window_ts_weighted = window_ts * gaussian_window[:, None]
        
        # Correlation matrix
        connectivity_matrices[i] = np.corrcoef(window_ts_weighted, rowvar=False) 
        
    return connectivity_matrices

def spectral_partition(connectivity_matrix):
    W = np.abs(connectivity_matrix)
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler_vector = eigenvectors[:, 1] #second smallest eigenvector 
    cluster_labels = np.where(fiedler_vector >= 0, 1, 0) #binary partition

    return cluster_labels, fiedler_vector

def spectral_clustering(connectivity_matrices):
    num_windows = connectivity_matrices.shape[0]
    N = connectivity_matrices.shape[1]  

    all_clusters = np.zeros((num_windows, N))
    all_fiedlers = np.zeros((num_windows, N))

    for i in range(num_windows):
        cluster_labels, fiedler_vector = spectral_partition(connectivity_matrices[i])
        all_clusters[i] = cluster_labels
        all_fiedlers[i] = fiedler_vector

    return all_clusters, all_fiedlers

def compute_roi_switches(cluster_matrix):
    return np.sum(np.abs(np.diff(cluster_matrix, axis=0)), axis=0)

def compute_window_measures(connectivity_matrices, threshold=0.3):
    """
    For each window, compute graph-theoretic measures.
    connectivity_matrices: shape (num_windows, N, N)
    Returns a dict of measures, each of shape (num_windows,) or (num_windows, N) for node-level measures
    """
    num_windows = connectivity_matrices.shape[0]

    clustering_coeffs  = np.zeros(num_windows)
    path_lengths       = np.zeros(num_windows)
    global_efficiencies = np.zeros(num_windows)
    modularities       = np.zeros(num_windows)
    node_strengths     = np.zeros((num_windows, connectivity_matrices.shape[1]))

    for i in range(num_windows):
        # threshold the matrix to keep only strong connections
        matrix = connectivity_matrices[i].copy()
        matrix[np.abs(matrix) < threshold] = 0
        np.fill_diagonal(matrix, 0)  # remove self-connections

        G = nx.from_numpy_array(np.abs(matrix))  # use absolute values for weights

        # 1. Clustering coefficient
        clustering_coeffs[i] = nx.average_clustering(G, weight='weight')

        # 2. Path length and Global efficiency (only if graph is connected)
        if nx.is_connected(G):
            path_lengths[i] = nx.average_shortest_path_length(G, weight='weight')
        else:
            # use largest connected component if graph is disconnected
            largest_cc = G.subgraph(max(nx.connected_components(G), key=len))
            path_lengths[i] = nx.average_shortest_path_length(largest_cc, weight='weight')

        # 3. Global efficiency
        global_efficiencies[i] = nx.global_efficiency(G)

        # 4. Modularity
        communities = nx.community.greedy_modularity_communities(G, weight='weight')
        modularities[i] = nx.community.modularity(G, communities, weight='weight')

        # 5. Node strength (sum of each node's edge weights)
        node_strengths[i] = np.sum(np.abs(matrix), axis=1)

    return {
        'clustering':       clustering_coeffs,    # (num_windows,)
        'path_length':      path_lengths,          # (num_windows,)
        'global_efficiency': global_efficiencies,  # (num_windows,)
        'modularity':       modularities,          # (num_windows,)
        'node_strength':    node_strengths         # (num_windows, N)
    }


def average_measures(measures_list):
    """
    Average graph measures across subjects.
    measures_list: list of measure dicts, one per subject
    """
    avg = {}
    for key in measures_list[0]:
        avg[key] = np.mean([m[key] for m in measures_list], axis=0)
    return avg