import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import umap
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from data import get_data


def apply_dimensionality_reduction(X_data, method="pca", n_components=2, random_state=42, **kwargs):
    """
    Apply dimensionality reduction to data.

    Args:
        X_data (np.ndarray): Input data
        method (str): 'pca', 'umap', or 'none'
        n_components (int): Number of components to keep
        random_state (int): Random state for reproducibility
        **kwargs: Additional parameters for the reduction method

    Returns:
        np.ndarray: Reduced data
        object: The fitted dimensionality reduction object
        dict: Information about the reduction
    """
    if method == "none":
        return X_data, None, {"method": "none", "explained_variance": None}

    elif method == "pca":
        if n_components > 1 and n_components <= 1.0:
            # Use variance ratio
            pca = PCA(n_components=n_components, random_state=random_state)
        else:
            # Use specific number of components
            pca = PCA(n_components=n_components, random_state=random_state)

        X_reduced = pca.fit_transform(X_data)

        info = {
            "method": "pca",
            "explained_variance": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
            "n_components": pca.n_components_,
        }

        return X_reduced, pca, info

    elif method == "umap":
        # Default UMAP parameters
        umap_params = {
            "n_neighbors": kwargs.get("n_neighbors", 15),
            "min_dist": kwargs.get("min_dist", 0.1),
            "metric": kwargs.get("metric", "euclidean"),
            "random_state": random_state,
        }

        reducer = umap.UMAP(n_components=n_components, **umap_params)
        X_reduced = reducer.fit_transform(X_data)

        info = {
            "method": "umap",
            "n_components": n_components,
            "n_neighbors": umap_params["n_neighbors"],
            "min_dist": umap_params["min_dist"],
        }

        return X_reduced, reducer, info

    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")


def train_kmeans_model(n_clusters=5, random_state=42):
    """
    Train a KMeans model on the CLEF data.

    Args:
        n_clusters (int): Number of clusters to form
        random_state (int): Random state for reproducibility

    Returns:
        KMeans: Trained KMeans model
        np.ndarray: Training data
        np.ndarray: Test data
        float: Silhouette score
    """
    data_path = os.getenv("CLEF_DATA_PATH_UNFEDERATED")
    if data_path is None:
        raise ValueError("CLEF_DATA_PATH_UNFEDERATED environment variable is not set")
    # Load the preprocessed data
    X_train, y_train, X_test, y_test, y_train_original, y_test_original = get_data(data_path)

    # The data is already loaded as numpy arrays
    X_train_np = X_train
    X_test_np = X_test

    # Create and train the KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X_train_np)

    # Calculate silhouette score on training data
    silhouette_avg = silhouette_score(X_train_np, kmeans.labels_)

    print(f"KMeans training completed with {n_clusters} clusters")
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    print(f"Inertia: {kmeans.inertia_:.4f}")

    return kmeans, X_train_np, X_test_np, silhouette_avg


def evaluate_kmeans(kmeans, X_train, X_test):
    """
    Evaluate the trained KMeans model.

    Args:
        kmeans (KMeans): Trained KMeans model
        X_train (np.ndarray): Training data
        X_test (np.ndarray): Test data
    """
    # Get cluster assignments for training and test data
    train_clusters = kmeans.predict(X_train)
    test_clusters = kmeans.predict(X_test)

    print("\nCluster Analysis:")
    print(f"Training data - Cluster distribution: {np.bincount(train_clusters)}")
    print(f"Test data - Cluster distribution: {np.bincount(test_clusters)}")

    # Calculate cluster centers distance
    cluster_centers = kmeans.cluster_centers_
    print(f"Number of cluster centers: {len(cluster_centers)}")
    print(f"Cluster centers shape: {cluster_centers.shape}")

    return train_clusters, test_clusters


def find_optimal_clusters(X_train, max_clusters=10):
    """
    Find the optimal number of clusters using elbow method and silhouette analysis.

    Args:
        X_train (np.ndarray): Training data
        max_clusters (int): Maximum number of clusters to try

    Returns:
        list: Inertia values for each k
        list: Silhouette scores for each k
    """
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_clusters + 1)

    print(f"\nFinding optimal number of clusters (2 to {max_clusters}):")

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_train)

        inertias.append(kmeans.inertia_)
        silhouette_avg = silhouette_score(X_train, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)

        print(f"k={k}: Inertia={kmeans.inertia_:.4f}, Silhouette={silhouette_avg:.4f}")

    return inertias, silhouette_scores


def plot_elbow_curve(k_range, inertias, silhouette_scores):
    """
    Plot the elbow curve and silhouette scores.

    Args:
        k_range (range): Range of k values
        inertias (list): Inertia values
        silhouette_scores (list): Silhouette scores
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Elbow curve
    ax1.plot(k_range, inertias, "bo-")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method")
    ax1.grid(True)

    # Silhouette scores
    ax2.plot(k_range, silhouette_scores, "ro-")
    ax2.set_xlabel("Number of clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Analysis")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def analyze_feature_ranges(X_data, feature_names=None):
    """
    Print min and max values for each feature as a table.

    Args:
        X_data (np.ndarray): Data to analyze
        feature_names (list): Optional feature names for better labeling
    """
    print("\n" + "=" * 80)
    print("FEATURE RANGE ANALYSIS")
    print("=" * 80)

    # Calculate statistics
    min_vals = np.min(X_data, axis=0)
    max_vals = np.max(X_data, axis=0)
    ranges = max_vals - min_vals
    std_vals = np.std(X_data, axis=0)
    mean_vals = np.mean(X_data, axis=0)

    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X_data.shape[1])]

    # Print table header
    print(f"{'Feature':<20} {'Min':<12} {'Max':<12} {'Range':<12} {'Std Dev':<12} {'Mean':<12}")
    print("-" * 80)

    # Print each feature
    for i, (name, min_val, max_val, range_val, std_val, mean_val) in enumerate(
        zip(feature_names, min_vals, max_vals, ranges, std_vals, mean_vals)
    ):
        print(f"{name:<20} {min_val:<12.4f} {max_val:<12.4f} {range_val:<12.4f} {std_val:<12.4f} {mean_val:<12.4f}")

    # Summary statistics
    print("-" * 80)
    print(f"Total features: {X_data.shape[1]}")
    print(f"Features with zero variance: {np.sum(std_vals < 1e-10)}")
    print(f"Features with very small range (< 0.01): {np.sum(ranges < 0.01)}")
    print("=" * 80)


def plot_interactive_3d_clusters(
    kmeans, X_data, reduction_method="pca", title="Interactive 3D Clustering", **reduction_kwargs
):
    """
    Create an interactive 3D visualization of clustered data using configurable dimensionality reduction.

    Args:
        kmeans (KMeans): Trained KMeans model
        X_data (np.ndarray): Data to visualize
        reduction_method (str): 'pca', 'umap', or 'none'
        title (str): Plot title
        **reduction_kwargs: Additional parameters for dimensionality reduction
    """
    # Apply dimensionality reduction
    X_reduced, reducer, info = apply_dimensionality_reduction(
        X_data, method=reduction_method, n_components=3, **reduction_kwargs
    )

    # Transform cluster centers to reduced space
    if reduction_method == "none":
        centers_reduced = kmeans.cluster_centers_
    else:
        centers_reduced = reducer.transform(kmeans.cluster_centers_)

    # Get cluster labels
    cluster_labels = kmeans.predict(X_data)

    # Create the interactive 3D plot
    fig = go.Figure()

    # Add data points for each cluster
    for cluster_id in range(kmeans.n_clusters):
        mask = cluster_labels == cluster_id
        fig.add_trace(
            go.Scatter3d(
                x=X_reduced[mask, 0],
                y=X_reduced[mask, 1],
                z=X_reduced[mask, 2],
                mode="markers",
                name=f"Cluster {cluster_id}",
                marker=dict(size=5, opacity=0.7, color=cluster_id, colorscale="Viridis"),
                text=[f"Point {i}<br>Cluster: {cluster_id}" for i in np.where(mask)[0]],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    # Add cluster centers
    fig.add_trace(
        go.Scatter3d(
            x=centers_reduced[:, 0],
            y=centers_reduced[:, 1],
            z=centers_reduced[:, 2],
            mode="markers",
            name="Cluster Centers",
            marker=dict(size=15, symbol="x", color="red", line=dict(width=3, color="darkred")),
            text=[f"Center {i}" for i in range(len(centers_reduced))],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    # Set axis labels based on method
    if reduction_method == "pca":
        axis_labels = [
            f'PC1 ({info["explained_variance"][0]:.1%} variance)',
            f'PC2 ({info["explained_variance"][1]:.1%} variance)',
            f'PC3 ({info["explained_variance"][2]:.1%} variance)',
        ]
        print(f"3D PCA explained variance: {sum(info['explained_variance']):.1%}")
        print(
            f"PC1: {info['explained_variance'][0]:.1%}, PC2: {info['explained_variance'][1]:.1%}, PC3: {info['explained_variance'][2]:.1%}"
        )
    elif reduction_method == "umap":
        axis_labels = ["UMAP1", "UMAP2", "UMAP3"]
        print(f"UMAP reduction with {info['n_neighbors']} neighbors, min_dist={info['min_dist']}")
    else:
        axis_labels = ["Dim1", "Dim2", "Dim3"]
        print("No dimensionality reduction applied")

    # Update layout
    fig.update_layout(
        title=f"{title} ({reduction_method.upper()})",
        scene=dict(xaxis_title=axis_labels[0], yaxis_title=axis_labels[1], zaxis_title=axis_labels[2]),
        width=800,
        height=600,
        showlegend=True,
        hovermode="closest",
    )

    # Show the interactive plot
    fig.show()


def plot_2d_clusters(kmeans, X_data, reduction_method="pca", title="2D Clustering Visualization", **reduction_kwargs):
    """
    Create a 2D visualization of clustered data using configurable dimensionality reduction.

    Args:
        kmeans (KMeans): Trained KMeans model
        X_data (np.ndarray): Data to visualize
        reduction_method (str): 'pca', 'umap', or 'none'
        title (str): Plot title
        **reduction_kwargs: Additional parameters for dimensionality reduction
    """
    # Apply dimensionality reduction
    X_reduced, reducer, info = apply_dimensionality_reduction(
        X_data, method=reduction_method, n_components=2, **reduction_kwargs
    )

    # Transform cluster centers to reduced space
    if reduction_method == "none":
        centers_reduced = kmeans.cluster_centers_
    else:
        centers_reduced = reducer.transform(kmeans.cluster_centers_)

    # Get cluster labels
    cluster_labels = kmeans.predict(X_data)

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Plot data points colored by cluster
    scatter = plt.scatter(
        X_reduced[:, 0], X_reduced[:, 1], c=cluster_labels, cmap="viridis", alpha=0.6, s=50, label="Data points"
    )

    # Plot cluster centers
    plt.scatter(
        centers_reduced[:, 0], centers_reduced[:, 1], c="red", marker="x", s=200, linewidths=3, label="Cluster centers"
    )

    # Set axis labels based on method
    if reduction_method == "pca":
        xlabel = f'PC1 ({info["explained_variance"][0]:.1%} variance explained)'
        ylabel = f'PC2 ({info["explained_variance"][1]:.1%} variance explained)'
        print(f"2D PCA explained variance: {sum(info['explained_variance']):.1%}")
        print(f"PC1: {info['explained_variance'][0]:.1%}, PC2: {info['explained_variance'][1]:.1%}")
    elif reduction_method == "umap":
        xlabel = "UMAP1"
        ylabel = "UMAP2"
        print(f"UMAP reduction with {info['n_neighbors']} neighbors, min_dist={info['min_dist']}")
    else:
        xlabel = "Dim1"
        ylabel = "Dim2"
        print("No dimensionality reduction applied")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title} ({reduction_method.upper()})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add colorbar
    plt.colorbar(scatter, label="Cluster")

    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to run KMeans clustering on CLEF data.
    """
    print("Starting KMeans clustering on CLEF dataset...")

    # Train KMeans with default parameters
    n_clusters = 4
    kmeans, X_train, X_test, silhouette = train_kmeans_model(n_clusters=n_clusters)

    # Analyze feature ranges to understand data characteristics
    analyze_feature_ranges(X_train)

    # Evaluate the model
    train_clusters, test_clusters = evaluate_kmeans(kmeans, X_train, X_test)

    # Visualize clusters with different dimensionality reduction methods
    print("\n" + "=" * 60)
    print("2D VISUALIZATIONS")
    print("=" * 60)

    # 2D PCA
    plot_2d_clusters(kmeans, X_train, reduction_method="pca", title=f"KMeans Clustering ({n_clusters} clusters)")

    # 2D UMAP
    plot_2d_clusters(
        kmeans,
        X_train,
        reduction_method="umap",
        title=f"KMeans Clustering ({n_clusters} clusters)",
        n_neighbors=15,
        min_dist=0.1,
    )

    print("\n" + "=" * 60)
    print("3D VISUALIZATIONS")
    print("=" * 60)

    # # 3D PCA
    # plot_interactive_3d_clusters(kmeans, X_train, reduction_method='pca',
    #                             title="Interactive 3D KMeans Clustering")

    # 3D UMAP
    plot_interactive_3d_clusters(
        kmeans, X_train, reduction_method="umap", title="Interactive 3D KMeans Clustering", n_neighbors=15, min_dist=0.1
    )

    # Find optimal number of clusters
    max_clusters = 30
    inertias, silhouette_scores = find_optimal_clusters(X_train, max_clusters=max_clusters)

    # Plot results (uncomment to display plots)
    k_range = range(2, max_clusters + 1)
    plot_elbow_curve(k_range, inertias, silhouette_scores)

    print("\nKMeans clustering analysis completed!")


if __name__ == "__main__":
    main()
