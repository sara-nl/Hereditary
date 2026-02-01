import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def calculate_sse(X, k_range):
    """Calculate Sum of Squared Errors for different numbers of clusters."""
    sse = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
    return sse


def plot_elbow_method(k_range, sse):
    """Plot the elbow method curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True, alpha=0.3)
    
    # Highlight the elbow point using angle method
    if len(sse) > 2:
        # Calculate angles between consecutive points
        angles = []
        for i in range(1, len(sse) - 1):
            # Vector from point i-1 to i
            v1 = np.array([k_range[i] - k_range[i-1], sse[i] - sse[i-1]])
            # Vector from point i to i+1
            v2 = np.array([k_range[i+1] - k_range[i], sse[i+1] - sse[i]])
            
            # Calculate angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angles.append(angle)
        
        # Find the point with maximum angle (sharpest turn)
        if angles:
            elbow_idx = np.argmax(angles) + 1  # +1 because angles start from index 1
            if elbow_idx < len(k_range):
                plt.plot(k_range[elbow_idx], sse[elbow_idx], 'ro', markersize=12, 
                        label=f'Elbow point: k={k_range[elbow_idx]}')
                plt.legend()
                print(f"\nDetected elbow point: k={k_range[elbow_idx]}")
                
        # Alternative: Use the "knee" detection method based on distance from line
        # This is often more reliable for elbow detection
        try:
            from kneed import KneeLocator
            knee = KneeLocator(k_range, sse, curve='convex', direction='decreasing')
            if knee.knee is not None:
                knee_idx = k_range.index(knee.knee)
                plt.plot(k_range[knee_idx], sse[knee_idx], 'g^', markersize=12, 
                        label=f'Knee point: k={knee.knee}')
                plt.legend()
                print(f"Alternative knee detection: k={knee.knee}")
        except ImportError:
            pass  # kneed library not available
    
    # Create output directory if it doesn't exist
    output_dir = "kmeans_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.savefig(os.path.join(output_dir, "elbow_method.png"), transparent=True, dpi=300, bbox_inches='tight')
    plt.show()


def run_elbow_method():
    """Generate data and run elbow method analysis."""
    # Generate synthetic data (same as in example_kmeans_vis.py)
    n_samples = 1500
    n_true_clusters = 4
    X, y_true = make_blobs(
        n_samples=n_samples,
        centers=n_true_clusters,
        cluster_std=0.3,
        center_box=(-2.0, 2.0),
        random_state=43,
    )
    
    # Test different numbers of clusters
    k_range = range(1, 11)  # Test k from 1 to 10
    sse = calculate_sse(X, k_range)
    
    # Print SSE values
    print("Number of clusters vs SSE:")
    for k, sse_val in zip(k_range, sse):
        print(f"k={k:2d}: SSE={sse_val:.2f}")
    
    # Plot the elbow method
    plot_elbow_method(k_range, sse)
    
    print(f"\nElbow method analysis complete. Plots saved in 'kmeans_output' directory.")
    print(f"True number of clusters used for data generation: {n_true_clusters}")


if __name__ == "__main__":
    run_elbow_method()