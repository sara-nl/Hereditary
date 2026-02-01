import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin


def plot_kmeans_step(X, labels, centers, iteration, title, new_centers=None):
    """Plot a single step of the K-means algorithm."""
    plt.figure(figsize=(8, 6))
    
    # Create output directory if it doesn't exist
    output_dir = "kmeans_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot the decision boundary.
    h = 0.02  # step size of the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh
    Z = pairwise_distances_argmin(np.c_[xx.ravel(), yy.ravel()], centers)
    Z = Z.reshape(xx.shape)

    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )

    # Plot the data points
    plt.plot(X[:, 0], X[:, 1], "k.", markersize=4)

    # Plot the centroids
    # Plot new centers and arrows if provided
    if new_centers is not None:
        # Plot old centers in grey
        plt.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            s=100,
            color="grey",
            zorder=10,
        )
        # Plot new centers as white 'x'
        plt.scatter(
            new_centers[:, 0],
            new_centers[:, 1],
            marker="x",
            s=169,
            linewidths=3,
            color="w",
            zorder=10,
        )
        # Draw arrows from old to new centers
        for i in range(centers.shape[0]):
            plt.arrow(centers[i, 0], centers[i, 1], new_centers[i, 0] - centers[i, 0], new_centers[i, 1] - centers[i, 1], 
                      head_width=0.1, head_length=0.1, fc='grey', ec='grey', zorder=9)
    else:
        # Plot initial centroids
        plt.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="x",
            s=169,
            linewidths=3,
            color="w",
            zorder=10,
        )

    plt.title(title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig(os.path.join(output_dir, f"kmeans_step_{iteration}.png"), transparent=True)
    plt.show()


def run_kmeans_visualized():
    """Generate data and run K-means step-by-step, visualizing each step."""
    # 1. Generate synthetic data
    n_samples = 1500
    n_clusters = 4
    X, y_true = make_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        cluster_std=0.3,
        center_box=(-2.0, 2.0),
        random_state=43,
    )

    # 2. Initialize centroids
    rng = np.random.RandomState(43)
    centers_init = X[rng.choice(n_samples, n_clusters, replace=False)]
    centers = np.copy(centers_init)

    # Plot initial state
    plot_kmeans_step(X, None, centers, 0, "Initial Centroids")

    # 3. Run K-means step-by-step
    n_iterations = 5
    for i in range(1, n_iterations + 1):
        # Assignment step
        labels = pairwise_distances_argmin(X, centers)
        plot_kmeans_step(X, labels, centers, i * 2 - 1, f"Iteration {i}: Assignment Step")

        # Update step
        new_centers = np.array([X[labels == j].mean(0) for j in range(n_clusters)])
        plot_kmeans_step(X, labels, centers, i * 2, f"KMeans Iteration {i}: Update Step", new_centers=new_centers)

        # Check for convergence
        if np.all(centers == new_centers):
            print(f"Converged at iteration {i}")
            break
        centers = new_centers

    print("K-means visualization script finished.")

if __name__ == "__main__":
    run_kmeans_visualized()
