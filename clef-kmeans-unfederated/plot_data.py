import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from data import get_data

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def get_onset_labels(data_path=None):
    """
    Load the original data to extract onset type information.
    
    Args:
        data_path (str): Path to the CLEF data directory
        
    Returns:
        tuple: (train_onset_labels, test_onset_labels) as numpy arrays
        tuple: (train_onset_labels_detailed, test_onset_labels_detailed): detailed onset type, but has missing values
    """        
    # Load original statistics data to get onset information
    train_statistics = pd.read_csv(os.path.join(data_path, "train/datasetC_train-static-vars.csv"))
    test_statistics = pd.read_csv(os.path.join(data_path, "test/datasetC_test-static-vars.csv"))
    
    # Extract onset type for each patient
    onset_columns = ['onset_bulbar', 'onset_limbs', 'onset_axial', 'onset_generalized']
    
    def get_onset_type(row):
        for onset_type in onset_columns:
            if row[onset_type] == 1:
                return onset_type.replace('onset_', '')
        return 'unknown'
    
    train_onset_labels = train_statistics.apply(get_onset_type, axis=1).values
    test_onset_labels = test_statistics.apply(get_onset_type, axis=1).values

    
    return train_onset_labels, test_onset_labels, train_statistics["onset_limb_type"], test_statistics["onset_limb_type"]

def plot_dimensionality_reduction(X_train, X_test, train_onset_labels, test_onset_labels, method='pca', tsne_dims=5):
    """
    Create side-by-side scatter plots of dimensionality reduced data colored by onset type.
    
    Args:
        X_train (np.ndarray): Training data
        X_test (np.ndarray): Test data
        train_onset_labels (np.ndarray): Training onset type labels
        test_onset_labels (np.ndarray): Test onset type labels
        method (str): 'pca', 'umap', or 'tsne'
        tsne_dims (int): Number of PCA components to reduce to before t-SNE
    """
    # Fit dimensionality reduction on training data
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        X_train_reduced = reducer.fit_transform(X_train)
        X_test_reduced = reducer.transform(X_test)
        xlabel = f'PC1 ({reducer.explained_variance_ratio_[0]:.1%} variance)'
        ylabel = f'PC2 ({reducer.explained_variance_ratio_[1]:.1%} variance)'
        method_name = 'PCA'
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
        X_train_reduced = reducer.fit_transform(X_train)
        X_test_reduced = reducer.transform(X_test)
        xlabel = 'UMAP1'
        ylabel = 'UMAP2'
        method_name = 'UMAP'
    elif method == 'tsne':
        # Apply PCA first to reduce noise and speed up t-SNE
        print(f"Reducing dimensionality to {tsne_dims} with PCA before t-SNE...")
        pca = PCA(n_components=min(tsne_dims, X_train.shape[1], X_test.shape[0]), random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # t-SNE perplexity must be less than the number of samples
        reducer = TSNE(n_components=2, perplexity=100, random_state=42)
        X_train_reduced = reducer.fit_transform(X_train_pca)
        
        # t-SNE does not have a transform method, so we fit on test separately
        X_test_reduced = TSNE(n_components=2, perplexity=100, random_state=42).fit_transform(X_test_pca)
        
        xlabel = 't-SNE 1'
        ylabel = 't-SNE 2'
        method_name = f't-SNE (PCA {tsne_dims} dims)'
    else:
        raise ValueError("Method must be 'pca', 'umap', or 'tsne'")
    
    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get unique onset types and colors
    unique_onsets = np.unique(np.concatenate([train_onset_labels, test_onset_labels]))
    colors = ['red', 'blue', 'yellow', 'green'][:len(unique_onsets)]
    
    # Plot training data
    for i, onset_type in enumerate(unique_onsets):
        mask = train_onset_labels == onset_type
        ax1.scatter(X_train_reduced[mask, 0], X_train_reduced[mask, 1], 
                   c=[colors[i]], label=onset_type, alpha=0.7, s=50)
    
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(f'{method_name} - Training Data')
    ax1.grid(True, alpha=0.3)
    
    # Plot test data
    for i, onset_type in enumerate(unique_onsets):
        mask = test_onset_labels == onset_type
        ax2.scatter(X_test_reduced[mask, 0], X_test_reduced[mask, 1], 
                   c=[colors[i]], label=onset_type, alpha=0.7, s=50)
    
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.set_title(f'{method_name} - Test Data')
    ax2.grid(True, alpha=0.3)
    
    # Add legend inside the figure (more robust than placing it outside the canvas)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title='Onset Type',
        loc='lower center',
        ncol=min(len(labels), 4),
        frameon=True,
    )
    
    plt.suptitle(f'{method_name} Visualization of ALS Data Colored by Onset Type', fontsize=14)
    plt.tight_layout(rect=(0, 0.10, 1, 0.95))
    
    # Print statistics
    print(f"\n{method_name} Statistics:")
    if method == 'pca':
        print(f"Explained variance ratio: {reducer.explained_variance_ratio_}")
    elif method == 'umap':
        print(f"UMAP with n_neighbors={reducer.n_neighbors}, min_dist={reducer.min_dist}")
    elif method == 'tsne':
        print(f"t-SNE with perplexity={reducer.perplexity} (after PCA to {tsne_dims} dims)")
    print(f"Training onset type distribution:")
    for onset_type in unique_onsets:
        count = np.sum(train_onset_labels == onset_type)
        percentage = (count / len(train_onset_labels)) * 100
        print(f"  {onset_type}: {count} ({percentage:.1f}%)")
    print(f"Test onset type distribution:")
    for onset_type in unique_onsets:
        count = np.sum(test_onset_labels == onset_type)
        percentage = (count / len(test_onset_labels)) * 100
        print(f"  {onset_type}: {count} ({percentage:.1f}%)")
    
    plt.show()

def plot_interactive_3d_onset(X_train, X_test, train_onset_labels, test_onset_labels, n_neighbors=30, min_dist=0.1):
    """
    Create interactive 3D UMAP plots of ALS data colored by onset type for both train and test.
    """
    print(f"\nGenerating 3D UMAP visualization (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    
    # Fit UMAP on training data
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    X_train_3d = reducer.fit_transform(X_train)
    X_test_3d = reducer.transform(X_test)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=('Training Data', 'Test Data')
    )
    
    # Ensure labels are of a consistent type to avoid sorting errors in np.unique
    train_labels_clean = np.array(train_onset_labels).astype(str)
    test_labels_clean = np.array(test_onset_labels).astype(str)
    
    unique_onsets = np.unique(np.concatenate([train_labels_clean, test_labels_clean]))
    # Use consistent colors with 2D plots
    onset_colors = {
        'bulbar': 'red',
        'limbs': 'blue',
        'axial': 'yellow',
        'generalized': 'green',
        'unknown': 'gray'
    }
    
    # Track which labels we've added to the legend
    added_to_legend = set()

    def add_traces(X_data, labels, col):
        # Also ensure labels in traces are strings for comparison
        labels_str = np.array(labels).astype(str)
        for onset_type in unique_onsets:
            mask = labels_str == onset_type
            if np.any(mask):
                color = onset_colors.get(onset_type, None)
                show_legend = onset_type not in added_to_legend
                if show_legend:
                    added_to_legend.add(onset_type)
                
                fig.add_trace(
                    go.Scatter3d(
                        x=X_data[mask, 0],
                        y=X_data[mask, 1],
                        z=X_data[mask, 2],
                        mode='markers',
                        marker=dict(
                            size=3, 
                            opacity=0.6,
                            color=color
                        ),
                        name=f"Onset: {onset_type}",
                        legendgroup=onset_type,
                        showlegend=show_legend,
                        hovertemplate=f"Onset: {onset_type}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>"
                    ),
                    row=1, col=col
                )

    add_traces(X_train_3d, train_onset_labels, 1)
    add_traces(X_test_3d, test_onset_labels, 2)
    
    # Update layout
    axis_config = dict(
        showbackground=True,
        backgroundcolor="rgb(230, 230,230)",
        gridcolor="rgb(255, 255, 255)",
        zerolinecolor="rgb(255, 255, 255)",
    )
    
    fig.update_layout(
        title=dict(
            text=f"3D UMAP Visualization of ALS Data Colored by Onset Type (n_neighbors={n_neighbors}, min_dist={min_dist})",
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title='UMAP1',
            yaxis_title='UMAP2',
            zaxis_title='UMAP3',
            xaxis=axis_config,
            yaxis=axis_config,
            zaxis=axis_config,
        ),
        scene2=dict(
            xaxis_title='UMAP1',
            yaxis_title='UMAP2',
            zaxis_title='UMAP3',
            xaxis=axis_config,
            yaxis=axis_config,
            zaxis=axis_config,
        ),
        height=800,
        width=1400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.95,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    
    fig.show()

def main():
    """
    Main function to load data and create PCA/UMAP plots colored by onset type.
    """
    print("Loading ALS data and creating onset type visualizations...")
    
    data_path = os.getenv("CLEF_DATA_PATH_UNFEDERATED")
    if data_path is None:
        raise ValueError("CLEF_DATA_PATH_UNFEDERATED environment variable is not set")
    
    # Load preprocessed data for dimensionality reduction
    X_train, y_train, X_test, y_test, y_train_original, y_test_original = get_data(data_path)
    
    # Get onset type labels from original data
    train_onset_labels, test_onset_labels, train_onset_labels_detailed, test_onset_labels_detailed = get_onset_labels(data_path)
    
    # Data is already returned as numpy arrays
    X_train_np = X_train
    X_test_np = X_test


    # Look at explained variance of the PCs
    pca = PCA().fit(X_train_np)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(cumvar, marker='o')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.title('PCA variance curve')
    plt.axhline(0.80, color='r', ls='--')
    plt.show()
    
    print(f"Training data shape: {X_train_np.shape}")
    print(f"Test data shape: {X_test_np.shape}")
    print(f"Training onset labels: {len(train_onset_labels)}")
    print(f"Test onset labels: {len(test_onset_labels)}")
    
    # Create PCA plots (train and test side by side)
    print("\n" + "="*60)
    print("PCA VISUALIZATIONS")
    print("="*60)
    
    plot_dimensionality_reduction(X_train_np, X_test_np, train_onset_labels, test_onset_labels, method='pca')
    
    # Create UMAP plots (train and test side by side)
    print("\n" + "="*60)
    print("UMAP VISUALIZATIONS")
    print("="*60)
    
    plot_dimensionality_reduction(X_train_np, X_test_np, train_onset_labels, test_onset_labels, method='umap')
    
    # Create t-SNE plots (train and test side by side)
    print("\n" + "="*60)
    print("T-SNE VISUALIZATIONS")
    print("="*60)
    
    plot_dimensionality_reduction(X_train_np, X_test_np, train_onset_labels, test_onset_labels, method='tsne')

    # Create 3D UMAP interactive plot
    print("\n" + "="*60)
    print("3D UMAP INTERACTIVE VISUALIZATION")
    print("="*60)
    
    plot_interactive_3d_onset(X_train_np, X_test_np, train_onset_labels, test_onset_labels)

    # Create 3D UMAP interactive plot of detailed onset type
    # detailed onset labels have missing values, first filter x_train and x_test_np based on the missing values in train_onset_labels_detailed and test_onset_labels_detailed
    train_idx = np.where(pd.notna(train_onset_labels_detailed))[0]
    test_idx = np.where(pd.notna(test_onset_labels_detailed))[0]
    
    print("\n" + "="*60)
    print("3D UMAP INTERACTIVE VISUALIZATION (DETAILED ONSET)")
    print("="*60)
    
    plot_interactive_3d_onset(X_train_np[train_idx], X_test_np[test_idx], train_onset_labels_detailed.iloc[train_idx].values, test_onset_labels_detailed.iloc[test_idx].values)
    
    print("\nVisualization completed!")

if __name__ == "__main__":
    main()