"""
This script plots the distribution of subjects across partitions for a given CSV file.

example usage:
python plot_partition_distribution.py --input /path/to/csv/partitioning_2.csv --output /path/to/output/directory
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_partition_distribution(csv_path, output_dir=None):
    """Create a histogram of partition distribution from a CSV file.

    Args:
        csv_path (str): Path to CSV file containing Subject_ID and Partition_ID
        output_dir (str, optional): Directory to save the plot. If None, shows plot instead.
    """
    # Read CSV file
    df = pd.read_csv(csv_path)

    # Count number of subjects per partition
    partition_counts = df["Partition_ID"].value_counts().sort_index()

    # Create figure
    plt.figure(figsize=(10, 6))

    # Create histogram
    bars = plt.bar(partition_counts.index, partition_counts.values)

    # Add value labels on top of each bar with smaller font size
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0, height, f"{int(height)}", ha="center", va="bottom", fontsize=8
        )  # Reduced font size

    # Customize plot
    fname = os.path.basename(csv_path)
    plt.title(f"Distribution of subjects across partitions for {fname}")
    plt.xlabel("Partition ID")
    plt.ylabel("Number of Subjects")
    plt.grid(True, alpha=0.3)

    # Set x-axis to show integer ticks
    plt.xticks(partition_counts.index)

    # Add total count in top right
    total_subjects = len(df)
    plt.text(
        0.95,
        0.95,
        f"Total Subjects: {total_subjects}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    if output_dir:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "partition_distribution.png")
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot partition distribution from CSV file")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path")
    parser.add_argument("--output", "-o", help="Output directory for plot (optional)")

    args = parser.parse_args()

    plot_partition_distribution(args.input, args.output)
