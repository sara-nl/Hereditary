"""
Script to partition the CLEF dataset into n equal partitions.
"""

import pandas as pd
import os
import numpy as np
import argparse
import hashlib


def parse_args():
    parser = argparse.ArgumentParser(description="Partition CLEF dataset into multiple parts")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CLEF dataset directory")
    parser.add_argument(
        "--partitions", type=str, required=True, help='Comma-separated list of partition numbers (e.g., "2,4,6,10")'
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Output directory for partitions. If not specified, partitions will be created in the input directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling. Default is 42")
    parser.add_argument(
        "--method",
        type=str,
        choices=["sequential", "by_time"],
        default="sequential",
        help="Method to use for partitioning data",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Whether to shuffle data before partitioning (only applies to sequential method)",
    )

    args = parser.parse_args()

    # Convert partition string to list of integers
    num_partitions = [int(x.strip()) for x in args.partitions.split(",")]

    # Create output directory if specified and doesn't exist
    output_dir = args.output_dir if args.output_dir else args.data_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return args.data_path, num_partitions, output_dir, args.seed, args.method, args.shuffle


def calculate_file_hash(filepath):
    """Calculate SHA-256 hash of a file.

    Args:
        filepath (str): Path to the file

    Returns:
        str: Hexadecimal representation of the file's SHA-256 hash
    """
    with open(filepath, "rb", buffering=0) as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save_configuration(output_dir, data_path, num_partitions, seed, shuffle):
    """Save configuration details to a text file.

    Args:
        output_dir (str): Directory where partitions are saved
        data_path (str): Path to the input data
        num_partitions (list): List of partition numbers
        seed (int): Random seed used for shuffling
        shuffle (bool): Whether shuffling was applied
    """
    config_file = os.path.join(output_dir, "partition_config.txt")

    # Calculate hashes of input files for reproducibility purposes
    input_files = {
        "train_static": os.path.join(data_path, "train", "datasetC_train-static-vars.csv"),
        "test_static": os.path.join(data_path, "test", "datasetC_test-static-vars.csv"),
        "train_visits": os.path.join(data_path, "train", "datasetC_train-visits.csv"),
        "test_visits": os.path.join(data_path, "test", "datasetC_test-visits.csv"),
        "train_outcome": os.path.join(data_path, "train", "datasetC_train-outcome.csv"),
        "test_outcome": os.path.join(data_path, "test", "datasetC_test-outcome.csv"),
    }

    file_hashes = {name: calculate_file_hash(path) for name, path in input_files.items()}

    with open(config_file, "w") as f:
        f.write("CLEF Dataset Partition Configuration\n")
        f.write("==================================\n\n")
        f.write(f"Input Data Path: {data_path}\n")
        f.write(f"Partition Numbers: {num_partitions}\n")
        f.write(f"Random Seed: {seed}\n")
        f.write(f"Shuffle Applied: {shuffle}\n\n")

        f.write("Input File Hashes (SHA-256):\n")
        f.write("---------------------------\n")
        for name, hash_value in file_hashes.items():
            f.write(f"{name}: {hash_value}\n")
        f.write("\n")

        # Save the script content
        f.write("Script Content:\n")
        f.write("==============\n\n")
        with open(__file__, "r") as script:
            f.write(script.read())


def save_partitions_to_csv(data_dict, partition_subdir):
    for name, data in data_dict.items():
        data.to_csv(os.path.join(partition_subdir, name), index=False)


def partition_data(
    train_statistics,
    test_statistics,
    train_visits,
    test_visits,
    train_y,
    test_y,
    num_partitions,
    output_dir,
    method,
    seed,
    shuffle,
    data_path,
):
    """Partition training and test data into n equal parts.

    Args:
        train_statistics (pd.DataFrame): Training static data
        test_statistics (pd.DataFrame): Test static data
        train_visits (pd.DataFrame): Training visits data
        test_visits (pd.DataFrame): Test visits data
        train_y (pd.DataFrame): Training labels
        test_y (pd.DataFrame): Test labels
        num_partitions (int): Number of partitions to create
        output_dir (str): Directory where partitions will be saved
        method (str): Method to use for partitioning data
        seed (int): Random seed for shuffling
        shuffle (bool, optional): Whether to shuffle data before partitioning. Defaults to False.
        data_path (str): Path to the input data

    Returns:
        None: Saves partitioned data to CSV files
    """
    # Make copies to avoid modifying original data
    train_stats_df = train_statistics.copy()
    test_stats_df = test_statistics.copy()
    train_visits_df = train_visits.copy()
    test_visits_df = test_visits.copy()
    train_y_df = train_y.copy()
    test_y_df = test_y.copy()

    # Sort by time to create vertical partitions
    if method == "by_time":
        train_y_df = train_y_df.sort_values(by="Time")
        test_y_df = test_y_df.sort_values(by="Time")

    # Get unique patient IDs
    train_patient_ids = train_y_df["PatientID"].unique()
    test_patient_ids = test_y_df["PatientID"].unique()

    # Shuffle patient IDs if requested
    if method == "sequential" and shuffle:
        np.random.seed(seed)
        np.random.shuffle(train_patient_ids)
        np.random.shuffle(test_patient_ids)

    # Calculate partition sizes for patients
    train_partition_size = len(train_patient_ids) // num_partitions
    test_partition_size = len(test_patient_ids) // num_partitions

    # Create base directory for partitions if it doesn't exist
    partition_dir = os.path.join(output_dir, f"partitions_{num_partitions}")
    if not os.path.exists(partition_dir):
        os.makedirs(partition_dir)

    # Create and save partitions
    for i in range(num_partitions):
        # Create partition subdirectory
        partition_subdir = os.path.join(partition_dir, f"partition_{i}")
        if not os.path.exists(partition_subdir):
            os.makedirs(partition_subdir)

        # Calculate patient indices for training data
        train_start_idx = i * train_partition_size
        train_end_idx = train_start_idx + train_partition_size if i < num_partitions - 1 else len(train_patient_ids)

        # Calculate patient indices for test data
        test_start_idx = i * test_partition_size
        test_end_idx = test_start_idx + test_partition_size if i < num_partitions - 1 else len(test_patient_ids)

        # Get patient IDs for this partition
        train_partition_ids = train_patient_ids[train_start_idx:train_end_idx]
        test_partition_ids = test_patient_ids[test_start_idx:test_end_idx]

        # Filter data for these patients
        train_stats_partition = train_stats_df[train_stats_df["PatientID"].isin(train_partition_ids)]
        test_stats_partition = test_stats_df[test_stats_df["PatientID"].isin(test_partition_ids)]

        train_visits_partition = train_visits_df[train_visits_df["PatientID"].isin(train_partition_ids)]
        test_visits_partition = test_visits_df[test_visits_df["PatientID"].isin(test_partition_ids)]

        train_y_partition = train_y_df[train_y_df["PatientID"].isin(train_partition_ids)]
        test_y_partition = test_y_df[test_y_df["PatientID"].isin(test_partition_ids)]

        # Save partitions
        data_to_save = {
            "train_statistics.csv": train_stats_partition,
            "test_statistics.csv": test_stats_partition,
            "train_visits.csv": train_visits_partition,
            "test_visits.csv": test_visits_partition,
            "train_labels.csv": train_y_partition,
            "test_labels.csv": test_y_partition,
        }

        save_partitions_to_csv(data_to_save, partition_subdir)
        # Save configuration after creating the partition
        save_configuration(partition_dir, data_path, [num_partitions], seed, shuffle)


if __name__ == "__main__":
    DATA_PATH, NUM_PARTITIONS, OUTPUT_DIR, SEED, METHOD, SHUFFLE = parse_args()

    train_statistics = pd.read_csv(os.path.join(DATA_PATH, "train", "datasetC_train-static-vars.csv"))
    test_statistics = pd.read_csv(os.path.join(DATA_PATH, "test", "datasetC_test-static-vars.csv"))

    train_visits = pd.read_csv(os.path.join(DATA_PATH, "train", "datasetC_train-visits.csv"))
    test_visits = pd.read_csv(os.path.join(DATA_PATH, "test", "datasetC_test-visits.csv"))

    train_y = pd.read_csv(os.path.join(DATA_PATH, "train", "datasetC_train-outcome.csv"))
    test_y = pd.read_csv(os.path.join(DATA_PATH, "test", "datasetC_test-outcome.csv"))

    for num_partitions in NUM_PARTITIONS:
        print(f"Partitioning into {num_partitions} partitions")
        partition_data(
            train_statistics,
            test_statistics,
            train_visits,
            test_visits,
            train_y,
            test_y,
            num_partitions,
            OUTPUT_DIR,
            METHOD,
            SEED,
            SHUFFLE,
            DATA_PATH,
        )
