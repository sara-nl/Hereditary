"""
Script to partition the CLEF dataset into n equal partitions.
"""
import pandas as pd
import os
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Partition CLEF dataset into multiple parts')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the CLEF dataset directory')
    parser.add_argument('--partitions', type=str, required=True,
                      help='Comma-separated list of partition numbers (e.g., "2,4,6,10")')
    
    args = parser.parse_args()
    
    # Convert partition string to list of integers
    num_partitions = [int(x.strip()) for x in args.partitions.split(',')]
    
    return args.data_path, num_partitions


def partition_data(train_statistics, test_statistics, train_visits, test_visits, train_y, test_y, num_partitions, shuffle=False):
    """Partition training and test data into n equal parts.
    
    Args:
        train_statistics (pd.DataFrame): Training static data
        test_statistics (pd.DataFrame): Test static data
        train_visits (pd.DataFrame): Training visits data
        test_visits (pd.DataFrame): Test visits data
        train_y (pd.DataFrame): Training labels
        test_y (pd.DataFrame): Test labels
        num_partitions (int): Number of partitions to create
        shuffle (bool, optional): Whether to shuffle data before partitioning. Defaults to False.
        
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
    
    # Get unique patient IDs
    train_patient_ids = train_stats_df['PatientID'].unique()
    test_patient_ids = test_stats_df['PatientID'].unique()
    
    # Shuffle patient IDs if requested
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(train_patient_ids)
        np.random.shuffle(test_patient_ids)
    
    # Calculate partition sizes for patients
    train_partition_size = len(train_patient_ids) // num_partitions
    test_partition_size = len(test_patient_ids) // num_partitions
    
    # Create base directory for partitions if it doesn't exist
    partition_dir = os.path.join(DATA_PATH, f"partitions_{num_partitions}")
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
        train_end_idx = train_start_idx + train_partition_size if i < num_partitions-1 else len(train_patient_ids)
        
        # Calculate patient indices for test data
        test_start_idx = i * test_partition_size
        test_end_idx = test_start_idx + test_partition_size if i < num_partitions-1 else len(test_patient_ids)
        
        # Get patient IDs for this partition
        train_partition_ids = train_patient_ids[train_start_idx:train_end_idx]
        test_partition_ids = test_patient_ids[test_start_idx:test_end_idx]
        
        # Filter data for these patients
        # Statistics
        train_stats_partition = train_stats_df[train_stats_df['PatientID'].isin(train_partition_ids)]
        test_stats_partition = test_stats_df[test_stats_df['PatientID'].isin(test_partition_ids)]
        
        # Visits
        train_visits_partition = train_visits_df[train_visits_df['PatientID'].isin(train_partition_ids)]
        test_visits_partition = test_visits_df[test_visits_df['PatientID'].isin(test_partition_ids)]
        
        # Labels
        train_y_partition = train_y_df[train_y_df['PatientID'].isin(train_partition_ids)]
        test_y_partition = test_y_df[test_y_df['PatientID'].isin(test_partition_ids)]
        
        # Save partitions
        train_stats_partition.to_csv(
            os.path.join(partition_subdir, "train_statistics.csv"),
            index=False
        )
        test_stats_partition.to_csv(
            os.path.join(partition_subdir, "test_statistics.csv"),
            index=False
        )
        
        train_visits_partition.to_csv(
            os.path.join(partition_subdir, "train_visits.csv"),
            index=False
        )
        test_visits_partition.to_csv(
            os.path.join(partition_subdir, "test_visits.csv"),
            index=False
        )
        
        train_y_partition.to_csv(
            os.path.join(partition_subdir, "train_labels.csv"),
            index=False
        )
        test_y_partition.to_csv(
            os.path.join(partition_subdir, "test_labels.csv"),
            index=False
        )

if __name__ == "__main__":
    # Replace the hardcoded variables with parsed arguments
    DATA_PATH, NUM_PARTITIONS = parse_args()

    train_statistics = pd.read_csv(os.path.join(DATA_PATH, "train", "datasetC_train-static-vars.csv"))
    test_statistics = pd.read_csv(os.path.join(DATA_PATH, "test", "datasetC_test-static-vars.csv"))

    train_visits = pd.read_csv(os.path.join(DATA_PATH, "train", "datasetC_train-visits.csv"))
    test_visits = pd.read_csv(os.path.join(DATA_PATH, "test", "datasetC_test-visits.csv"))  

    train_y = pd.read_csv(os.path.join(DATA_PATH, "train", "datasetC_train-outcome.csv"))
    test_y = pd.read_csv(os.path.join(DATA_PATH, "test", "datasetC_test-outcome.csv"))


    for num_partitions in NUM_PARTITIONS:
        print(f"Partitioning into {num_partitions} partitions")
        partition_data(train_statistics, test_statistics, train_visits, test_visits, train_y, test_y, num_partitions)
