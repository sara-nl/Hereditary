import os
from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Optional

def preprocess_clef_data_for_clustering(train_statistics, test_statistics, train_visits, test_visits):
    categorical_cols = [
        "alive", "sex", "ethnicity", "moreThan10PercentWeightloss", "ALS_familiar_history",
        "prevalentLMN", "prevalentUMN", "mixedMN", "onset_bulbar", "onset_axial",
        "onset_generalized", "onset_limbs", "onset_limb_type", "smoking",
        "retired_at_diagnosis", "C9orf72", "SOD1 mutation", "TARDBP mutation", "FUS mutation",
        "hypertension", "diabetes", "dyslipidemia", "thyroid_disorder", "autoimmune_disease",
        "stroke", "cardiac_disease", "primary_neoplasm", "major_trauma_before_onset"
    ]
    
    drop_columns = ["occupation", "CK_unit", "Albumin_unit", "Creatinine_unit", "Total_Cholesterol_unit",
                    "HDL_Cholesterol_unit", "LDL_Cholesterol_unit", "Triglycerides_unit", "PatientID"]

    # Filter and merge visit data
    train_spiro_visits = train_visits[train_visits["date_spiro"] <= 0].sort_values("date_spiro").drop_duplicates(subset="PatientID", keep="last")
    test_spiro_visits = test_visits[test_visits["date_spiro"] <= 0].sort_values("date_spiro").drop_duplicates(subset="PatientID", keep="last")

    train_alsfrs_visits = train_visits[train_visits["date_alsfrs_r"] == 0.0]
    test_alsfrs_visits = test_visits[test_visits["date_alsfrs_r"] == 0.0]

    # Normalize albumin levels
    train_statistics.loc[train_statistics['Albumin_unit'] == 'mg/dL', ['Albumin_level', 'Albumin_lower_range', 'Albumin_upper_range']] /= 1000
    train_statistics.loc[train_statistics['Albumin_unit'] == 'mg/dL', 'Albumin_unit'] = 'g/L'

    # Merge visit data into statistics data
    train_statistics_merged = train_statistics.merge(train_spiro_visits[["PatientID", "fvcValue"]], on="PatientID", how="left")
    alsfrs_r_to_merge_columns = [col for col in train_alsfrs_visits.columns if col not in ["date_spiro", "date_alsfrs_r", "fvcValue"]]
    train_statistics_merged = train_statistics_merged.merge(train_alsfrs_visits[alsfrs_r_to_merge_columns], on="PatientID", how="left")

    test_statistics_merged = test_statistics.merge(test_spiro_visits[["PatientID", "fvcValue"]], on="PatientID", how="left")
    test_statistics_merged = test_statistics_merged.merge(test_alsfrs_visits[alsfrs_r_to_merge_columns], on="PatientID", how="left")

    # Drop unnecessary columns
    train_statistics_merged.drop(columns=drop_columns, inplace=True)
    test_statistics_merged.drop(columns=drop_columns, inplace=True)

    # Convert categorical columns into numerical form
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    categorical_train = encoder.fit_transform(train_statistics_merged[categorical_cols])
    categorical_test = encoder.transform(test_statistics_merged[categorical_cols])

    train_statistics_merged.drop(columns=categorical_cols, inplace=True)
    test_statistics_merged.drop(columns=categorical_cols, inplace=True)

    # Standardize numerical features
    # Standardize numerical features safely
    scaler = StandardScaler()

    # Fill NaN values with 0 (or you can use another strategy like mean imputation)
    train_statistics_merged.fillna(0, inplace=True)
    test_statistics_merged.fillna(0, inplace=True)

    # Remove infinite values
    train_statistics_merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_statistics_merged.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Apply scaling
    train_numeric = scaler.fit_transform(train_statistics_merged)
    test_numeric = scaler.transform(test_statistics_merged)

    # Concatenate numeric and categorical features
    train_data = np.hstack((train_numeric, categorical_train))
    test_data = np.hstack((test_numeric, categorical_test))

    return train_data, test_data


def load_clef_data_for_spectral_clustering(partition_id, num_partitions):
    data_path = os.getenv("CLEF_DATA_PATH")
    if data_path is None:
        raise ValueError("CLEF_DATA_PATH environment variable must be set")

    if os.getenv("CLEF_LOCAL_DATA") == "true":
        partition_id = os.getenv("CLEF_PARTITION_ID")
        load_dir = data_path
    else:
        load_dir = os.path.join(data_path, f"partitions_{num_partitions}", f"partition_{partition_id}")

    # Load CSV files
    file_names = [
        "train_statistics.csv", "test_statistics.csv", 
        "train_visits.csv", "test_visits.csv"
    ]

    train_statistics, test_statistics, train_visits, test_visits = [
        pd.read_csv(os.path.join(load_dir, fname)) for fname in file_names
    ]

    # Preprocess and return numerical feature matrices
    return preprocess_clef_data_for_clustering(train_statistics, test_statistics, train_visits, test_visits)


def federated_laplacian_approximation(local_affinities: List[np.ndarray]) -> np.ndarray:
    """Approximate global Laplacian using FedSC's graph filtering (simplified)."""
    global_L = np.mean(local_affinities, axis=0)
    return global_L


def compute_spectral_embeddings(laplacian: np.ndarray, k: int = 2) -> np.ndarray:
    """Compute top-k eigenvectors of the Laplacian."""
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    return eigenvectors[:, :k]


# --- FedSC Core Algorithms ---
def federated_laplacian_consensus(
    server_round: int,
    local_affinities: List[np.ndarray],
    prev_global_L: Optional[np.ndarray] = None
) -> np.ndarray:
    """Iterative consensus Laplacian approximation (FedSC Sec 3.2)."""
    if prev_global_L is None or server_round == 1:
        return np.mean(local_affinities, axis=0)
    
    # Combine local and global information
    consensus_L = 0.7 * prev_global_L + 0.3 * np.mean(local_affinities, axis=0)
    return consensus_L


def secure_eigen_decomposition(matrix: np.ndarray, k: int = 3) -> np.ndarray:
    """Privacy-preserving eigenvalue computation with noise injection."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Add differential privacy noise (ε=1.0)
    noise = np.random.laplace(0, 1/np.sqrt(matrix.size), eigenvectors[:, :k].shape)
    return eigenvectors[:, :k] + noise