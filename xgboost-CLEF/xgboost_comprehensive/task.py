"""xgboost-CLEF: A Flower / XGBoost app applied to the CLEF dataset."""

from logging import INFO
import os

import xgboost as xgb
import pandas as pd

from flwr.common import log



def transform_dataset_to_dmatrix(data, enable_categorical=False):
    """Transform dataset to DMatrix format for xgboost."""
    x = data["inputs"]
    y = data["label"]
    new_data = xgb.DMatrix(x, label=y, enable_categorical=enable_categorical)
    return new_data


def preprocess_clef_data(train_statistics, test_statistics, train_visits, test_visits):
    categorical_cols = [
        "alive",
        "sex",
        "ethnicity",
        "moreThan10PercentWeightloss",
        "ALS_familiar_history",
        "prevalentLMN",
        "prevalentUMN",
        "mixedMN",
        "onset_bulbar",
        "onset_axial",
        "onset_generalized",
        "onset_limbs",
        "onset_limb_type",
        "smoking",
        "retired_at_diagnosis",
        "C9orf72",
        "SOD1 mutation",
        "TARDBP mutation",
        "FUS mutation",
        "hypertension",
        "diabetes",
        "dyslipidemia",
        "thyroid_disorder",
        "autoimmune_disease",
        "stroke",
        "cardiac_disease",
        "primary_neoplasm",
        "major_trauma_before_onset",
        "surgical_interventions_before_onset",
        "head_trauma_last_5_years",
        "head_trauma_more_than_5_years",
        "neck_trauma_last_5_years",
        "neck_trauma_more_than_5_years",
        "cervical_trauma_last_5_years",
        "cervical_trauma_more_than_5_years",
        "thoracic_trauma_last_5_years",
        "thoracic_trauma_more_than_5_years",
        "lumbo_sacral_trauma_last_5_years",
        "lumbo_sacral_trauma_more_than_5_years",
        "cervical_spine_surgery_last_5_years",
        "cervical_spine_surgery_more_than_5_years",
        "thoracic_spine_surgery_last_5_years",
        "thoracic_spine_surgery_more_than_5_years",
        "lumbo_sacral_spine_surgery_last_5_years",
        "lumbo_sacral_spine_surgery_more_than_5_years",
        "upper_limb_surgery_last_5_years",
        "upper_limb_surgery_more_than_5_years",
        "lower_limb_surgery_last_5_years",
        "lower_limb_surgery_more_than_5_years",
        "abdominal_surgery_last_5_years",
        "abdominal_surgery_more_than_5_years",
        "thoracic_surgery_last_5_years",
        "thoracic_surgery_more_than_5_years",
        "pelvic_surgery_last_5_years",
        "pelvic_surgery_more_than_5_years",
        "head_neck_surgery_last_5_years",
        "head_neck_surgery_more_than_5_years",
    ]
    drop_columns = [
        "occupation",
        "CK_unit",
        "Albumin_unit",
        "Creatinine_unit",
        "Total_Cholesterol_unit",
        "HDL_Cholesterol_unit",
        "LDL_Cholesterol_unit",
        "Triglycerides_unit",
        "PatientID",
    ]

    # keep only rows where date_spiro is 0 or smaller and keep the one with the highest value
    train_spiro_visits = train_visits[train_visits["date_spiro"] <= 0].sort_values("date_spiro").drop_duplicates(subset="PatientID", keep="last")
    test_spiro_visits = test_visits[test_visits["date_spiro"] <= 0].sort_values("date_spiro").drop_duplicates(subset="PatientID", keep="last")

    # keep only rows where date_alsfrs_r = 0.0  
    train_alsfrs_visits = train_visits[train_visits["date_alsfrs_r"] == 0.0]
    test_alsfrs_visits = test_visits[test_visits["date_alsfrs_r"] == 0.0]

    # Update Albumin_level, Albumin_lower_range, Albumin_upper_range where is Albumin_unit mg/dL
    train_statistics.loc[train_statistics['Albumin_unit'] == 'mg/dL', ['Albumin_level', 'Albumin_lower_range', 'Albumin_upper_range']] /= 1000
    train_statistics.loc[train_statistics['Albumin_unit'] == 'mg/dL', 'Albumin_unit'] = 'g/L'

    # add the fvcValue column to the train_statistics dataframe, matching the PatientID:
    train_statistics_merged = train_statistics.merge(train_spiro_visits[["PatientID", "fvcValue"]], on="PatientID", how="left")
    # add all columns but PatientID, date_spiro and fvcValue to the train_statistics dataframe from the train_visits dataframe:
    alsfrs_r_to_merge_columns= [col for col in train_alsfrs_visits.columns if col not in ["date_spiro", "date_alsfrs_r", "fvcValue"]]
    train_statistics_merged = train_statistics_merged.merge(train_alsfrs_visits[alsfrs_r_to_merge_columns], on="PatientID", how="left")

    test_statistics_merged = test_statistics.merge(test_spiro_visits[["PatientID", "fvcValue"]], on="PatientID", how="left")
    test_statistics_merged = test_statistics_merged.merge(test_alsfrs_visits[alsfrs_r_to_merge_columns], on="PatientID", how="left")

    assert len(train_statistics_merged) == len(train_statistics), "merging made your df train explode"
    assert len(test_statistics_merged) == len(test_statistics), "merging made your df test explode"

    # drop columns we shouldn't use for training:  
    train_statistics_merged = train_statistics_merged.drop(columns=drop_columns)
    test_statistics_merged = test_statistics_merged.drop(columns=drop_columns)

    for col in categorical_cols:
        train_statistics_merged[col] = train_statistics_merged[col].astype("category")
        test_statistics_merged[col] = test_statistics_merged[col].astype("category")

    return train_statistics_merged, test_statistics_merged


def load_data(partition_id, num_partitions):
    data_path = os.getenv("CLEF_DATA_PATH")
    if data_path is None:
        raise ValueError("CLEF_DATA_PATH environment variable must be set")

    # Determine the directory to load data from
    if os.getenv("CLEF_LOCAL_DATA") == "true":
        partition_id = os.getenv("CLEF_PARTITION_ID")
        load_dir = data_path
    else:
        load_dir = os.path.join(data_path, f"partitions_{num_partitions}", f"partition_{partition_id}")
    
    log(INFO, f"Loading file: {load_dir}")
    # Load all required CSV files
    file_names = [
        "train_statistics.csv",
        "test_statistics.csv", 
        "train_visits.csv",
        "test_visits.csv",
        "train_labels.csv",
        "test_labels.csv"
    ]
    
    train_statistics, test_statistics, train_visits, test_visits, train_y, test_y = [
        pd.read_csv(os.path.join(load_dir, fname)) for fname in file_names
    ]

    train_statistics_merged, test_statistics_merged = preprocess_clef_data(train_statistics, test_statistics, train_visits, test_visits)
    train_label = train_y.Time
    test_label = test_y.Time

    train_dmatrix = transform_dataset_to_dmatrix({"inputs": train_statistics_merged, "label": train_label}, enable_categorical=True)
    test_dmatrix = transform_dataset_to_dmatrix({"inputs": test_statistics_merged, "label": test_label}, enable_categorical=True)
    
    num_train = len(train_label)
    num_test = len(test_label)
    log(INFO, f"partition {partition_id}: num_train: {num_train}, num_test: {num_test}")

    return train_dmatrix, test_dmatrix, num_train, num_test
        

def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
