import json
import os

import numpy as np
import pandas as pd


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
        "centre_x",
        "centre_y",
    ]

    # keep only rows where date_spiro is 0 or smaller and keep the one with the highest value
    train_spiro_visits = (
        train_visits[train_visits["date_spiro"] <= 0]
        .sort_values("date_spiro")
        .drop_duplicates(subset="PatientID", keep="last")
    )
    test_spiro_visits = (
        test_visits[test_visits["date_spiro"] <= 0]
        .sort_values("date_spiro")
        .drop_duplicates(subset="PatientID", keep="last")
    )

    # keep only rows where date_alsfrs_r = 0.0
    train_alsfrs_visits = train_visits[train_visits["date_alsfrs_r"] == 0.0]
    test_alsfrs_visits = test_visits[test_visits["date_alsfrs_r"] == 0.0]

    # Update Albumin_level, Albumin_lower_range, Albumin_upper_range where is Albumin_unit mg/dL
    # set albumumin_unit column to type object
    train_statistics["Albumin_unit"] = train_statistics["Albumin_unit"].astype(object)
    
    train_statistics.loc[
        train_statistics["Albumin_unit"] == "mg/dL", ["Albumin_level", "Albumin_lower_range", "Albumin_upper_range"]
    ] /= 1000
    train_statistics.loc[train_statistics["Albumin_unit"] == "mg/dL", "Albumin_unit"] = "g/L"

    # add the fvcValue column to the train_statistics dataframe, matching the PatientID:
    train_statistics_merged = train_statistics.merge(
        train_spiro_visits[["PatientID", "fvcValue"]], on="PatientID", how="left"
    )
    # add all columns but PatientID, date_spiro and fvcValue to the train_statistics dataframe from the train_visits dataframe:
    alsfrs_r_to_merge_columns = [
        col for col in train_alsfrs_visits.columns if col not in ["date_spiro", "date_alsfrs_r", "fvcValue"]
    ]
    train_statistics_merged = train_statistics_merged.merge(
        train_alsfrs_visits[alsfrs_r_to_merge_columns], on="PatientID", how="left"
    )

    test_statistics_merged = test_statistics.merge(
        test_spiro_visits[["PatientID", "fvcValue"]], on="PatientID", how="left"
    )
    test_statistics_merged = test_statistics_merged.merge(
        test_alsfrs_visits[alsfrs_r_to_merge_columns], on="PatientID", how="left"
    )

    assert len(train_statistics_merged) == len(train_statistics), "merging made your df train explode"
    assert len(test_statistics_merged) == len(test_statistics), "merging made your df test explode"

    # drop columns we shouldn't use for training:
    train_statistics_merged = train_statistics_merged.drop(columns=drop_columns, errors="ignore")
    test_statistics_merged = test_statistics_merged.drop(columns=drop_columns, errors="ignore")

    for col in categorical_cols:
        train_statistics_merged[col] = train_statistics_merged[col].astype("category")
        test_statistics_merged[col] = test_statistics_merged[col].astype("category")

    return train_statistics_merged, test_statistics_merged


def one_hot_encode_w_missing(df_train, df_test, columns, categories_dict=None):
    result_df_train = df_train.copy()
    result_df_test = df_test.copy()

    for col in columns:
        if categories_dict and col in categories_dict:
            # Use pre-calculated unique values to ensure consistent columns across partitions
            unique_values = categories_dict[col]
        else:
            # Fallback to values present in the current training set
            unique_values = result_df_train[col].dropna().unique()

        # Create one column per category
        for val in unique_values:
            result_df_train[f"{col}_{val}"] = (result_df_train[col] == val).astype(float)
            result_df_test[f"{col}_{val}"] = (result_df_test[col] == val).astype(float)
        # Drop the original column
        result_df_train.drop(columns=[col], inplace=True)
        result_df_test.drop(columns=[col], inplace=True)
    return result_df_train, result_df_test


def prep_data(train_df, test_df, categories_dict=None):
    """
    Prepares the DataFrame by dropping unnecessary columns and converting categorical columns to category type.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        categories_dict (dict): Optional dictionary of unique values for categorical columns.

    Returns:
        pd.DataFrame: The prepared DataFrame.
    """
    complete_categorical_cols = [
        "alive",
        "sex",
        "ethnicity",
        "onset_bulbar",
        "onset_axial",
        "onset_generalized",
        "onset_limbs",
    ]

    incomplete_categorical_cols = [
        "moreThan10PercentWeightloss",
        "ALS_familiar_history",
        "prevalentLMN",
        "prevalentUMN",
        "mixedMN",
        "smoking",
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
        "onset_limb_type",
    ]

    to_fill_continuous_cols = ["height", "weight", "weight_before_onset"]

    complete_continuous_cols = [
        "onsetDate",
        "diagnosisDate",
        "height",
        "weight_before_onset",
        "weight",
        "age_onset",
        "slope",
        "alsfrs_r_tot_score",
        "bulbar_subscore",
        "motor_subscore",
        "respiratory_subscore",
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q7",
        "q8",
        "q9",
        "q10",
        "q11",
        "q12",
    ]

    # Drop unnecessary columns
    train_df = train_df.drop(columns=["PatientID"])
    test_df = test_df.drop(columns=["PatientID"])

    # Update the 'SOD1 mutation' column, only c.281G>T has been labled as pathogenic online
    train_df["SOD1 mutation"] = train_df["SOD1 mutation"].apply(
        lambda x: True if x == "c.281G>T (p.(Gly94Val))" else (False if isinstance(x, str) else x)
    )

    # Update the 'C9orf72' column to true/false
    train_df["C9orf72"] = train_df["C9orf72"].apply(lambda x: True if x == "expansion" else False)
    test_df["C9orf72"] = test_df["C9orf72"].apply(lambda x: True if x == "expansion" else False)

    # Map the 'TARDBP mutation' column to True/False
    train_df["TARDBP mutation"] = train_df["TARDBP mutation"].apply(
        lambda x: True if x == "c.1144G>A, p.(Ala382Thr)" else (False if isinstance(x, str) else x)
    )

    test_df["TARDBP mutation"] = test_df["TARDBP mutation"].apply(
        lambda x: True if x == "c.1144G>A, p.(Ala382Thr)" else (False if isinstance(x, str) else x)
    )
    # Fill missing values in the specified columns with their mean value
    for col in to_fill_continuous_cols:
        train_df[col] = train_df[col].fillna(train_df[col].mean())
        test_df[col] = test_df[col].fillna(test_df[col].mean())

    # Hardcoded list of columns to drop because more than 50% is missing. 
    # Ideally these would be determined through federated analytics
    columns_to_drop = [
        'retired_at_diagnosis', 'smoking_startYear', 'smoking_endYear',
        'dailyCigarettes', 'packYear', 'CK_level', 'CK_lower_range',
        'CK_upper_range', 'Albumin_level', 'Albumin_lower_range',
        'Albumin_upper_range', 'Creatinine_level', 'Creatinine_lower_range',
        'Creatinine_upper_range', 'Total_Cholesterol_level',
        'Total_Cholesterol_lower_range', 'Total_Cholesterol_upper_range',
        'HDL_Cholesterol_level', 'HDL_Cholesterol_lower_range',
        'HDL_Cholesterol_upper_range', 'LDL_Cholesterol_level',
        'LDL_Cholesterol_lower_range', 'LDL_Cholesterol_upper_range',
        'Triglycerides_level', 'Triglycerides_lower_range',
        'Triglycerides_upper_range', 'major_trauma_before_onset',
        'surgical_interventions_before_onset', 'head_trauma_last_5_years',
        'head_trauma_more_than_5_years', 'neck_trauma_last_5_years',
        'neck_trauma_more_than_5_years', 'cervical_trauma_last_5_years',
        'cervical_trauma_more_than_5_years', 'thoracic_trauma_last_5_years',
        'thoracic_trauma_more_than_5_years', 'lumbo_sacral_trauma_last_5_years',
        'lumbo_sacral_trauma_more_than_5_years',
        'cervical_spine_surgery_last_5_years',
        'cervical_spine_surgery_more_than_5_years',
        'thoracic_spine_surgery_last_5_years',
        'thoracic_spine_surgery_more_than_5_years',
        'lumbo_sacral_spine_surgery_last_5_years',
        'lumbo_sacral_spine_surgery_more_than_5_years',
        'upper_limb_surgery_last_5_years',
        'upper_limb_surgery_more_than_5_years',
        'lower_limb_surgery_last_5_years',
        'lower_limb_surgery_more_than_5_years',
        'abdominal_surgery_last_5_years', 'abdominal_surgery_more_than_5_years',
        'thoracic_surgery_last_5_years', 'thoracic_surgery_more_than_5_years',
        'pelvic_surgery_last_5_years', 'pelvic_surgery_more_than_5_years',
        'head_neck_surgery_last_5_years', 'head_neck_surgery_more_than_5_years',
        'fvcValue'
    ]

    train_df = train_df.drop(columns=columns_to_drop, errors="ignore")
    test_df = test_df.drop(columns=columns_to_drop, errors="ignore")

    # Fill missing values for ALL continuous columns that remain
    # This prevents the "Data contains NaN" error during validation
    all_continuous_cols = list(set(complete_continuous_cols + to_fill_continuous_cols))
    for col in all_continuous_cols:
        if col in train_df.columns:
            fill_value = train_df[col].mean()
            # If the entire column is NaN, use 0 as a fallback
            if pd.isna(fill_value):
                fill_value = 0
            train_df[col] = train_df[col].fillna(fill_value)
            
        if col in test_df.columns:
            # Use training mean to fill test NaNs
            test_df[col] = test_df[col].fillna(fill_value)
    # print(f"Dropped columns with more than 50% NaN values: {columns_to_drop.tolist()}")

    # Convert categorical columns to category type
    all_cols_to_encode = complete_categorical_cols + incomplete_categorical_cols
    train_df, test_df = one_hot_encode_w_missing(train_df, test_df, all_cols_to_encode, categories_dict=categories_dict)

    # Identify continuous column indices after one-hot encoding
    all_continuous_cols = list(set(complete_continuous_cols + to_fill_continuous_cols))
    continuous_indices = []
    for i, col_name in enumerate(train_df.columns):
        if col_name in all_continuous_cols:
            continuous_indices.append(i)

    # Normally you would normalize the data, however, in the case of federated learning you should perform federated normalization. 

    # all_continuous_cols = complete_continuous_cols + to_fill_continuous_cols
    # scaler = StandardScaler()

    # # Fit on training data and transform both datasets
    # train_df[all_continuous_cols] = scaler.fit_transform(train_df[all_continuous_cols])
    # test_df[all_continuous_cols] = scaler.transform(test_df[all_continuous_cols])

    return train_df, test_df, continuous_indices


def load_data(data_path):
    if not data_path:
        raise ValueError("data_path must be provided to load_data")

    # Load all required CSV files
    file_names = [
        "train/datasetC_train-static-vars.csv",
        "test/datasetC_test-static-vars.csv",
        "train/datasetC_train-visits.csv",
        "test/datasetC_test-visits.csv",
        "train/datasetC_train-outcome.csv",
        "test/datasetC_test-outcome.csv",
    ]

    results = []
    for fname in file_names:
        full_path = os.path.join(data_path, fname)
        if not os.path.exists(full_path):
             raise FileNotFoundError(f"Data file not found: {full_path}")
        results.append(pd.read_csv(full_path))

    train_statistics, test_statistics, train_visits, test_visits, train_y, test_y = results

    train_statistics_merged, test_statistics_merged = preprocess_clef_data(
        train_statistics, test_statistics, train_visits, test_visits
    )
    train_label = train_y.Time
    test_label = test_y.Time
    return train_statistics_merged, test_statistics_merged, train_label, test_label


def get_data(data_path):
    # Load and preprocess data
    train_statistics_merged, test_statistics_merged, train_label, test_label = load_data(data_path)
    
    # Try to load pre-calculated categorical values for consistent one-hot encoding
    categories_dict = None
    if data_path:
        json_path = os.path.join(os.path.dirname(data_path), "unique_categorical_values.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    categories_dict = json.load(f)
                print(f"Loaded consistent categorical values from {json_path}")
            except Exception as e:
                print(f"Warning: Failed to load {json_path}: {e}")

    train_df, test_df, continuous_indices = prep_data(train_statistics_merged, test_statistics_merged, categories_dict=categories_dict)

    # Convert to NumPy arrays
    print(train_df.columns)
    X_train = train_df.values.astype(np.float32)
    X_test = test_df.values.astype(np.float32)
    
    # Use raw labels for evaluation
    y_train = train_label.values.astype(np.float32)
    y_test = test_label.values.astype(np.float32)

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        continuous_indices,
    )
