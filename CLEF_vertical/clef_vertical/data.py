import os

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset


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
    train_statistics_merged = train_statistics_merged.drop(columns=drop_columns)
    test_statistics_merged = test_statistics_merged.drop(columns=drop_columns)

    for col in categorical_cols:
        train_statistics_merged[col] = train_statistics_merged[col].astype("category")
        test_statistics_merged[col] = test_statistics_merged[col].astype("category")

    return train_statistics_merged, test_statistics_merged


def one_hot_encode_w_missing(df_train, df_test, columns):
    result_df_train = df_train.copy()
    result_df_test = df_test.copy()

    for col in columns:
        # Get unique non-null values
        unique_values = result_df_train[col].dropna().unique()

        # Create one column per category
        for val in unique_values:
            result_df_train[f"{col}_{val}"] = (result_df_train[col] == val).astype(float)
            result_df_test[f"{col}_{val}"] = (result_df_test[col] == val).astype(float)
        # Drop the original column
        result_df_train.drop(columns=[col], inplace=True)
        result_df_test.drop(columns=[col], inplace=True)
    return result_df_train, result_df_test


def prep_data_for_nn(train_df, test_df):
    """
    Prepares the DataFrame for neural network training by dropping unnecessary columns and converting categorical columns to category type.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

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

    # Drop columns with more than 50% NaN values
    non_nan_percentage = train_df.notna().mean()
    columns_to_drop = non_nan_percentage[non_nan_percentage < 0.5].index
    train_df = train_df.drop(columns=columns_to_drop)
    test_df = test_df.drop(columns=columns_to_drop)
    # print(f"Dropped columns with more than 50% NaN values: {columns_to_drop.tolist()}")

    # Convert categorical columns to category type
    all_cols_to_encode = complete_categorical_cols + incomplete_categorical_cols
    train_df, test_df = one_hot_encode_w_missing(train_df, test_df, all_cols_to_encode)

    # normalize continuous columns
    all_continuous_cols = complete_continuous_cols + to_fill_continuous_cols
    scaler = StandardScaler()

    # Fit on training data and transform both datasets
    train_df[all_continuous_cols] = scaler.fit_transform(train_df[all_continuous_cols])
    test_df[all_continuous_cols] = scaler.transform(test_df[all_continuous_cols])

    return train_df, test_df


def load_data():
    data_path = os.getenv("CLEF_DATA_PATH")
    if data_path is None:
        raise ValueError("CLEF_DATA_PATH environment variable must be set")

    # Load all required CSV files
    file_names = [
        "train/datasetC_train-static-vars.csv",
        "test/datasetC_test-static-vars.csv",
        "train/datasetC_train-visits.csv",
        "test/datasetC_test-visits.csv",
        "train/datasetC_train-outcome.csv",
        "test/datasetC_test-outcome.csv",
    ]

    train_statistics, test_statistics, train_visits, test_visits, train_y, test_y = [
        pd.read_csv(os.path.join(data_path, fname)) for fname in file_names
    ]

    train_statistics_merged, test_statistics_merged = preprocess_clef_data(
        train_statistics, test_statistics, train_visits, test_visits
    )
    train_label = train_y.Time
    test_label = test_y.Time
    return train_statistics_merged, test_statistics_merged, train_label, test_label


def get_data():
    # Load and preprocess data
    train_statistics_merged, test_statistics_merged, train_label, test_label = load_data()
    train_df, test_df = prep_data_for_nn(train_statistics_merged, test_statistics_merged)

    personal_vars = [
        "age",
        "sex",
        "ethnicity",
        "height",
        "weight",
        "ALS_familiar_history",
        "prevalentLMN",
        "prevalentUMN",
        "mixedMN",
        "occupation",
        "hypertension",
        "diabetes",
        "dyslipidemia",
        "thyroid_disorder",
        "autoimmune_disease",
        "stroke",
        "cardiac_disease",
        "primary_neoplasm",
    ]

    def split_dataframe(df, personal_vars):
        """Split dataframe into personal and clinical data based on column names."""
        personal_cols = []
        clinical_cols = []

        for col in df.columns:
            if any(var in col for var in personal_vars):
                personal_cols.append(col)
            else:
                clinical_cols.append(col)

        return df[personal_cols], df[clinical_cols]

    # Split train and test data
    train_personal, train_clinical = split_dataframe(train_df, personal_vars)
    test_personal, test_clinical = split_dataframe(test_df, personal_vars)

    # Normalize the target values
    y_scaler = StandardScaler()
    y_train_normalized = y_scaler.fit_transform(train_label.values.reshape(-1, 1))
    y_test_normalized = y_scaler.transform(test_label.values.reshape(-1, 1))

    # Convert to PyTorch tensors
    X_train_personal = torch.tensor(train_personal.values, dtype=torch.float32)
    X_train_clinical = torch.tensor(train_clinical.values, dtype=torch.float32)
    X_test_personal = torch.tensor(test_personal.values, dtype=torch.float32)
    X_test_clinical = torch.tensor(test_clinical.values, dtype=torch.float32)
    y_train = torch.tensor(y_train_normalized, dtype=torch.float32)
    y_test = torch.tensor(y_test_normalized, dtype=torch.float32)

    # Store original values for evaluation
    y_train_original = torch.tensor(train_label.values, dtype=torch.float32).reshape(-1, 1)
    y_test_original = torch.tensor(test_label.values, dtype=torch.float32).reshape(-1, 1)

    return (
        X_train_personal,
        X_train_clinical,
        y_train,
        X_test_personal,
        X_test_clinical,
        y_test,
        y_train_original,
        y_test_original,
    )


def get_personal_data():
    # Load only personal data and return a TensorDataset
    X_train_personal, _, _, X_test_personal, _, _, _, _ = get_data()

    train_dataset = TensorDataset(X_train_personal)
    test_dataset = TensorDataset(X_test_personal)
    return X_train_personal, X_test_personal


def get_clinical_data():
    # Load only clinical data and return a TensorDataset
    _, X_train_clinical, _, _, X_test_clinical, _, _, _ = get_data()
    train_dataset = TensorDataset(X_train_clinical)
    test_dataset = TensorDataset(X_test_clinical)
    return X_train_clinical, X_test_clinical


def get_labels():
    # Load only the labels and return a TensorDataset, which also includes the indices
    _, _, y_train, _, _, y_test, y_train_original, y_test_original = get_data()

    # Create indices for training and test sets
    train_indices = torch.arange(len(y_train), dtype=torch.long)
    test_indices = torch.arange(len(y_test), dtype=torch.long)

    # Create TensorDatasets with indices and labels
    train_dataset = TensorDataset(train_indices.unsqueeze(1), y_train)
    test_dataset = TensorDataset(test_indices.unsqueeze(1), y_test)

    return train_dataset, test_dataset, y_train_original, y_test_original
