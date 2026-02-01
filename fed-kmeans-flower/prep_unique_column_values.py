import json
import os

import pandas as pd

from fed_kmeans_flower.data.clef_loading import load_data


def main():
    # Base data path from environment variable
    base_data_path = os.getenv("CLEF_DATA_PATH")
    if not base_data_path:
        raise ValueError("CLEF_DATA_PATH environment variable not set. Please set it to the root path containing L, T, and U folders.")
    
    # Partition mapping from loader.py
    partition_mapping = {
        0: "T",
        1: "L",
        2: "U",
    }
    
    # Categorical columns as defined in prep_data in clef_loading.py
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
    
    all_categorical_cols = complete_categorical_cols + incomplete_categorical_cols
    
    # To store unique values for each column
    unique_values_map = {col: set() for col in all_categorical_cols}
    
    for partition_id, sub_dir in partition_mapping.items():
        print(f"Processing partition {partition_id} ({sub_dir})...")
        data_path = os.path.join(base_data_path, sub_dir, "datasetC")
        
        try:
            # load_data returns train_df, test_df, train_label, test_label
            train_df, test_df, _, _ = load_data(data_path)
            
            # Combine train and test to get all possible values in this partition
            combined_df = pd.concat([train_df, test_df], axis=0)
            
            # Apply cleaning logic from prep_data in clef_loading.py
            # SOD1 mutation
            if "SOD1 mutation" in combined_df.columns:
                combined_df["SOD1 mutation"] = combined_df["SOD1 mutation"].apply(
                    lambda x: True if x == "c.281G>T (p.(Gly94Val))" else (False if isinstance(x, str) else x)
                )
            
            # C9orf72
            if "C9orf72" in combined_df.columns:
                combined_df["C9orf72"] = combined_df["C9orf72"].apply(lambda x: True if x == "expansion" else False)
                
            # TARDBP mutation
            if "TARDBP mutation" in combined_df.columns:
                combined_df["TARDBP mutation"] = combined_df["TARDBP mutation"].apply(
                    lambda x: True if x == "c.1144G>A, p.(Ala382Thr)" else (False if isinstance(x, str) else x)
                )
            
            # Collect unique values
            for col in all_categorical_cols:
                if col in combined_df.columns:
                    # Dropna because one_hot_encode_w_missing does it
                    uniques = combined_df[col].dropna().unique()
                    for val in uniques:
                        # Convert to string for JSON serialization compatibility and consistency
                        # Note: one_hot_encode_w_missing uses the value directly in the column name
                        unique_values_map[col].add(val)
                else:
                    print(f"  Warning: Column {col} not found in partition {partition_id}")
                    
        except Exception as e:
            print(f"  Error processing partition {partition_id}: {e}")
            
    # Convert sets to sorted lists for JSON
    final_unique_values = {}
    for col, values in unique_values_map.items():
        # Filter out NaN just in case, though dropna() should have handled it
        clean_values = [v for v in values if pd.notna(v)]
        # Sort for consistency, handle mixed types by converting to str for sorting
        final_unique_values[col] = sorted(list(clean_values), key=lambda x: str(x))
        
    # Save to JSON in each partition folder
    output_filename = "unique_categorical_values.json"
    
    # Also save to current directory for reference
    with open(output_filename, "w") as f:
        json.dump(final_unique_values, f, indent=4)
    print(f"\nSaved summary to {output_filename}")

    for sub_dir in partition_mapping.values():
        target_path = os.path.join(base_data_path, sub_dir, output_filename)
        try:
            with open(target_path, "w") as f:
                json.dump(final_unique_values, f, indent=4)
            print(f"Saved copy to {target_path}")
        except Exception as e:
            print(f"Error saving to {target_path}: {e}")

    print(f"\nTotal categorical columns: {len(final_unique_values)}")

if __name__ == "__main__":
    main()
