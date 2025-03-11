import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def prepare_mimic_dataset(input_csv, output_dir, mimic_base_dir):
    """
    Prepare MIMIC-CXR dataset for the "No Finding" task.
    
    Args:
        input_csv: Path to the mimic-cxr-subpopbench.csv file
        output_dir: Directory to save the metadata file
        mimic_base_dir: Base directory of MIMIC-CXR-JPG dataset
    """
    print(f"Loading data from {input_csv}")
    mimic = pd.read_csv(input_csv)
    
    # Check for NaN values in the 'No Finding' column
    nan_count = mimic['No Finding'].isna().sum()
    print(f"Found {nan_count} NaN values in 'No Finding' column")
    
    # Create a binary label for "No Finding" - safely handle NaN values
    # Assumption: NaN means "not No Finding" (so we fill with 0)
    mimic['label'] = mimic['No Finding'].fillna(0).astype(int)
    
    # Add path column based on the filename
    # Adjust this based on the actual file structure
    mimic['path'] = mimic['filename'].apply(
        lambda x: os.path.join('files', x.replace('dcm', 'jpg'))
    )
    
    # Verify images exist
    valid_paths = []
    for idx, row in mimic.iterrows():
        full_path = os.path.join(mimic_base_dir, row['path'])
        if os.path.exists(full_path):
            valid_paths.append(idx)
        if idx % 1000 == 0:
            print(f"Checked {idx} paths...")
    
    mimic = mimic.loc[valid_paths]
    print(f"Kept {len(mimic)} valid image paths out of {len(mimic)}")
    
    # Select only necessary columns for our dataset
    metadata = mimic[['study_id', 'path', 'label', 'a']]
    
    # Add a stratification column for balancing
    metadata['stratify_col'] = metadata['label'].astype(str) + '_' + metadata['a'].astype(str)
    
    # Save the metadata
    output_path = os.path.join(output_dir, 'metadata_no_finding.csv')
    metadata.to_csv(output_path, index=False)
    print(f"Saved metadata to {output_path}")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total studies: {metadata['study_id'].nunique()}")
    print(f"Total images: {len(metadata)}")
    print(f"Label distribution: \n{metadata['label'].value_counts(normalize=True)}")
    print(f"Average images per study: {len(metadata) / metadata['study_id'].nunique():.2f}")
    
    # Create a smaller sample for testing if needed
    sample = metadata.groupby('study_id').first().reset_index()
    sample = sample.sample(n=min(1000, len(sample)))
    sample_output_path = os.path.join(output_dir, 'metadata_no_finding_sample.csv')
    sample.to_csv(sample_output_path, index=False)
    print(f"Saved sample metadata to {sample_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare MIMIC-CXR data for No Finding task')
    parser.add_argument('--input_csv', type=str, required=True, 
                        help='Path to mimic-cxr-subpopbench.csv')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save metadata files')
    parser.add_argument('--mimic_base_dir', type=str, required=True,
                        help='Base directory of MIMIC-CXR-JPG dataset')
    
    args = parser.parse_args()
    prepare_mimic_dataset(args.input_csv, args.output_dir, args.mimic_base_dir) 