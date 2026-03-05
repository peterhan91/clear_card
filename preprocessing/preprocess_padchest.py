"""
Preprocessing script for PadChest dataset with patient-level train/validation/test splits.

Split strategy:
    - Test: All Physician (human-annotated) labeled images
    - Train/Val: RNN_model labeled images from patients NOT in the test set
      (patients appearing in both groups are excluded from train/val to prevent leakage)
    - Train/Val split at patient level (default 90/10)

Output: Single CSV with a `split` column and multi-hot 0/1 label columns.
"""

import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path


def parse_labels(labels_str):
    """Parse PadChest Labels string like "['pleural effusion', 'kyphosis']" into a list of labels."""
    try:
        labels_arr = labels_str.strip('][').split(', ')
        parsed = []
        for label in labels_arr:
            processed = label.split("'")[1].strip().lower()
            if processed:
                parsed.append(processed)
        return parsed
    except Exception:
        return []


def get_unique_labels(df):
    """Get sorted list of all unique labels in the Labels column."""
    unique = set()
    for labels_str in df['Labels'].dropna():
        for label in parse_labels(labels_str):
            unique.add(label)
    return sorted(unique)


def preprocess_padchest(
    csv_path: str,
    output_path: str,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Loading PadChest CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Total images: {len(df)}, Unique patients: {df['PatientID'].nunique()}")

    # Separate by labeling method
    physician_df = df[df['MethodLabel'] == 'Physician'].copy()
    rnn_df = df[df['MethodLabel'] == 'RNN_model'].copy()
    print(f"  Physician-labeled: {len(physician_df)} images, {physician_df['PatientID'].nunique()} patients")
    print(f"  RNN_model-labeled: {len(rnn_df)} images, {rnn_df['PatientID'].nunique()} patients")

    # Test patients = any patient with at least one Physician-labeled image
    test_patient_ids = set(physician_df['PatientID'].unique())

    # Remove overlapping patients from RNN set
    rnn_clean = rnn_df[~rnn_df['PatientID'].isin(test_patient_ids)].copy()
    n_dropped = len(rnn_df) - len(rnn_clean)
    print(f"\n  Dropped {n_dropped} RNN images from patients also in test set")
    print(f"  RNN images remaining for train/val: {len(rnn_clean)}, patients: {rnn_clean['PatientID'].nunique()}")

    # Split remaining RNN patients into train/val at patient level
    trainval_patients = rnn_clean['PatientID'].unique()
    np.random.seed(seed)
    shuffled = np.random.permutation(trainval_patients)

    n_val = int(len(shuffled) * val_ratio)
    val_patients = set(shuffled[:n_val])
    train_patients = set(shuffled[n_val:])

    print(f"\n  Train patients: {len(train_patients)}")
    print(f"  Val patients: {len(val_patients)}")
    print(f"  Test patients: {len(test_patient_ids)}")

    # Verify no overlap
    assert len(train_patients & val_patients) == 0
    assert len(train_patients & test_patient_ids) == 0
    assert len(val_patients & test_patient_ids) == 0
    print("  VERIFIED: No patient overlap between splits!")

    # Combine into single dataframe with split column
    physician_df['split'] = 'test'
    rnn_clean.loc[rnn_clean['PatientID'].isin(train_patients), 'split'] = 'train'
    rnn_clean.loc[rnn_clean['PatientID'].isin(val_patients), 'split'] = 'valid'
    combined = pd.concat([rnn_clean, physician_df], ignore_index=True)

    # Get all unique labels and create multi-hot columns
    print("\nParsing labels into multi-hot encoding...")
    unique_labels = get_unique_labels(combined)
    # Remove empty string if present
    unique_labels = [l for l in unique_labels if l]
    print(f"  Found {len(unique_labels)} unique labels")

    # Initialize all label columns to 0
    for label in unique_labels:
        combined[label] = 0

    # Fill in multi-hot values
    for idx, row in combined.iterrows():
        parsed = parse_labels(row['Labels']) if pd.notna(row['Labels']) else []
        for label in parsed:
            if label in unique_labels:
                combined.at[idx, label] = 1

    # Select output columns: metadata + split + label columns
    meta_cols = ['ImageID', 'PatientID', 'StudyID', 'split', 'PatientSex_DICOM',
                 'Projection', 'MethodLabel', 'Report']
    meta_cols = [c for c in meta_cols if c in combined.columns]
    out_df = combined[meta_cols + unique_labels].copy()

    out_df.to_csv(output_path, index=False)
    print(f"\nWrote {len(out_df)} rows to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("PadChest Preprocessing Summary")
    print("=" * 60)
    for split in ['train', 'valid', 'test']:
        sub = out_df[out_df['split'] == split]
        print(f"  {split:>10}: {len(sub):>7} images, {sub['PatientID'].nunique():>5} patients")
    print(f"  Label columns: {len(unique_labels)}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess PadChest with patient-level splits (test = human-annotated)'
    )
    parser.add_argument(
        '--csv_path', type=str, required=True,
        help='Path to PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv'
    )
    parser.add_argument(
        '--output_path', type=str, default='data/padchest_labels.csv',
        help='Output CSV path'
    )
    parser.add_argument(
        '--val_ratio', type=float, default=0.1,
        help='Fraction of train patients for validation (default: 0.1)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for train/val split'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    preprocess_padchest(
        csv_path=args.csv_path,
        output_path=args.output_path,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
