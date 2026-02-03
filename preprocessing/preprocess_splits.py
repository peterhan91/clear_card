"""
Preprocessing script for CheXpert-Plus and ReXGradient datasets with proper
train/validation/test splits ensuring NO patient-level overlap between splits.

For CheXpert-Plus:
    - Train: from chexpert_train.csv (excluding patients in val/test)
    - Validation: official CheXpert validation set (chexpert_valid.csv)
    - Test: official CheXpert test set (chexpert_test.csv)

For ReXGradient:
    - Creates train/val/test splits at patient level
    - Default split: 80% train, 10% val, 10% test
"""

import argparse
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Set, Tuple, List
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from data_process import img_to_hdf5, img_to_hdf5_parallel


def extract_patient_id_from_path(path: str) -> str:
    """
    Extract patient ID from CheXpert image path.

    Examples:
        'train/patient42142/study5/view1_frontal.jpg' -> 'patient42142'
        'CheXpert-v1.0/valid/patient64541/study1/view1_frontal.jpg' -> 'patient64541'
    """
    match = re.search(r'(patient\d+)', path)
    if match:
        return match.group(1)
    return None


def extract_rexgradient_patient_id(row_id: str) -> str:
    """
    Extract patient ID from ReXGradient id column.

    Example:
        'pGRDNLZHK1CJMB9DS_aGRDNLD4ATLU63FN8_s1.2...' -> 'GRDNLZHK1CJMB9DS'
    """
    # Format: pPATIENT_ID_aACCESSION_sSERIES
    if row_id.startswith('p'):
        parts = row_id.split('_')
        if len(parts) >= 1:
            return parts[0][1:]  # Remove 'p' prefix
    return row_id


def get_chexpert_patient_ids(csv_path: str, path_column: str = 'Path') -> Set[str]:
    """Get set of patient IDs from a CheXpert CSV file."""
    df = pd.read_csv(csv_path)

    # Handle different column names
    if path_column not in df.columns and 'path_to_image' in df.columns:
        path_column = 'path_to_image'

    patient_ids = set()
    for path in df[path_column]:
        pid = extract_patient_id_from_path(str(path))
        if pid:
            patient_ids.add(pid)

    return patient_ids


def preprocess_chexpert_plus(
    data_dir: str,
    output_dir: str,
    image_root: str,
    resolution: int = 448,
    num_workers: int = None
) -> dict:
    """
    Preprocess CheXpert-Plus dataset with proper train/val/test splits.

    Ensures NO patient overlap between splits by:
    1. Getting patient IDs from official validation and test sets
    2. Removing those patients from training set

    Args:
        data_dir: Directory containing CSV files
        output_dir: Directory for output H5 files
        image_root: Root directory for CheXpert images
        resolution: Target image resolution

    Returns:
        Dictionary with statistics about each split
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load CSVs
    train_csv = os.path.join(data_dir, 'chexpert_train.csv')
    valid_csv = os.path.join(data_dir, 'chexpert_valid.csv')
    test_csv = os.path.join(data_dir, 'chexpert_test.csv')

    print("Loading CheXpert CSVs...")
    df_train = pd.read_csv(train_csv)
    df_valid = pd.read_csv(valid_csv)
    df_test = pd.read_csv(test_csv)

    # Get patient IDs from validation and test sets
    print("Extracting patient IDs from validation set...")
    valid_patient_ids = get_chexpert_patient_ids(valid_csv, 'Path')
    print(f"  Found {len(valid_patient_ids)} unique patients in validation set")

    print("Extracting patient IDs from test set...")
    test_patient_ids = get_chexpert_patient_ids(test_csv, 'Path')
    print(f"  Found {len(test_patient_ids)} unique patients in test set")

    # Check for overlap between val and test (should be none)
    val_test_overlap = valid_patient_ids & test_patient_ids
    if val_test_overlap:
        print(f"  WARNING: {len(val_test_overlap)} patients overlap between val and test!")

    # Extract patient IDs from training set
    print("Extracting patient IDs from training set...")
    if 'deid_patient_id' in df_train.columns:
        df_train['patient_id'] = df_train['deid_patient_id']
    else:
        df_train['patient_id'] = df_train['path_to_image'].apply(extract_patient_id_from_path)

    train_patient_ids_before = set(df_train['patient_id'].unique())
    print(f"  Found {len(train_patient_ids_before)} unique patients in training set")

    # Remove patients that are in validation or test sets
    excluded_patients = valid_patient_ids | test_patient_ids
    df_train_clean = df_train[~df_train['patient_id'].isin(excluded_patients)]

    removed_count = len(df_train) - len(df_train_clean)
    print(f"  Removed {removed_count} images from training set (patients in val/test)")

    train_patient_ids_after = set(df_train_clean['patient_id'].unique())
    print(f"  Training set now has {len(train_patient_ids_after)} unique patients")

    # Verify no overlap
    train_val_overlap = train_patient_ids_after & valid_patient_ids
    train_test_overlap = train_patient_ids_after & test_patient_ids

    assert len(train_val_overlap) == 0, f"Patient overlap between train and val: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"Patient overlap between train and test: {train_test_overlap}"
    print("  VERIFIED: No patient overlap between splits!")

    stats = {}

    # Process training set
    print("\nProcessing training set...")
    train_paths = df_train_clean['path_to_image'].apply(
        lambda x: os.path.join(image_root, x)
    ).tolist()

    train_h5_path = os.path.join(output_dir, 'chexpert_plus_train.h5')
    img_to_hdf5_parallel(train_paths, train_h5_path, resolution=resolution, num_workers=num_workers)

    # Save training metadata with h5_index for guaranteed pairing
    train_meta = df_train_clean[['path_to_image', 'patient_id']].copy().reset_index(drop=True)
    train_meta['h5_index'] = train_meta.index  # Explicit index for H5 pairing
    if 'impression' in df_train_clean.columns:
        train_meta['impression'] = df_train_clean['impression'].values
    elif 'section_impression' in df_train_clean.columns:
        train_meta['impression'] = df_train_clean['section_impression'].values
    train_meta.to_csv(os.path.join(output_dir, 'chexpert_plus_train_metadata.csv'), index=False)

    stats['train'] = {
        'images': len(train_paths),
        'patients': len(train_patient_ids_after),
        'h5_path': train_h5_path
    }

    # Process validation set
    print("\nProcessing validation set...")
    valid_paths = df_valid['Path'].apply(
        lambda x: os.path.join(image_root, x)
    ).tolist()

    valid_h5_path = os.path.join(output_dir, 'chexpert_plus_valid.h5')
    img_to_hdf5_parallel(valid_paths, valid_h5_path, resolution=resolution, num_workers=num_workers)

    # Save validation metadata (includes labels) with h5_index
    valid_meta = df_valid.copy().reset_index(drop=True)
    valid_meta['h5_index'] = valid_meta.index
    valid_meta.to_csv(os.path.join(output_dir, 'chexpert_plus_valid_metadata.csv'), index=False)

    stats['valid'] = {
        'images': len(valid_paths),
        'patients': len(valid_patient_ids),
        'h5_path': valid_h5_path
    }

    # Process test set
    print("\nProcessing test set...")
    test_paths = df_test['Path'].apply(
        lambda x: os.path.join(image_root, x)
    ).tolist()

    test_h5_path = os.path.join(output_dir, 'chexpert_plus_test.h5')
    img_to_hdf5_parallel(test_paths, test_h5_path, resolution=resolution, num_workers=num_workers)

    # Save test metadata (includes labels) with h5_index
    test_meta = df_test.copy().reset_index(drop=True)
    test_meta['h5_index'] = test_meta.index
    test_meta.to_csv(os.path.join(output_dir, 'chexpert_plus_test_metadata.csv'), index=False)

    stats['test'] = {
        'images': len(test_paths),
        'patients': len(test_patient_ids),
        'h5_path': test_h5_path
    }

    # Print summary
    print("\n" + "="*60)
    print("CheXpert-Plus Preprocessing Summary")
    print("="*60)
    for split, s in stats.items():
        print(f"{split:>10}: {s['images']:>7} images, {s['patients']:>5} patients")
    print("="*60)

    return stats


def preprocess_rexgradient(
    data_dir: str,
    output_dir: str,
    image_root: str,
    resolution: int = 448,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = None
) -> dict:
    """
    Preprocess ReXGradient dataset with patient-level train/val/test splits.

    Args:
        data_dir: Directory containing rexgradient_all.csv
        output_dir: Directory for output H5 files
        image_root: Root directory for images (prepended to paths in CSV)
        resolution: Target image resolution
        train_ratio: Fraction for training (default 0.8)
        val_ratio: Fraction for validation (default 0.1)
        test_ratio: Fraction for test (default 0.1)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with statistics about each split
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, 'rexgradient_all.csv')
    print(f"Loading ReXGradient data from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"  Total images: {len(df)}")

    # Extract patient IDs
    print("Extracting patient IDs...")
    df['patient_id'] = df['id'].apply(extract_rexgradient_patient_id)

    unique_patients = df['patient_id'].unique()
    print(f"  Found {len(unique_patients)} unique patients")

    # Shuffle patients and split
    np.random.seed(seed)
    shuffled_patients = np.random.permutation(unique_patients)

    n_patients = len(shuffled_patients)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)

    train_patients = set(shuffled_patients[:n_train])
    val_patients = set(shuffled_patients[n_train:n_train + n_val])
    test_patients = set(shuffled_patients[n_train + n_val:])

    print(f"  Train patients: {len(train_patients)}")
    print(f"  Val patients: {len(val_patients)}")
    print(f"  Test patients: {len(test_patients)}")

    # Verify no overlap
    assert len(train_patients & val_patients) == 0, "Train-val overlap!"
    assert len(train_patients & test_patients) == 0, "Train-test overlap!"
    assert len(val_patients & test_patients) == 0, "Val-test overlap!"
    print("  VERIFIED: No patient overlap between splits!")

    # Assign splits
    df['split'] = df['patient_id'].apply(
        lambda x: 'train' if x in train_patients else ('valid' if x in val_patients else 'test')
    )

    stats = {}

    for split in ['train', 'valid', 'test']:
        print(f"\nProcessing {split} set...")
        df_split = df[df['split'] == split].reset_index(drop=True)

        image_paths = df_split['path_to_image'].apply(
            lambda x: os.path.join(image_root, x) if image_root else x
        ).tolist()
        h5_path = os.path.join(output_dir, f'rexgradient_{split}.h5')

        img_to_hdf5_parallel(image_paths, h5_path, resolution=resolution, num_workers=num_workers)

        # Save metadata with h5_index for guaranteed pairing
        meta_cols = ['id', 'patient_id', 'path_to_image', 'Impression', 'Findings']
        meta_cols = [c for c in meta_cols if c in df_split.columns]
        split_meta = df_split[meta_cols].copy().reset_index(drop=True)
        split_meta['h5_index'] = split_meta.index
        split_meta.to_csv(
            os.path.join(output_dir, f'rexgradient_{split}_metadata.csv'),
            index=False
        )

        split_patients = train_patients if split == 'train' else (val_patients if split == 'valid' else test_patients)
        stats[split] = {
            'images': len(df_split),
            'patients': len(split_patients),
            'h5_path': h5_path
        }

    # Print summary
    print("\n" + "="*60)
    print("ReXGradient Preprocessing Summary")
    print("="*60)
    for split, s in stats.items():
        print(f"{split:>10}: {s['images']:>7} images, {s['patients']:>5} patients")
    print("="*60)

    return stats


def verify_h5_csv_pairing(h5_path: str, csv_path: str, num_samples: int = 5) -> bool:
    """
    Verify that H5 images and CSV metadata are correctly paired.

    Checks:
    1. H5 and CSV have same number of entries
    2. h5_index column matches actual indices

    Args:
        h5_path: Path to H5 file
        csv_path: Path to metadata CSV
        num_samples: Number of samples to print for verification

    Returns:
        True if pairing is correct
    """
    import h5py

    if not os.path.exists(h5_path) or not os.path.exists(csv_path):
        print(f"Files not found: {h5_path} or {csv_path}")
        return False

    with h5py.File(h5_path, 'r') as h5f:
        h5_len = h5f['cxr'].shape[0]

    df = pd.read_csv(csv_path)
    csv_len = len(df)

    print(f"H5 entries: {h5_len}, CSV entries: {csv_len}")

    if h5_len != csv_len:
        print(f"ERROR: Length mismatch! H5={h5_len}, CSV={csv_len}")
        return False

    if 'h5_index' in df.columns:
        # Verify h5_index is sequential 0 to N-1
        expected_indices = list(range(csv_len))
        actual_indices = df['h5_index'].tolist()
        if expected_indices != actual_indices:
            print("ERROR: h5_index column is not sequential!")
            return False
        print("OK: h5_index column is sequential")

    # Print sample pairings
    print(f"\nSample pairings (first {num_samples}):")
    path_col = 'path_to_image' if 'path_to_image' in df.columns else 'Path'
    for i in range(min(num_samples, len(df))):
        path = df.iloc[i][path_col] if path_col in df.columns else "N/A"
        impression = df.iloc[i].get('impression', df.iloc[i].get('Impression', 'N/A'))
        if isinstance(impression, str) and len(impression) > 50:
            impression = impression[:50] + "..."
        print(f"  [{i}] {os.path.basename(str(path))}: {impression}")

    print("\nOK: H5 and CSV pairing verified!")
    return True


def verify_no_patient_overlap(output_dir: str, dataset: str) -> bool:
    """
    Verify that there is no patient overlap between train/val/test splits.

    Args:
        output_dir: Directory containing metadata CSVs
        dataset: 'chexpert_plus' or 'rexgradient'

    Returns:
        True if no overlap, False otherwise
    """
    splits = ['train', 'valid', 'test']
    patient_sets = {}

    for split in splits:
        csv_path = os.path.join(output_dir, f'{dataset}_{split}_metadata.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if 'patient_id' in df.columns:
                patient_sets[split] = set(df['patient_id'].unique())
            elif 'Path' in df.columns:
                patient_sets[split] = set(
                    df['Path'].apply(extract_patient_id_from_path).unique()
                )

    # Check all pairs
    all_good = True
    for i, s1 in enumerate(splits):
        for s2 in splits[i+1:]:
            if s1 in patient_sets and s2 in patient_sets:
                overlap = patient_sets[s1] & patient_sets[s2]
                if overlap:
                    print(f"WARNING: {len(overlap)} patients overlap between {s1} and {s2}")
                    all_good = False
                else:
                    print(f"OK: No overlap between {s1} and {s2}")

    return all_good


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess CheXpert-Plus and ReXGradient with patient-level splits'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['chexpert-plus', 'rexgradient', 'both'],
        help='Dataset to preprocess'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Directory containing input CSV files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/h5',
        help='Directory for output H5 files'
    )
    parser.add_argument(
        '--image_root',
        type=str,
        default='',
        help='Root directory for images (prepended to paths in CSV)'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=448,
        help='Target image resolution'
    )
    parser.add_argument(
        '--verify_only',
        action='store_true',
        help='Only verify existing splits (no preprocessing)'
    )
    parser.add_argument(
        '--verify_pairing',
        action='store_true',
        help='Verify H5 and CSV pairing after preprocessing'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for ReXGradient split'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: auto)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.verify_only:
        print("Verifying patient splits...")
        if args.dataset in ['chexpert-plus', 'both']:
            print("\nCheXpert-Plus:")
            verify_no_patient_overlap(args.output_dir, 'chexpert_plus')
        if args.dataset in ['rexgradient', 'both']:
            print("\nReXGradient:")
            verify_no_patient_overlap(args.output_dir, 'rexgradient')

    elif args.verify_pairing:
        print("Verifying H5-CSV pairing...")
        splits = ['train', 'valid', 'test']
        if args.dataset in ['chexpert-plus', 'both']:
            print("\n=== CheXpert-Plus ===")
            for split in splits:
                print(f"\n--- {split} ---")
                h5_path = os.path.join(args.output_dir, f'chexpert_plus_{split}.h5')
                csv_path = os.path.join(args.output_dir, f'chexpert_plus_{split}_metadata.csv')
                verify_h5_csv_pairing(h5_path, csv_path)
        if args.dataset in ['rexgradient', 'both']:
            print("\n=== ReXGradient ===")
            for split in splits:
                print(f"\n--- {split} ---")
                h5_path = os.path.join(args.output_dir, f'rexgradient_{split}.h5')
                csv_path = os.path.join(args.output_dir, f'rexgradient_{split}_metadata.csv')
                verify_h5_csv_pairing(h5_path, csv_path)

    else:
        if args.dataset in ['chexpert-plus', 'both']:
            preprocess_chexpert_plus(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                image_root=args.image_root,
                resolution=args.resolution,
                num_workers=args.num_workers
            )

        if args.dataset in ['rexgradient', 'both']:
            preprocess_rexgradient(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                image_root=args.image_root,
                resolution=args.resolution,
                seed=args.seed,
                num_workers=args.num_workers
            )
