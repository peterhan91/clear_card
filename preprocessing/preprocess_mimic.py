#!/usr/bin/env python3
"""
Preprocess MIMIC-CXR JPEGs into per-split H5 files for the PheWAS
phenotype experiments.

Reads mimic_phewas_labels.parquet (which has a 'split' column with
train/validate/test from the official MIMIC-CXR 2.0.0 split),
resolves JPEG paths, and converts each split into an H5 file via
img_to_hdf5_parallel.

Usage:
  python preprocess_mimic.py \
      --labels ../data/mimic_phewas_labels.parquet \
      --mimic_jpg_dir /cbica/projects/CXR/data/MIMIC-CXR/data/images/mimic-cxr-jpg-2.1.0/mimic-cxr-jpg-2.1.0.physionet.org \
      --output_dir /cbica/projects/CXR/data_p \
      --resolution 448 \
      --num_workers 16
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_process import img_to_hdf5_parallel


def build_jpeg_path(row, mimic_jpg_dir: str) -> str:
    """Build the full JPEG path from subject_id, study_id, dicom_id."""
    sid = str(row['subject_id'])
    prefix = f"p{sid[:2]}"
    return os.path.join(
        mimic_jpg_dir, 'files', prefix,
        f"p{sid}", f"s{row['study_id']}",
        f"{row['dicom_id']}.jpg",
    )


def load_labels(path: str) -> pd.DataFrame:
    """Load labels from parquet or CSV."""
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def preprocess_mimic(
    labels_path: str,
    mimic_jpg_dir: str,
    output_dir: str,
    resolution: int = 448,
    num_workers: int = None,
):
    os.makedirs(output_dir, exist_ok=True)

    df = load_labels(labels_path)
    print(f"Loaded {len(df)} rows from {labels_path}")
    print(f"Splits: {df['split'].value_counts().to_dict()}")

    split_map = {
        'train': 'mimic_train.h5',
        'validate': 'mimic_validate.h5',
        'test': 'mimic_test.h5',
    }

    for split_name, h5_name in split_map.items():
        df_split = df[df['split'] == split_name].reset_index(drop=True)
        if len(df_split) == 0:
            print(f"Skipping {split_name}: 0 images")
            continue

        paths = [build_jpeg_path(row, mimic_jpg_dir) for _, row in df_split.iterrows()]

        # Quick sanity check on first image
        if not os.path.exists(paths[0]):
            print(f"ERROR: first image not found: {paths[0]}")
            print("Check --mimic_jpg_dir")
            return

        h5_path = os.path.join(output_dir, h5_name)
        print(f"\n{'='*60}")
        print(f"Processing {split_name}: {len(paths)} images -> {h5_path}")
        print(f"{'='*60}")

        img_to_hdf5_parallel(
            paths, h5_path,
            resolution=resolution,
            num_workers=num_workers,
        )

        # Save metadata with h5_index
        meta = df_split[['dicom_id', 'subject_id', 'study_id', 'split',
                          'ViewPosition', 'icd_source']].copy()
        meta['h5_index'] = meta.index
        meta_path = os.path.join(output_dir, h5_name.replace('.h5', '_metadata.csv'))
        meta.to_csv(meta_path, index=False)
        print(f"  Metadata saved to {meta_path}")

    print(f"\n{'='*60}")
    print("MIMIC-CXR preprocessing complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess MIMIC-CXR JPEGs into per-split H5 files")
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to labels parquet or CSV')
    parser.add_argument('--mimic_jpg_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=448)
    parser.add_argument('--num_workers', type=int, default=16)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    preprocess_mimic(
        labels_path=args.labels,
        mimic_jpg_dir=args.mimic_jpg_dir,
        output_dir=args.output_dir,
        resolution=args.resolution,
        num_workers=args.num_workers,
    )
