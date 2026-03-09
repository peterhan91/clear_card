#!/usr/bin/env python3
"""
Preprocess PadChest PNGs into per-split H5 files.

Reads padchest_labels.csv (which has a 'split' column with train/valid/test),
resolves PNG paths by scanning the PadChest image directory, and converts each
split into an H5 file via img_to_hdf5_parallel.

Usage:
  python preprocess_padchest_h5.py \
      --labels_csv ../data/padchest_labels.csv \
      --padchest_img_dir /cbica/projects/CXR/data/PadChest \
      --output_dir ../data/h5_1024_padchest \
      --resolution 1024 \
      --num_workers 16
"""
import argparse
import os
import sys
import glob
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_process import img_to_hdf5_parallel


def build_image_index(padchest_img_dir: str) -> dict:
    """Build filename -> full path mapping by scanning subdirectories."""
    print(f"Scanning {padchest_img_dir} for PNG files...")
    all_pngs = glob.glob(os.path.join(padchest_img_dir, '*', '*.png'))
    index = {os.path.basename(p): p for p in all_pngs}
    print(f"  Found {len(index)} unique PNGs")
    return index


def preprocess_padchest_h5(
    labels_csv: str,
    padchest_img_dir: str,
    output_dir: str,
    resolution: int = 1024,
    num_workers: int = None,
):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(labels_csv)
    print(f"Loaded {len(df)} rows from {labels_csv}")
    print(f"Splits: {df['split'].value_counts().to_dict()}")

    img_index = build_image_index(padchest_img_dir)

    split_map = {
        'train': 'padchest_train.h5',
        'valid': 'padchest_valid.h5',
        'test': 'padchest_test.h5',
    }

    for split_name, h5_name in split_map.items():
        df_split = df[df['split'] == split_name].reset_index(drop=True)
        if len(df_split) == 0:
            print(f"Skipping {split_name}: 0 images")
            continue

        paths = []
        missing = 0
        for img_id in df_split['ImageID']:
            if img_id in img_index:
                paths.append(img_index[img_id])
            else:
                paths.append(None)
                missing += 1

        if missing > 0:
            print(f"WARNING: {missing}/{len(df_split)} images not found for {split_name}")
            # Filter out missing
            valid_mask = [p is not None for p in paths]
            paths = [p for p in paths if p is not None]
            df_split = df_split[valid_mask].reset_index(drop=True)

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
        meta = df_split.copy()
        meta['h5_index'] = meta.index
        meta_path = os.path.join(output_dir, h5_name.replace('.h5', '_metadata.csv'))
        meta.to_csv(meta_path, index=False)
        print(f"  Metadata saved to {meta_path}")

    print(f"\n{'='*60}")
    print("PadChest H5 preprocessing complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Preprocess PadChest PNGs into per-split H5 files")
    parser.add_argument('--labels_csv', type=str, default='data/padchest_labels.csv')
    parser.add_argument('--padchest_img_dir', type=str,
                        default='/cbica/projects/CXR/data/PadChest')
    parser.add_argument('--output_dir', type=str, default='data/h5_1024_padchest')
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()

    preprocess_padchest_h5(
        labels_csv=args.labels_csv,
        padchest_img_dir=args.padchest_img_dir,
        output_dir=args.output_dir,
        resolution=args.resolution,
        num_workers=args.num_workers,
    )
