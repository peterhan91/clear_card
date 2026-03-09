#!/usr/bin/env python3
"""
Preprocess CheXchoNet JPEGs into per-split H5 files.

Images are already 224x224 grayscale, so we keep resolution=224 by default.
Reads chexchonet_labels.csv (which has a 'split' column with train/valid/test),
resolves JPEG paths, and converts each split into an H5 file.

Usage:
  python preprocess_chexchonet_h5.py \
      --labels_csv ../data/chexchonet_labels.csv \
      --image_dir ../data/physionet.org/files/chexchonet/1.0.0/images \
      --output_dir ../data/h5_224_chexchonet \
      --resolution 224 \
      --num_workers 16
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data_process import img_to_hdf5_parallel


def preprocess_chexchonet_h5(
    labels_csv: str,
    image_dir: str,
    output_dir: str,
    resolution: int = 224,
    num_workers: int = None,
):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(labels_csv)
    print(f"Loaded {len(df)} rows from {labels_csv}")
    print(f"Splits: {df['split'].value_counts().to_dict()}")

    split_map = {
        'train': 'chexchonet_train.h5',
        'valid': 'chexchonet_valid.h5',
        'test': 'chexchonet_test.h5',
    }

    for split_name, h5_name in split_map.items():
        df_split = df[df['split'] == split_name].reset_index(drop=True)
        if len(df_split) == 0:
            print(f"Skipping {split_name}: 0 images")
            continue

        paths = [os.path.join(image_dir, fn) for fn in df_split['cxr_filename']]

        # Sanity check
        missing = sum(1 for p in paths if not os.path.exists(p))
        if missing > 0:
            print(f"WARNING: {missing}/{len(paths)} images not found for {split_name}")

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
    print("CheXchoNet H5 preprocessing complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Preprocess CheXchoNet JPEGs into per-split H5 files")
    parser.add_argument('--labels_csv', type=str, default='data/chexchonet_labels.csv')
    parser.add_argument('--image_dir', type=str,
                        default='data/physionet.org/files/chexchonet/1.0.0/images')
    parser.add_argument('--output_dir', type=str, default='data/h5_224_chexchonet')
    parser.add_argument('--resolution', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()

    preprocess_chexchonet_h5(
        labels_csv=args.labels_csv,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        resolution=args.resolution,
        num_workers=args.num_workers,
    )
