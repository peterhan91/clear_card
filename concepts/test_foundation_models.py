#!/usr/bin/env python3
"""Quick smoke test: load each FM, encode 8 images, verify output shape."""
import sys, os, torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from exp_linear_mimic_foundation import (
    load_model_and_transform, encode_images, MIMICH5PILDataset,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Args:
    ark_checkpoint = "checkpoints/Ark6_swinLarge768_ep50.pth.tar"
    chexzero_checkpoint = "checkpoints/CheXzero_Models/best_64_0.0001_original_17000_0.863.pt"

args = Args()
h5_path = "data/h5/mimic_test.h5"

models = ['rad_dino', 'biomedclip', 'chexzero', 'ark_plus']

for model_name in models:
    print(f"\n{'='*50}")
    print(f"Testing: {model_name}")
    try:
        model, xform, feat_dim = load_model_and_transform(model_name, args, device)
        ds = MIMICH5PILDataset(h5_path, num_samples=8, transform=xform)
        loader = DataLoader(ds, batch_size=8, num_workers=0)
        feats = encode_images(model_name, model, loader, device)
        assert feats.shape == (8, feat_dim)
        print(f"  PASS  shape={feats.shape}")
        del model; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback; traceback.print_exc()

print(f"\nAll tests complete.")
