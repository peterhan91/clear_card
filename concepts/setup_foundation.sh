#!/bin/bash
# ============================================================
# Setup dependencies for foundation model linear probing
# Run this ONCE on a node with internet access.
# ============================================================
set -e

echo "=========================================="
echo "Installing Python dependencies ..."
echo "=========================================="

# Core (likely already present)
pip install torch torchvision numpy pandas scikit-learn tqdm pillow h5py

# Ark+ (Swin Transformer via timm)
pip install "timm>=0.5.4"

# RAD-DINO (HuggingFace transformers)
pip install transformers

# BiomedCLIP (open_clip)
pip install open-clip-torch

# CheXzero (OpenAI CLIP architecture)
pip install git+https://github.com/openai/CLIP.git

echo ""
echo "=========================================="
echo "Pre-downloading model weights ..."
echo "=========================================="

# RAD-DINO  (microsoft/rad-dino on HuggingFace)
python -c "
from transformers import AutoModel, AutoImageProcessor
print('Downloading RAD-DINO ...')
AutoModel.from_pretrained('microsoft/rad-dino')
AutoImageProcessor.from_pretrained('microsoft/rad-dino')
print('  Done.')
"

# BiomedCLIP  (microsoft/BiomedCLIP on HuggingFace via open_clip)
python -c "
from open_clip import create_model_from_pretrained, get_tokenizer
print('Downloading BiomedCLIP ...')
create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
print('  Done.')
"

echo ""
echo "=========================================="
echo "Setup complete."
echo "=========================================="
echo ""
echo "Automatic downloads done for:"
echo "  - RAD-DINO   (microsoft/rad-dino)"
echo "  - BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)"
echo ""
echo "You still need to manually obtain:"
echo "  1. Ark+ checkpoint"
echo "     Request access: https://forms.gle/qkoDGXNiKRPTDdCe8"
echo "     Expected file:  Ark6_swinLarge768_ep50.pth.tar"
echo ""
echo "  2. CheXzero checkpoint"
echo "     Download from:  https://github.com/rajpurkarlab/CheXzero"
echo "     Expected file:  best_64_5e-05_original_22000_0.864.pt"
