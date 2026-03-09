# Project: CLEAR (Concept-Level Explanation And Reasoning) for CXR

## Environment

- **Package manager:** micromamba (not conda/mamba)
- **Primary env:** `ml311` — Python 3.11, PyTorch 2.8+cu129, peft, transformers
- **Activate before any python/pip command:**
  ```bash
  eval "$(micromamba shell hook --shell bash)" && micromamba activate ml311
  ```
- **Env prefix:** `/cbica/home/hanti/.conda/envs/ml311`
- All `python` and `pip` commands must run inside the activated micromamba env.

## Key Paths

- **Checkpoints:** `/cbica/projects/CXR/codes/clear_card/checkpoints/`
- **MIMIC-CXR JPEGs:** `/cbica/projects/CXR/data_p/mimic-cxr-jpg/`
- **Labels CSV:** `data/mimic_opportunistic_labels.csv`
- **Concept embeddings:** `concepts/embeddings_output/`
- **Feature cache:** `concepts/cache/`
- **Results:** `concepts/results/`

## Best CLEAR Model

- DINOv3 ViT-7B LoRA (r=32, alpha=64), combined CheXpert+ReXGradient
- Checkpoint: `checkpoints/dinov3-vit7b16-lora-r32-a64-combined_vit7b16_20260224_144733/best_model.pt`
