#!/usr/bin/env python3
"""
Debug script: LR sweep for CLEAR linear probing on MIMIC PheWAS (542 labels).
Uses cached LLM-projected features. Logs per-epoch train loss, val AUC, test AUC.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from exp_linear_mimic import (
    load_all_splits, load_concepts, load_concept_embeddings,
    project_to_llm_space, LogisticRegressionModel,
    DATASET_CONFIGS, EMBEDDING_MODELS, get_cache_tag,
    DEFAULT_MODEL_PATH, MIN_TEST_POSITIVES,
)

SEED = 42
BATCH_SIZE = 512
MAX_EPOCHS = 200
PATIENCE = 10

# LR sweep values
LEARNING_RATES = [5e-2, 2e-2, 1e-2, 5e-3, 1e-3]
WEIGHT_DECAYS = [0, 1e-8]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_aucs(y_true, y_pred, labels, min_positives=MIN_TEST_POSITIVES):
    aucs = {}
    for i, label in enumerate(labels):
        n_pos = int(y_true[:, i].sum())
        if n_pos < min_positives or n_pos == len(y_true):
            continue
        try:
            aucs[label] = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            pass
    return aucs


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_t, all_p = [], []
    for inputs, targets in loader:
        outputs = model(inputs.to(device))
        all_t.append(targets.numpy())
        all_p.append(outputs.cpu().numpy())
    return np.concatenate(all_t), np.concatenate(all_p)


def run_one(train_repr, train_labels, val_repr, val_labels,
            test_repr, test_labels, label_cols, lr, wd, device):
    set_seed(SEED)
    input_dim = train_repr.shape[1]
    output_dim = len(label_cols)

    train_ds = TensorDataset(torch.from_numpy(train_repr).float(),
                             torch.from_numpy(train_labels).float())
    val_ds = TensorDataset(torch.from_numpy(val_repr).float(),
                           torch.from_numpy(val_labels).float())
    test_ds = TensorDataset(torch.from_numpy(test_repr).float(),
                            torch.from_numpy(test_labels).float())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = LogisticRegressionModel(input_dim, output_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val_auc = 0.0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_ctr = 0
    history = []

    for epoch in range(MAX_EPOCHS):
        # Train
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # Val
        y_t_val, y_p_val = evaluate(model, val_loader, device)
        val_aucs = compute_aucs(y_t_val, y_p_val, label_cols, min_positives=1)
        val_mean = np.mean(list(val_aucs.values())) if val_aucs else 0.0

        # Test (for monitoring only — not used for selection)
        y_t_test, y_p_test = evaluate(model, test_loader, device)
        test_aucs = compute_aucs(y_t_test, y_p_test, label_cols)
        test_mean = np.mean(list(test_aucs.values())) if test_aucs else 0.0

        # Weight stats
        w = model.linear.weight.data
        w_norm = w.norm().item()
        w_max = w.abs().max().item()
        grad_norm = model.linear.weight.grad.norm().item() if model.linear.weight.grad is not None else 0

        row = {
            'epoch': epoch + 1,
            'train_loss': round(avg_loss, 6),
            'val_auc': round(val_mean, 4),
            'test_auc': round(test_mean, 4),
            'n_val_eval': len(val_aucs),
            'n_test_eval': len(test_aucs),
            'w_norm': round(w_norm, 4),
            'w_max': round(w_max, 6),
            'grad_norm': round(grad_norm, 6),
        }
        history.append(row)

        print(f"  ep {epoch+1:3d}  loss={avg_loss:.5f}  "
              f"val={val_mean:.4f}  test={test_mean:.4f}  "
              f"|W|={w_norm:.2f}  max|W|={w_max:.5f}  |grad|={grad_norm:.5f}")

        if val_mean > best_val_auc:
            best_val_auc = val_mean
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    # Final test with best model
    model.load_state_dict(best_state)
    model.to(device)
    y_t_test, y_p_test = evaluate(model, test_loader, device)
    test_aucs = compute_aucs(y_t_test, y_p_test, label_cols)
    final_test = np.mean(list(test_aucs.values())) if test_aucs else 0.0

    # Check prediction distribution
    preds_flat = y_p_test.flatten()
    print(f"  Prediction stats: mean={preds_flat.mean():.4f}  "
          f"std={preds_flat.std():.4f}  "
          f"min={preds_flat.min():.4f}  max={preds_flat.max():.4f}")

    # Check how many labels have AUC < 0.5
    below_05 = sum(1 for a in test_aucs.values() if a < 0.5)
    print(f"  Labels with AUC < 0.5: {below_05}/{len(test_aucs)}")

    return {
        'lr': lr, 'wd': wd,
        'best_val_auc': round(best_val_auc, 4),
        'final_test_auc': round(final_test, 4),
        'n_test_eval': len(test_aucs),
        'below_05': below_05,
        'stopped_epoch': len(history),
        'history': history,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = 'mimic'
    model_key = 'kalm_gemma3_12b'

    print("=" * 70)
    print("CLEAR LR Sweep Debug — MIMIC PheWAS 542")
    print("=" * 70)

    # Load labels
    print("\n[1] Loading labels...")
    default_labels = DATASET_CONFIGS[dataset]['default_labels']
    splits = load_all_splits(default_labels, dataset=dataset)
    train_df, train_labels, label_cols = splits['train']
    val_df, val_labels, _ = splits['validate']
    test_df, test_labels, _ = splits['test']
    print(f"  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}  "
          f"labels={len(label_cols)}")

    # Load cached features
    print("\n[2] Loading cached features...")
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    tag = get_cache_tag(DEFAULT_MODEL_PATH)

    image_features = {}
    for split in ['train', 'validate', 'test']:
        path = os.path.join(cache_dir, f"mimic_{split}_{tag}.pt")
        image_features[split] = torch.load(path, weights_only=True)
        print(f"  {split}: {image_features[split].shape}")

    concept_path = os.path.join(cache_dir, f"concept_features_{tag}.pt")
    concept_features = torch.load(concept_path, weights_only=True)
    print(f"  concepts: {concept_features.shape}")

    # Load concept embeddings
    print("\n[3] Loading concept embeddings...")
    concepts, concept_indices = load_concepts()
    concept_embeds = load_concept_embeddings(model_key, concept_indices, concepts)
    print(f"  LLM embeddings: {concept_embeds.shape}")

    # Project to LLM space
    print("\n[4] Projecting to LLM space...")
    split_repr = {}
    for split in ['train', 'validate', 'test']:
        repr_t = project_to_llm_space(
            image_features[split], concept_features, concept_embeds, batch_size=500)
        split_repr[split] = repr_t.numpy()
        print(f"  {split}: {split_repr[split].shape}")

    # Inspect feature stats
    print("\n[5] Feature statistics:")
    for split in ['train', 'validate', 'test']:
        r = split_repr[split]
        print(f"  {split}: mean={r.mean():.6f}  std={r.std():.6f}  "
              f"min={r.min():.4f}  max={r.max():.4f}  "
              f"norm_mean={np.linalg.norm(r, axis=1).mean():.4f}")

    # Check label prevalence
    print("\n[6] Label prevalence in train set:")
    prevalences = train_labels.sum(axis=0) / len(train_labels)
    print(f"  mean={prevalences.mean():.4f}  median={np.median(prevalences):.4f}  "
          f"min={prevalences.min():.4f}  max={prevalences.max():.4f}")
    low_prev = (prevalences < 0.01).sum()
    very_low = (prevalences < 0.001).sum()
    print(f"  Labels with <1% prevalence: {low_prev}/{len(label_cols)}")
    print(f"  Labels with <0.1% prevalence: {very_low}/{len(label_cols)}")

    # Free memory
    del image_features, concept_features, concept_embeds
    torch.cuda.empty_cache()

    # Run LR sweep
    results = []
    for wd in WEIGHT_DECAYS:
        for lr in LEARNING_RATES:
            print(f"\n{'=' * 70}")
            print(f"LR={lr}  WD={wd}")
            print(f"{'=' * 70}")
            t0 = time.time()
            res = run_one(
                split_repr['train'], train_labels,
                split_repr['validate'], val_labels,
                split_repr['test'], test_labels,
                label_cols, lr, wd, device)
            elapsed = time.time() - t0
            res['elapsed_s'] = round(elapsed, 1)
            results.append(res)
            print(f"  => val={res['best_val_auc']}  test={res['final_test_auc']}  "
                  f"stopped={res['stopped_epoch']}  time={elapsed:.0f}s")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'LR':>10s}  {'WD':>10s}  {'Val AUC':>8s}  {'Test AUC':>9s}  "
          f"{'<0.5':>5s}  {'Epoch':>5s}  {'Time':>6s}")
    print("-" * 65)
    for r in results:
        print(f"{r['lr']:>10.0e}  {r['wd']:>10.0e}  {r['best_val_auc']:>8.4f}  "
              f"{r['final_test_auc']:>9.4f}  {r['below_05']:>5d}  "
              f"{r['stopped_epoch']:>5d}  {r['elapsed_s']:>5.0f}s")

    # Save full results
    save_path = os.path.join(os.path.dirname(__file__), 'results', 'debug_lr_sweep.json')
    # Strip history for JSON (keep only summary)
    save_results = []
    for r in results:
        sr = {k: v for k, v in r.items() if k != 'history'}
        sr['history_epochs'] = [h['epoch'] for h in r['history']]
        sr['history_val'] = [h['val_auc'] for h in r['history']]
        sr['history_test'] = [h['test_auc'] for h in r['history']]
        sr['history_loss'] = [h['train_loss'] for h in r['history']]
        sr['history_wnorm'] = [h['w_norm'] for h in r['history']]
        save_results.append(sr)
    with open(save_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nFull results saved to {save_path}")


if __name__ == "__main__":
    main()
