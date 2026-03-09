#!/usr/bin/env python3
"""
AUROC comparison: concept-based linear probing vs. foundation models
on MIMIC-CXR PheWAS phenotypes.

Generates:
  1. Macro AUROC bar chart comparing all methods
  2. Per-phenotype ROC curves for top phenotypes (by CLEAR AUC)
  3. Summary CSV with per-phenotype AUCs across all methods

Usage:
  python plot_auroc_comparison.py
  python plot_auroc_comparison.py --top_k 20
  python plot_auroc_comparison.py --phecodes 428.1 496.0 162.0
"""

import os
import glob
import json
import pickle
import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PHECODE_INFO_PATH = os.path.join(os.path.dirname(__file__), '..', 'data',
                                  'mimic_phecode_info.csv')

MIN_TEST_POSITIVES = 20

# Method directory mapping and display styles
METHOD_CONFIG = OrderedDict([
    ('concept_kalm', {
        'dir': 'linear_mimic_kalm_gemma3_12b',
        'name': 'CLEAR',
        'color': 'indianred',
    }),
    ('ark_plus', {
        'dir': 'linear_mimic_foundation_ark_plus',
        'name': 'Ark+',
        'color': 'steelblue',
    }),
    ('rad_dino', {
        'dir': 'linear_mimic_foundation_rad_dino',
        'name': 'RAD-DINO',
        'color': 'darkgreen',
    }),
    ('chexzero', {
        'dir': 'linear_mimic_foundation_chexzero',
        'name': 'CheXzero',
        'color': 'darkorange',
    }),
    ('biomedclip', {
        'dir': 'linear_mimic_foundation_biomedclip',
        'name': 'BiomedCLIP',
        'color': 'mediumpurple',
    }),
])

RESULTS_BASE = os.path.join(os.path.dirname(__file__), 'results')


def load_phecode_info():
    """Load phecode -> phenotype name mapping."""
    if not os.path.exists(PHECODE_INFO_PATH):
        return {}
    df = pd.read_csv(PHECODE_INFO_PATH, dtype={'phecode': str})
    return dict(zip(df['phecode'], df['phenotype']))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_predictions(results_dir):
    """Load per-seed prediction pickles from a results directory."""
    pred_dir = os.path.join(results_dir, 'predictions')
    paths = sorted(glob.glob(os.path.join(pred_dir, 'seed_*_predictions.pkl')))
    if not paths:
        return [], None

    predictions = []
    labels = None
    for p in paths:
        with open(p, 'rb') as f:
            d = pickle.load(f)
        y_true = np.array(d['y_true'])
        y_pred = np.array(d['y_pred'])
        predictions.append((y_true, y_pred))
        if labels is None:
            labels = d['labels']

    return predictions, labels


def load_summary_aucs(results_dir):
    """Load per-label AUCs from summary JSON."""
    pattern = os.path.join(results_dir, "summary_*.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        return {}
    with open(paths[-1]) as f:
        summary = json.load(f)
    return summary.get('per_label_stats', {})


def discover_methods():
    """Auto-discover available method results with prediction pickles."""
    available = OrderedDict()
    for key, cfg in METHOD_CONFIG.items():
        rdir = os.path.join(RESULTS_BASE, cfg['dir'])
        preds, labels = load_predictions(rdir)
        if preds:
            per_label = load_summary_aucs(rdir)
            available[key] = {
                'predictions': preds,
                'labels': labels,
                'per_label_stats': per_label,
                'name': cfg['name'],
                'color': cfg['color'],
            }
            print(f"  {cfg['name']:<20s} {len(preds)} seeds, "
                  f"{np.array(preds[0][0]).shape[0]} test samples, "
                  f"{len(labels)} labels")
    return available


# ---------------------------------------------------------------------------
# ROC computation
# ---------------------------------------------------------------------------
def compute_roc_with_ci(predictions, label_idx):
    """Compute mean ROC curve and 95% CI across seeds."""
    fpr_grid = np.linspace(0, 1, 100)
    tpr_curves = []
    auc_scores = []

    for y_true, y_pred in predictions:
        yt = y_true[:, label_idx]
        yp = y_pred[:, label_idx]
        n_pos = int(yt.sum())
        if n_pos < MIN_TEST_POSITIVES or n_pos == len(yt):
            continue
        fpr, tpr, _ = roc_curve(yt, yp)
        tpr_interp = np.interp(fpr_grid, fpr, tpr)
        tpr_curves.append(tpr_interp)
        auc_scores.append(auc(fpr, tpr))

    if not tpr_curves:
        return None

    tpr_curves = np.array(tpr_curves)
    return (
        fpr_grid,
        np.mean(tpr_curves, axis=0),
        np.percentile(tpr_curves, 2.5, axis=0),
        np.percentile(tpr_curves, 97.5, axis=0),
        np.array(auc_scores),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_macro_auroc_bar(methods, output_dir, phecode_names):
    """Bar chart comparing macro AUROC across methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    names = []
    means = []
    stds = []
    colors = []

    for key, info in methods.items():
        per_label = info['per_label_stats']
        if not per_label:
            continue
        aucs = [v['mean'] for v in per_label.values()]
        names.append(info['name'])
        means.append(np.mean(aucs))
        stds.append(np.std(aucs))
        colors.append(info['color'])

    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=1, capsize=5,
                  error_kw={'linewidth': 1.5})

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=16, rotation=15)
    ax.set_ylabel('Macro AUROC', fontsize=18)
    ax.set_title('MIMIC-CXR PheWAS Linear Probing Comparison', fontsize=20)
    ax.tick_params(axis='y', labelsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')

    # Value labels on bars
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.002,
                f'{m:.3f}', ha='center', va='bottom', fontsize=13, fontweight='bold')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(output_dir, f"macro_auroc_comparison.{ext}"),
                    dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved macro_auroc_comparison.{{png,pdf}}")
    plt.close(fig)


def plot_roc_for_label(methods, label_name, output_dir, phecode_names):
    """Plot ROC curves for one phenotype label."""
    fig, ax = plt.subplots(figsize=(8, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_aspect('equal')

    valid = 0
    for key, info in methods.items():
        label_idx = info['_label_idx']
        result = compute_roc_with_ci(info['predictions'], label_idx)
        if result is None:
            continue
        fpr_grid, tpr_mean, tpr_lo, tpr_hi, auc_scores = result
        valid += 1

        ax.plot(fpr_grid, tpr_mean, color=info['color'], linewidth=3,
                alpha=1.0,
                label=f"{info['name']} (AUC = {np.mean(auc_scores):.3f})")
        ax.fill_between(fpr_grid, tpr_lo, tpr_hi,
                        color=info['color'], alpha=0.2)

    if valid == 0:
        plt.close(fig)
        return

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('False positive rate', fontsize=34)
    ax.set_ylabel('True positive rate', fontsize=34)

    display = phecode_names.get(label_name, label_name)
    ax.set_title(display, fontsize=28, pad=10)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True,
              fontsize=18, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    os.makedirs(output_dir, exist_ok=True)
    safe = label_name.replace('.', '_').replace(' ', '_').lower()
    for ext in ['png', 'pdf']:
        path = os.path.join(output_dir, f"roc_curve_{safe}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white',
                    edgecolor='none')
    plt.close(fig)


def save_comparison_csv(methods, output_dir, phecode_names):
    """Save a CSV with per-phenotype AUCs across all methods."""
    rows = []
    # Collect all phecodes
    all_phecodes = set()
    for info in methods.values():
        all_phecodes.update(info['per_label_stats'].keys())

    for phecode in sorted(all_phecodes):
        row = {
            'phecode': phecode,
            'phenotype': phecode_names.get(phecode, phecode),
        }
        for key, info in methods.items():
            stats = info['per_label_stats'].get(phecode)
            if stats:
                row[f"{info['name']}_mean"] = stats['mean']
                row[f"{info['name']}_std"] = stats['std']
            else:
                row[f"{info['name']}_mean"] = np.nan
                row[f"{info['name']}_std"] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "auroc_comparison_all_phenotypes.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path} ({len(df)} phenotypes)")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='AUROC comparison: concept-based vs. foundation models')
    parser.add_argument('--phecodes', nargs='+', default=None,
                        help='Specific phecodes to plot ROC curves for')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Plot ROC curves for top-k phenotypes by CLEAR AUC')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'results', 'roc_curves'),
                        help='Output directory for plots')
    args = parser.parse_args()

    print('=' * 60)
    print('AUROC Comparison: MIMIC-CXR PheWAS Linear Probing')
    print('=' * 60)

    phecode_names = load_phecode_info()

    # Discover available methods
    print('\nDiscovering results ...')
    methods = discover_methods()
    if not methods:
        print('No prediction files found in results/ directories.')
        return

    # 1. Macro AUROC bar chart
    print('\nPlotting macro AUROC comparison ...')
    plot_macro_auroc_bar(methods, args.output_dir, phecode_names)

    # 2. Save comparison CSV
    print('\nSaving comparison CSV ...')
    comp_df = save_comparison_csv(methods, args.output_dir, phecode_names)

    # 3. Per-phenotype ROC curves
    # Determine which phenotypes to plot
    first_method = list(methods.values())[0]
    per_label = first_method['per_label_stats']

    if args.phecodes:
        plot_phecodes = args.phecodes
    else:
        # Top-k by the first method's (CLEAR's) AUC
        sorted_labels = sorted(per_label.items(),
                                key=lambda x: x[1]['mean'], reverse=True)
        plot_phecodes = [pc for pc, _ in sorted_labels[:args.top_k]]

    roc_dir = os.path.join(args.output_dir, 'per_phenotype')
    print(f'\nPlotting ROC curves for {len(plot_phecodes)} phenotypes ...')

    for phecode in plot_phecodes:
        methods_with_idx = OrderedDict()
        for key, info in methods.items():
            if phecode in info['labels']:
                methods_with_idx[key] = {
                    **info,
                    '_label_idx': info['labels'].index(phecode),
                }

        if not methods_with_idx:
            continue

        name = phecode_names.get(phecode, phecode)
        print(f"  {phecode} ({name[:40]})")
        plot_roc_for_label(methods_with_idx, phecode, roc_dir, phecode_names)

    print(f'\nAll outputs saved to {args.output_dir}/')


if __name__ == '__main__':
    main()
