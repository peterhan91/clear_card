#!/usr/bin/env python3
"""
ROC curve comparison: concept-based linear probing vs. foundation models
on MIMIC-CXR 27 opportunistic screening phenotypes.

Loads per-seed prediction pickles from results/ directories and generates
one ROC curve plot per phenotype label, with each method overlaid as a
separate curve with 95% CI shaded band.

Adapted from cxr_concept/concepts/plot_linear_probing_roc_curves.py for
the CLEAR pipeline with 27 MIMIC-CXR phenotypes.

Usage:
  python plot_auroc_comparison.py
  python plot_auroc_comparison.py --labels heart_failure copd lung_cancer
  python plot_auroc_comparison.py --core_only
"""

import os
import glob
import pickle
import argparse
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SKIP_EVAL_LABELS = {'pulmonary_valve'}

CORE_PHENOTYPES = [
    'heart_failure', 'hfref', 'hfpef', 'atrial_fibrillation',
    'cad', 'pulmonary_htn', 'copd', 'lung_cancer',
]

LABEL_DISPLAY = {
    'heart_failure': 'Heart Failure',
    'hfref': 'HFrEF',
    'hfpef': 'HFpEF',
    'atrial_fibrillation': 'Atrial Fibrillation',
    'mitral_valve': 'Mitral Valve Disease',
    'aortic_valve': 'Aortic Valve Disease',
    'tricuspid_valve': 'Tricuspid Valve Disease',
    'cad': 'Coronary Artery Disease',
    'myocardial_infarction': 'Myocardial Infarction',
    'pulmonary_htn': 'Pulmonary Hypertension',
    'stroke': 'Stroke',
    'copd': 'COPD',
    'asthma': 'Asthma',
    'ild': 'Interstitial Lung Disease',
    'lung_cancer': 'Lung Cancer',
    'tuberculosis': 'Tuberculosis',
    't2dm': 'Type 2 Diabetes',
    'obesity': 'Obesity',
    'dyslipidemia': 'Dyslipidemia',
    'osteoporosis': 'Osteoporosis',
    'spinal_stenosis': 'Spinal Stenosis',
    'hypertension': 'Hypertension',
    'ckd': 'Chronic Kidney Disease',
    'pvd': 'Peripheral Vascular Disease',
    'liver_cirrhosis': 'Liver Cirrhosis',
    'anemia': 'Anemia',
}

# Method directory mapping and display styles
# Order determines legend order; color matches cxr_concept palette
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_predictions(results_dir):
    """Load per-seed prediction pickles from a results directory.

    Returns list of (y_true, y_pred) and the label list.
    """
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


def discover_methods():
    """Auto-discover available method results with prediction pickles."""
    available = OrderedDict()
    for key, cfg in METHOD_CONFIG.items():
        rdir = os.path.join(RESULTS_BASE, cfg['dir'])
        preds, labels = load_predictions(rdir)
        if preds:
            available[key] = {
                'predictions': preds,
                'labels': labels,
                'name': cfg['name'],
                'color': cfg['color'],
            }
            print(f"  {cfg['name']:<20s} {len(preds)} seeds, "
                  f"{np.array(preds[0][0]).shape[0]} test samples")
    return available


# ---------------------------------------------------------------------------
# ROC computation
# ---------------------------------------------------------------------------
def compute_roc_with_ci(predictions, label_idx):
    """Compute mean ROC curve and 95% CI across seeds.

    Returns (fpr_grid, tpr_mean, tpr_lower, tpr_upper, auc_scores).
    """
    fpr_grid = np.linspace(0, 1, 100)
    tpr_curves = []
    auc_scores = []

    for y_true, y_pred in predictions:
        yt = y_true[:, label_idx]
        yp = y_pred[:, label_idx]
        if len(np.unique(yt)) < 2:
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
def plot_roc_for_label(methods, label_name, output_dir):
    """Plot ROC curves for one label using per-method label indices."""
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

        print(f"  {info['name']}: AUC = {np.mean(auc_scores):.3f} "
              f"\u00b1 {np.std(auc_scores):.3f}")

    if valid == 0:
        print(f"  Skipping {label_name} -- no valid ROC curves")
        plt.close(fig)
        return

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('False positive rate', fontsize=34)
    ax.set_ylabel('True positive rate', fontsize=34)
    display = LABEL_DISPLAY.get(label_name, label_name)
    ax.set_title(display, fontsize=34, pad=10)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True,
              fontsize=22, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    os.makedirs(output_dir, exist_ok=True)
    safe = label_name.replace(' ', '_').lower()
    for ext in ['png', 'pdf']:
        path = os.path.join(output_dir, f"roc_curve_{safe}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white',
                    edgecolor='none')
    print(f"  Saved: roc_curve_{safe}.{{png,pdf}}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='ROC curve comparison: concept-based vs. foundation models')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Specific labels to plot (default: core phenotypes)')
    parser.add_argument('--all_labels', action='store_true',
                        help='Plot all 26 labels')
    parser.add_argument('--core_only', action='store_true',
                        help='Plot only core phenotypes')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'results', 'roc_curves'),
                        help='Output directory for plots')
    args = parser.parse_args()

    print('=' * 60)
    print('ROC Curve Comparison: MIMIC-CXR Opportunistic Screening')
    print('=' * 60)

    # Discover available methods
    print('\nDiscovering results ...')
    methods = discover_methods()
    if not methods:
        print('No prediction files found in results/ directories.')
        return

    # Get union of labels across all methods
    all_labels_set = set()
    for info in methods.values():
        all_labels_set.update(info['labels'])
    all_labels_set -= SKIP_EVAL_LABELS

    # Use first method's label order as canonical
    first_labels = list(methods.values())[0]['labels']
    ordered_labels = [l for l in first_labels
                      if l in all_labels_set and l not in SKIP_EVAL_LABELS]

    # Filter
    if args.labels:
        ordered_labels = [l for l in args.labels if l in all_labels_set]
    elif args.core_only or not args.all_labels:
        ordered_labels = [l for l in ordered_labels if l in CORE_PHENOTYPES]

    print(f'\nPlotting {len(methods)} methods x {len(ordered_labels)} labels\n')

    for label in ordered_labels:
        # Find label index per method (may differ if label sets differ)
        methods_with_idx = OrderedDict()
        for key, info in methods.items():
            if label in info['labels']:
                methods_with_idx[key] = {
                    **info,
                    '_label_idx': info['labels'].index(label),
                }

        if not methods_with_idx:
            continue

        print(f"Processing {label} ...")
        plot_roc_for_label(methods_with_idx, label, args.output_dir)

    print(f'\nAll ROC curves saved to {args.output_dir}/')


if __name__ == '__main__':
    main()
