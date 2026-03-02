"""
Clean extracted radiology concepts: fix truncated JSON, deduplicate, remove
short/uninformative observations, and produce a single clean concept vocabulary.

Usage:
    python concepts/clean_concepts.py
    python concepts/clean_concepts.py --input concepts/concepts_both.json --min-words 3
    python concepts/clean_concepts.py --dry-run   # preview without writing
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


# ── Uninformative patterns (regex, case-insensitive) ─────────────────────────
UNINFORMATIVE_PATTERNS = [
    r"^nan$",
    r"^unchanged$",
    r"^stable$",
    r"^none$",
    r"^normal$",
    r"^unremarkable$",
    r"^as above\.?$",
    r"^see above\.?$",
    r"^no change\.?$",
    r"^no comparison\.?$",
    r"^limited study\.?$",
    r"^patient is rotated$",
    r"^clinical correlation",
    r"^recommend ",
    r"^please refer ",
    r"^correlation with ",
    r"^compared to ",
    r"^little change since",
    r"^no significant (interval )?change",
]
_UNINFORMATIVE_RE = [re.compile(p, re.IGNORECASE) for p in UNINFORMATIVE_PATTERNS]


def is_uninformative(obs: str) -> bool:
    """Check if an observation matches known uninformative patterns."""
    return any(r.search(obs) for r in _UNINFORMATIVE_RE)


def try_fix_truncated(raw: str) -> list:
    """Attempt to recover observations from truncated JSON output."""
    text = raw.strip()
    if not text:
        return []

    # Already valid
    try:
        parsed = json.loads(text)
        return parsed.get("observations", []) if isinstance(parsed, dict) else []
    except json.JSONDecodeError:
        pass

    # Find the opening brace
    start = text.find("{")
    if start == -1:
        return []

    fragment = text[start:]

    # Try appending missing closing bracket + brace
    suffixes = ["}", '"}', '"]}', "]}"]
    for suffix in suffixes:
        try:
            parsed = json.loads(fragment + suffix)
            return parsed.get("observations", []) if isinstance(parsed, dict) else []
        except json.JSONDecodeError:
            continue

    return []


def normalize(obs: str) -> str:
    """Normalize an observation for deduplication."""
    s = obs.strip()
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    # Strip trailing period
    s = s.rstrip(".")
    return s


def normalize_key(obs: str) -> str:
    """Produce a lowercase key for dedup comparison."""
    return normalize(obs).lower()


def clean_concepts(
    input_path: str,
    output_path: str,
    min_words: int = 3,
    dry_run: bool = False,
):
    print(f"Loading {input_path} ...")
    with open(input_path) as f:
        data = json.load(f)
    print(f"  {len(data)} records loaded")

    # ── Phase 1: Fix truncated JSON ──────────────────────────────────────────
    n_fixed = 0
    for rec in data:
        if rec["n_observations"] == 0 and rec.get("raw_output", ""):
            recovered = try_fix_truncated(rec["raw_output"])
            if recovered:
                rec["observations"] = recovered
                rec["n_observations"] = len(recovered)
                n_fixed += 1
    print(f"  Phase 1 — Fixed {n_fixed} truncated records")

    # ── Phase 2: Collect all observations ────────────────────────────────────
    all_obs = []
    for rec in data:
        for obs in rec["observations"]:
            all_obs.append(normalize(obs))
    print(f"  Phase 2 — {len(all_obs)} total observations (pre-clean)")

    # ── Phase 3: Filter ──────────────────────────────────────────────────────
    removed_short = 0
    removed_uninformative = 0
    removed_dup = 0

    seen = {}          # normalize_key -> canonical form
    concept_counts = Counter()  # canonical -> count
    kept = []

    for obs in all_obs:
        word_count = len(obs.split())

        # Remove short
        if word_count < min_words:
            removed_short += 1
            continue

        # Remove uninformative
        if is_uninformative(obs):
            removed_uninformative += 1
            continue

        key = normalize_key(obs)

        # Deduplicate: keep the first-seen casing as canonical
        if key not in seen:
            seen[key] = obs
            kept.append(obs)

        concept_counts[key] += 1

    removed_dup = len(all_obs) - removed_short - removed_uninformative - len(kept)

    print(f"  Phase 3 — Filtering:")
    print(f"    Removed short (<{min_words} words): {removed_short}")
    print(f"    Removed uninformative:              {removed_uninformative}")
    print(f"    Removed duplicates:                 {removed_dup}")
    print(f"    Unique concepts kept:               {len(kept)}")

    # ── Phase 4: Build clean output ──────────────────────────────────────────
    # Sort by frequency (most common first)
    kept_sorted = sorted(kept, key=lambda o: concept_counts[normalize_key(o)], reverse=True)

    output = {
        "metadata": {
            "source": str(input_path),
            "total_records": len(data),
            "total_raw_observations": len(all_obs),
            "truncated_fixed": n_fixed,
            "removed_short": removed_short,
            "removed_uninformative": removed_uninformative,
            "removed_duplicates": removed_dup,
            "unique_concepts": len(kept_sorted),
            "min_words": min_words,
        },
        "concepts": [
            {"text": obs, "count": concept_counts[normalize_key(obs)]}
            for obs in kept_sorted
        ],
    }

    # ── Phase 5: Per-record cleaned version ──────────────────────────────────
    # Also produce a cleaned version of the per-record data
    clean_key_set = set(seen.keys())
    cleaned_records = []
    for rec in data:
        clean_obs = []
        for obs in rec["observations"]:
            normed = normalize(obs)
            key = normalize_key(obs)
            if key in clean_key_set and len(normed.split()) >= min_words and not is_uninformative(normed):
                clean_obs.append(seen[key])  # use canonical form
        cleaned_records.append({
            "id": rec["id"],
            "dataset": rec["dataset"],
            "impression": rec["impression"],
            "concepts": clean_obs,
        })

    # ── Summary stats ────────────────────────────────────────────────────────
    top_20 = kept_sorted[:20]
    print(f"\n  Top 20 most frequent concepts:")
    for i, obs in enumerate(top_20, 1):
        cnt = concept_counts[normalize_key(obs)]
        print(f"    {i:2d}. [{cnt:>6d}x] {obs}")

    freq_dist = Counter()
    for key, cnt in concept_counts.items():
        if cnt == 1:
            freq_dist["1 (unique)"] += 1
        elif cnt <= 5:
            freq_dist["2-5"] += 1
        elif cnt <= 20:
            freq_dist["6-20"] += 1
        elif cnt <= 100:
            freq_dist["21-100"] += 1
        else:
            freq_dist["100+"] += 1

    print(f"\n  Frequency distribution:")
    for bucket in ["1 (unique)", "2-5", "6-20", "21-100", "100+"]:
        print(f"    {bucket:>12s}: {freq_dist.get(bucket, 0)}")

    # ── Write output ─────────────────────────────────────────────────────────
    if dry_run:
        print(f"\n  [DRY RUN] Would write to {output_path}")
        return

    records_path = Path(output_path).with_suffix(".records.json")

    print(f"\n  Writing concept vocabulary to {output_path} ...")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"    {size_mb:.1f} MB")

    print(f"  Writing cleaned records to {records_path} ...")
    with open(records_path, "w") as f:
        json.dump(cleaned_records, f)
    size_mb = records_path.stat().st_size / (1024 * 1024)
    print(f"    {size_mb:.1f} MB")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Clean and deduplicate extracted radiology concepts")
    parser.add_argument(
        "--input",
        default="concepts/concepts_both.json",
        help="Path to raw extraction output (default: concepts/concepts_both.json)",
    )
    parser.add_argument(
        "--output",
        default="concepts/concepts_clean.json",
        help="Path for clean concept vocabulary (default: concepts/concepts_clean.json)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=3,
        help="Minimum word count to keep an observation (default: 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview stats without writing output files",
    )
    args = parser.parse_args()
    clean_concepts(args.input, args.output, args.min_words, args.dry_run)


if __name__ == "__main__":
    main()
