"""
Extract radiological observations (concepts) from CheXpert-Plus and ReXGradient
radiology reports using Qwen3-30B-A3B-Instruct-2507 served via vLLM OpenAI API.

Uses async concurrency to saturate the vLLM server for high throughput.

Usage:
    # Process both datasets (default, 64 concurrent requests)
    python concepts/get_concepts.py

    # Process single dataset
    python concepts/get_concepts.py --dataset chexpert
    python concepts/get_concepts.py --dataset rexgradient

    # Custom vLLM endpoint and concurrency
    python concepts/get_concepts.py --api-base http://localhost:8000/v1 --concurrency 128

    # Resume from checkpoint
    python concepts/get_concepts.py --resume
"""

import argparse
import asyncio
import json
import os
import time

import pandas as pd
from openai import AsyncOpenAI, OpenAI
from tqdm.asyncio import tqdm as atqdm


# ---------------------------------------------------------------------------
# Prompt template – few-shot extraction of radiological observations
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a radiology assistant. Extract discrete descriptive observations "
    "from the given radiology report impression. Return ONLY valid JSON."
)

FEW_SHOT_TEMPLATE = """\
Question: What are the descriptive observations in the report impression?

Example 1:
Impression: Mild pulmonary edema and small to moderate bilateral pleural effusions all improved since ___ following extubation. Heart size normal. No pneumothorax. Left subclavian line ends in the SVC.
Answer:
{"observations": ["Mild pulmonary edema", "Small to moderate bilateral pleural effusions", "Heart size normal", "No pneumothorax", "Left subclavian line ends in the SVC"]}

Example 2:
Impression: Patchy ill-defined left basilar opacity concerning for pneumonia. Small bilateral pleural effusions.
Answer:
{"observations": ["Patchy ill-defined left basilar opacity concerning for pneumonia", "Small bilateral pleural effusions"]}

Now extract observations from the following impression. Return ONLY a JSON object with an "observations" list.

Impression: {impression}
Answer:
"""


def build_prompt(impression: str) -> str:
    """Build the extraction prompt, safe against curly braces in impression text."""
    # Use replace instead of .format() to avoid KeyError on stray { } in reports
    return FEW_SHOT_TEMPLATE.replace("{impression}", impression)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_observations(raw: str) -> list:
    """Parse LLM output into a list of observation strings."""
    text = raw.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start:end])
            except json.JSONDecodeError:
                return []
        else:
            return []

    if isinstance(parsed, dict):
        return parsed.get("observations", [])
    return []


def load_impressions(dataset: str, data_dir: str) -> pd.DataFrame:
    """
    Load impressions from metadata CSVs.

    For cluster runs, reads from preprocessed metadata CSVs in the H5 output dir.
    For local dev, reads from data/ directory CSVs directly.

    Returns DataFrame with columns: [id, impression, dataset]
    """
    records = []  # list of DataFrames, concat at end

    if dataset in ("chexpert", "both"):
        cluster_path = os.path.join(data_dir, "h5", "chexpert_plus_train_metadata.csv")
        local_path = os.path.join(data_dir, "chexpert_train.csv")

        if os.path.exists(cluster_path):
            df = pd.read_csv(cluster_path)
            col = "impression"
        elif os.path.exists(local_path):
            df = pd.read_csv(local_path)
            col = "impression" if "impression" in df.columns else "section_impression"
        else:
            raise FileNotFoundError(
                f"CheXpert CSV not found at {cluster_path} or {local_path}"
            )

        df = df[df[col].notna()].copy()
        df[col] = df[col].astype(str).str.strip()
        df = df[df[col] != ""].reset_index(drop=True)
        chunk = pd.DataFrame({
            "id": [f"chexpert_{i}" for i in range(len(df))],
            "impression": df[col],
            "dataset": "chexpert",
        })
        records.append(chunk)

    if dataset in ("rexgradient", "both"):
        cluster_path = os.path.join(data_dir, "h5", "rexgradient_train_metadata.csv")
        local_path = os.path.join(data_dir, "rexgradient_all.csv")

        if os.path.exists(cluster_path):
            df = pd.read_csv(cluster_path)
        elif os.path.exists(local_path):
            df = pd.read_csv(local_path)
        else:
            raise FileNotFoundError(
                f"ReXGradient CSV not found at {cluster_path} or {local_path}"
            )

        col = "Impression" if "Impression" in df.columns else "impression"

        df = df[df[col].notna()].copy()
        df[col] = df[col].astype(str).str.strip()
        df = df[df[col] != ""].reset_index(drop=True)
        id_col = df["id"].astype(str) if "id" in df.columns else pd.Series(
            [f"rexgradient_{i}" for i in range(len(df))]
        )
        chunk = pd.DataFrame({
            "id": id_col.values,
            "impression": df[col],
            "dataset": "rexgradient",
        })
        records.append(chunk)

    if not records:
        return pd.DataFrame(columns=["id", "impression", "dataset"])
    return pd.concat(records, ignore_index=True)


# ---------------------------------------------------------------------------
# Async extraction engine
# ---------------------------------------------------------------------------

async def extract_one(
    client: AsyncOpenAI,
    model: str,
    row: dict,
    semaphore: asyncio.Semaphore,
    max_tokens: int,
) -> dict:
    """Extract concepts from a single impression with concurrency control."""
    async with semaphore:
        prompt = build_prompt(row["impression"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
            )
            raw_output = response.choices[0].message.content or ""
            observations = parse_observations(raw_output)
        except Exception as e:
            raw_output = f"ERROR: {e}"
            observations = []

        return {
            "id": row["id"],
            "dataset": row["dataset"],
            "impression": row["impression"],
            "raw_output": raw_output,
            "observations": observations,
            "n_observations": len(observations),
        }


async def extract_concepts_async(
    client: AsyncOpenAI,
    model: str,
    impressions_df: pd.DataFrame,
    output_path: str,
    concurrency: int = 64,
    checkpoint_every: int = 5000,
    max_tokens: int = 2048,
) -> list:
    """
    Extract concepts from all impressions using async concurrency.

    Fires up to `concurrency` requests in parallel against the vLLM server,
    with periodic checkpointing to disk.
    """
    checkpoint_path = output_path + ".checkpoint.json"

    # Load checkpoint or completed output for resume
    results = []
    processed_ids = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            results = json.load(f)
        processed_ids = {r["id"] for r in results}
        print(f"Resuming from checkpoint: {len(results)} already processed")
    elif os.path.exists(output_path):
        with open(output_path, "r") as f:
            results = json.load(f)
        processed_ids = {r["id"] for r in results}
        print(f"Resuming from completed output: {len(results)} already processed")

    remaining = impressions_df[~impressions_df["id"].isin(processed_ids)]
    rows = remaining.to_dict("records")
    print(f"Processing {len(rows)} impressions ({len(processed_ids)} already done)")
    print(f"Concurrency: {concurrency}")

    if not rows:
        return results

    semaphore = asyncio.Semaphore(concurrency)

    # Process in chunks for checkpointing
    for chunk_start in range(0, len(rows), checkpoint_every):
        chunk = rows[chunk_start : chunk_start + checkpoint_every]

        tasks = [
            extract_one(client, model, row, semaphore, max_tokens)
            for row in chunk
        ]

        chunk_results = []
        for coro in atqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"Chunk {chunk_start // checkpoint_every + 1}",
        ):
            result = await coro
            chunk_results.append(result)

        results.extend(chunk_results)

        # Checkpoint
        with open(checkpoint_path, "w") as f:
            json.dump(results, f)
        print(f"  Checkpoint saved: {len(results)} total records")

    # Final save with pretty formatting
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return results


def print_summary(results: list):
    """Print extraction summary statistics."""
    total = len(results)
    if total == 0:
        print("No results.")
        return

    n_with_obs = sum(1 for r in results if r["n_observations"] > 0)
    n_errors = sum(1 for r in results if (r.get("raw_output") or "").startswith("ERROR:"))
    total_obs = sum(r["n_observations"] for r in results)
    avg_obs = total_obs / n_with_obs if n_with_obs > 0 else 0

    print("\n" + "=" * 60)
    print("Concept Extraction Summary")
    print("=" * 60)
    print(f"  Total impressions processed: {total}")
    print(f"  Impressions with observations: {n_with_obs} ({100*n_with_obs/total:.1f}%)")
    print(f"  Errors: {n_errors}")
    print(f"  Total observations extracted: {total_obs}")
    print(f"  Avg observations per impression: {avg_obs:.1f}")

    datasets = set(r["dataset"] for r in results)
    for ds in sorted(datasets):
        ds_results = [r for r in results if r["dataset"] == ds]
        ds_total = len(ds_results)
        ds_obs = sum(r["n_observations"] for r in ds_results)
        print(f"  {ds}: {ds_total} impressions, {ds_obs} observations")

    print("=" * 60)


async def async_main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load impressions
    print(f"Loading impressions for dataset={args.dataset} from {args.data_dir}")
    impressions_df = load_impressions(args.dataset, args.data_dir)
    print(f"Loaded {len(impressions_df)} impressions")

    if len(impressions_df) == 0:
        print("No impressions found. Check your data directory and CSV files.")
        return

    # Setup async client pointing to vLLM
    client = AsyncOpenAI(
        base_url=args.api_base,
        api_key="EMPTY",
    )

    # Verify connection (sync check)
    try:
        sync_client = OpenAI(base_url=args.api_base, api_key="EMPTY")
        models = sync_client.models.list()
        available = [m.id for m in models.data]
        print(f"Connected to vLLM. Available models: {available}")
    except Exception as e:
        print(f"WARNING: Could not connect to vLLM at {args.api_base}: {e}")
        print("Make sure the vLLM server is running. Proceeding anyway...")

    output_path = os.path.join(args.output_dir, f"concepts_{args.dataset}.json")

    # Handle checkpoint / previous output
    checkpoint_path = output_path + ".checkpoint.json"
    if not args.resume:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        print("Starting fresh (use --resume to continue from previous run)")

    # Extract
    t0 = time.time()
    results = await extract_concepts_async(
        client=client,
        model=args.model,
        impressions_df=impressions_df,
        output_path=output_path,
        concurrency=args.concurrency,
        checkpoint_every=args.checkpoint_every,
        max_tokens=args.max_tokens,
    )
    elapsed = time.time() - t0

    print_summary(results)
    print(f"\nSaved to {output_path}")
    if results:
        print(f"Total time: {elapsed/60:.1f}min ({elapsed/len(results):.3f}s per impression)")
        print(f"Throughput: {len(results)/elapsed:.1f} impressions/sec")


def main():
    parser = argparse.ArgumentParser(
        description="Extract radiological concepts from CXR reports using Qwen3 via vLLM"
    )
    parser.add_argument(
        "--dataset",
        choices=["chexpert", "rexgradient", "both"],
        default="both",
        help="Which dataset(s) to process (default: both)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing CSV files (default: data/)",
    )
    parser.add_argument(
        "--output-dir",
        default="concepts",
        help="Output directory for results (default: concepts/)",
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Model name as served by vLLM",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max generation tokens (default: 2048)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Max concurrent requests to vLLM (default: 16)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5000,
        help="Save checkpoint every N records (default: 5000)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint",
    )
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
