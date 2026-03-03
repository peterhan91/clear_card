#!/usr/bin/env python3
"""
Generate concept embeddings using OpenAI text-embedding-3-large via the Batch API.

The Batch API gives 50% cost savings (~$0.50 for all 492K concepts).

Pipeline:
  1. Load concepts from CSV (with concept_idx)
  2. Split into chunks of ≤50,000 (Batch API limit)
  3. For each chunk: build JSONL → upload → create batch → poll → download
  4. Merge all embeddings into a single pickle (concept_idx -> np.ndarray)
  5. Output is compatible with exp_zeroshot.py

Usage:
  # First, create .env with: OPENAI_API_KEY=sk-...
  python get_embed_openai_batch.py --submit          # submit batches
  python get_embed_openai_batch.py --poll             # check status & download when ready
  python get_embed_openai_batch.py --submit --poll    # submit + wait for completion

  # Optional: reduce dimensions (Matryoshka)
  python get_embed_openai_batch.py --submit --poll --dimensions 1536
"""

import os
import sys
import json
import time
import pickle
import argparse
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = "text-embedding-3-large"
FULL_DIM = 3072
BATCH_CHUNK_SIZE = 50_000  # max inputs per batch job
POLL_INTERVAL = 60  # seconds
MAX_POLL_TIME = 24 * 3600  # 24 hours

CONCEPTS_CSV = os.path.join(os.path.dirname(__file__), 'concepts_clean.csv')
EMBEDDINGS_DIR = os.path.join(os.path.dirname(__file__), 'embeddings_output')
STATE_FILE = os.path.join(os.path.dirname(__file__), 'embeddings_output', 'openai_batch_state.json')


def load_env():
    """Load OPENAI_API_KEY from .env file."""
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    os.environ.setdefault(key.strip(), val.strip())

    if not os.environ.get('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY not found. Create .env file with: OPENAI_API_KEY=sk-...")
        sys.exit(1)


def get_client():
    """Create OpenAI client."""
    from openai import OpenAI
    return OpenAI()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_concepts(csv_path: str) -> pd.DataFrame:
    """Load concepts CSV with concept_idx and concept columns."""
    df = pd.read_csv(csv_path)
    assert 'concept_idx' in df.columns and 'concept' in df.columns, \
        "CSV must have 'concept_idx' and 'concept' columns"
    logger.info(f"Loaded {len(df)} concepts from {csv_path}")
    return df


# ---------------------------------------------------------------------------
# JSONL creation
# ---------------------------------------------------------------------------
def build_jsonl_bytes(concept_indices: List[int], concept_texts: List[str],
                      dimensions: Optional[int] = None) -> bytes:
    """Build JSONL payload for a chunk of concepts.

    Each line: {"custom_id": "<idx>", "method": "POST", "url": "/v1/embeddings",
                "body": {"model": "...", "input": "...", "encoding_format": "float"}}
    """
    lines = []
    for idx, text in zip(concept_indices, concept_texts):
        body = {
            "model": MODEL,
            "input": text,
            "encoding_format": "float",
        }
        if dimensions is not None:
            body["dimensions"] = dimensions

        record = {
            "custom_id": str(idx),  # concept_idx as string
            "method": "POST",
            "url": "/v1/embeddings",
            "body": body,
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    return "\n".join(lines).encode("utf-8")


# ---------------------------------------------------------------------------
# Batch submission
# ---------------------------------------------------------------------------
def submit_batches(df: pd.DataFrame, dimensions: Optional[int] = None) -> List[dict]:
    """Split concepts into chunks, submit each as a batch job.

    Returns list of chunk metadata dicts (saved to state file for polling).
    """
    client = get_client()
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    indices = df['concept_idx'].tolist()
    texts = df['concept'].tolist()
    n_chunks = (len(texts) + BATCH_CHUNK_SIZE - 1) // BATCH_CHUNK_SIZE

    logger.info(f"Submitting {len(texts)} concepts in {n_chunks} batch(es) of ≤{BATCH_CHUNK_SIZE}")

    dim = dimensions or FULL_DIM
    chunks_state = []

    for chunk_i in range(n_chunks):
        start = chunk_i * BATCH_CHUNK_SIZE
        end = min(start + BATCH_CHUNK_SIZE, len(texts))
        chunk_indices = indices[start:end]
        chunk_texts = texts[start:end]

        logger.info(f"Chunk {chunk_i + 1}/{n_chunks}: concepts {start}-{end - 1} ({len(chunk_texts)} concepts)")

        # Build JSONL
        payload = build_jsonl_bytes(chunk_indices, chunk_texts, dimensions)
        logger.info(f"  JSONL size: {len(payload) / 1024 / 1024:.1f} MB")

        # Upload file
        filename = f"openai_embed_chunk_{chunk_i}.jsonl"
        file_obj = client.files.create(
            file=(filename, BytesIO(payload), "application/jsonl"),
            purpose="batch",
        )
        logger.info(f"  Uploaded file: {file_obj.id}")

        # Create batch
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
            metadata={"description": f"CXR concept embeddings chunk {chunk_i + 1}/{n_chunks}"},
        )
        logger.info(f"  Batch created: {batch.id} (status: {batch.status})")

        chunks_state.append({
            "chunk_index": chunk_i,
            "batch_id": batch.id,
            "input_file_id": file_obj.id,
            "n_concepts": len(chunk_texts),
            "start_idx": start,
            "end_idx": end,
            "status": batch.status,
            "dimensions": dim,
        })

    # Save state for polling later
    state = {
        "model": MODEL,
        "dimensions": dim,
        "total_concepts": len(texts),
        "n_chunks": n_chunks,
        "chunks": chunks_state,
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)
    logger.info(f"Batch state saved to {STATE_FILE}")

    return chunks_state


# ---------------------------------------------------------------------------
# Polling & downloading
# ---------------------------------------------------------------------------
def poll_and_download() -> Optional[Dict[int, np.ndarray]]:
    """Poll all batch jobs. Once all complete, download and merge results.

    Returns merged embeddings dict (concept_idx -> np.ndarray) or None if not ready.
    """
    if not os.path.exists(STATE_FILE):
        logger.error(f"No state file found at {STATE_FILE}. Run with --submit first.")
        return None

    with open(STATE_FILE) as f:
        state = json.load(f)

    client = get_client()
    chunks = state["chunks"]
    dim = state["dimensions"]
    terminal_states = {"completed", "failed", "expired", "cancelled"}

    elapsed = 0
    while elapsed < MAX_POLL_TIME:
        all_done = True
        for chunk in chunks:
            if chunk["status"] in terminal_states:
                continue

            batch = client.batches.retrieve(chunk["batch_id"])
            chunk["status"] = batch.status

            req = batch.request_counts
            done = req.completed if req else "?"
            failed = req.failed if req else "?"
            total = req.total if req else "?"

            logger.info(
                f"  Chunk {chunk['chunk_index'] + 1}: status={batch.status}  "
                f"completed={done}  failed={failed}  total={total}"
            )

            if batch.status not in terminal_states:
                all_done = False
            else:
                # Save output/error file IDs
                if batch.output_file_id:
                    chunk["output_file_id"] = batch.output_file_id
                if batch.error_file_id:
                    chunk["error_file_id"] = batch.error_file_id

        # Update state file
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)

        if all_done:
            break

        logger.info(f"  Waiting {POLL_INTERVAL}s... (elapsed: {elapsed}s)")
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    # Check final status
    for chunk in chunks:
        if chunk["status"] != "completed":
            logger.error(f"Chunk {chunk['chunk_index'] + 1} ended with status: {chunk['status']}")

    # Download results
    all_embeddings: Dict[int, np.ndarray] = {}
    total_errors = 0

    for chunk in chunks:
        if chunk.get("output_file_id"):
            logger.info(f"Downloading results for chunk {chunk['chunk_index'] + 1}...")
            raw = client.files.content(chunk["output_file_id"]).content
            for line in raw.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                cid = int(record["custom_id"])  # concept_idx
                resp = record.get("response", {})

                if resp and resp.get("status_code") == 200:
                    vec = resp["body"]["data"][0]["embedding"]
                    all_embeddings[cid] = np.array(vec, dtype=np.float32)
                else:
                    logger.warning(f"  concept_idx {cid} failed: {resp}")
                    total_errors += 1

        # Check error file
        if chunk.get("error_file_id"):
            err_raw = client.files.content(chunk["error_file_id"]).content
            for line in err_raw.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                logger.error(f"  Error for concept_idx {record['custom_id']}: {record.get('error')}")
                total_errors += 1

    logger.info(f"Downloaded {len(all_embeddings)} embeddings, {total_errors} errors")

    if not all_embeddings:
        logger.error("No embeddings retrieved!")
        return None

    # Validate
    sample_key = next(iter(all_embeddings))
    sample_dim = all_embeddings[sample_key].shape[0]
    logger.info(f"Embedding dimension: {sample_dim} (expected: {dim})")

    dims_ok = all(v.shape[0] == sample_dim for v in all_embeddings.values())
    if not dims_ok:
        logger.warning("Dimension mismatch found in some embeddings!")

    zeros = sum(1 for v in all_embeddings.values() if np.allclose(v, 0))
    if zeros:
        logger.warning(f"{zeros} zero-vector embeddings found")

    return all_embeddings


def save_embeddings(embeddings: Dict[int, np.ndarray], dimensions: int):
    """Save embeddings in the same format as other models (concept_idx -> np.ndarray pickle)."""
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    suffix = f"openai_3large_{dimensions}d"
    out_path = os.path.join(EMBEDDINGS_DIR, f"cxr_embeddings_{suffix}.pickle")
    with open(out_path, 'wb') as f:
        pickle.dump(embeddings, f)
    logger.info(f"Saved {len(embeddings)} embeddings to {out_path}")

    # Also save indexed version for compatibility
    indexed_path = os.path.join(EMBEDDINGS_DIR, f"intermediate_{suffix}_indexed.pickle")
    with open(indexed_path, 'wb') as f:
        pickle.dump(embeddings, f)
    logger.info(f"Saved indexed copy to {indexed_path}")

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate concept embeddings using OpenAI text-embedding-3-large Batch API")

    parser.add_argument('--submit', action='store_true',
                        help='Submit batch jobs to OpenAI')
    parser.add_argument('--poll', action='store_true',
                        help='Poll for completion and download results')
    parser.add_argument('--concepts_file', type=str, default=CONCEPTS_CSV,
                        help='Path to concepts CSV file')
    parser.add_argument('--dimensions', type=int, default=None,
                        help=f'Embedding dimensions (default: {FULL_DIM}, or e.g. 1536 for Matryoshka reduction)')

    args = parser.parse_args()

    if not args.submit and not args.poll:
        parser.error("Specify --submit, --poll, or both")

    load_env()

    dim = args.dimensions or FULL_DIM

    if args.submit:
        df = load_concepts(args.concepts_file)
        submit_batches(df, dimensions=args.dimensions)

    if args.poll:
        embeddings = poll_and_download()
        if embeddings:
            out_path = save_embeddings(embeddings, dim)

            # Print summary
            logger.info("=" * 60)
            logger.info(f"DONE: {len(embeddings)} embeddings saved")
            logger.info(f"  Model: {MODEL}")
            logger.info(f"  Dimensions: {dim}")
            logger.info(f"  Output: {out_path}")
            logger.info("=" * 60)


if __name__ == "__main__":
    main()
