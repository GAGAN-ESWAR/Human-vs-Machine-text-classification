"""
data_loader.py — JSON Lines streaming / token sequence loading.

Each file is stored as JSON Lines (one JSON object per line).
Train records: {"id": int, "text": [int, ...], "label": "A"|"B"}
Test records:  {"id": int, "text": [int, ...]}
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional


def load_jsonl(path: str) -> List[dict]:
    """Load all records from a JSON Lines file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_train(path: str = "train.json") -> Tuple[np.ndarray, List[List[int]], np.ndarray]:
    """
    Load training data.

    Returns
    -------
    ids    : np.ndarray of shape (N,)  — document IDs
    texts  : list of N token-ID lists
    labels : np.ndarray of shape (N,)  — 0 for 'A' (human), 1 for 'B' (machine)
    """
    records = load_jsonl(path)
    ids = np.array([r["id"] for r in records], dtype=np.int64)
    texts = [r["text"] for r in records]
    label_map = {"A": 0, "B": 1}
    labels = np.array([label_map[r["label"]] for r in records], dtype=np.int32)
    return ids, texts, labels


def load_test(path: str = "test.json") -> Tuple[np.ndarray, List[List[int]]]:
    """
    Load test data (no labels).

    Returns
    -------
    ids   : np.ndarray of shape (N,)  — document IDs
    texts : list of N token-ID lists
    """
    records = load_jsonl(path)
    ids = np.array([r["id"] for r in records], dtype=np.int64)
    texts = [r["text"] for r in records]
    return ids, texts


def load_sample_submission(path: str = "sample_submission.csv") -> np.ndarray:
    """
    Load the sample submission CSV and return the ordered IDs as int64 array.
    Used for ID-order validation before writing a submission.
    """
    import csv
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(row["id"]))
    return np.array(ids, dtype=np.int64)


def labels_to_str(labels: np.ndarray) -> List[str]:
    """Convert integer labels (0/1) back to string labels ('A'/'B')."""
    return ["A" if l == 0 else "B" for l in labels]
