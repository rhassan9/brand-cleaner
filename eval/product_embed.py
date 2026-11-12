"""
Approach B: sentence embeddings with sentence-transformers, run locally on CPU.

Model: sentence-transformers/all-MiniLM-L6-v2 (Apache-2.0), pinned to a fixed
revision so the numbers are reproducible. Embeddings are L2-normalised, so the
dot product is the cosine similarity. They are cached under data/public/
(gitignored) keyed by model, revision and a hash of the input texts.
"""
import hashlib
import os
from typing import List

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Commit on the Hugging Face Hub. Change it only together with the reported results.
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO_ROOT, "data", "public", "embeddings")


def _cache_path(texts: List[str]) -> str:
    h = hashlib.sha256()
    h.update(f"{MODEL_NAME}@{MODEL_REVISION}\n".encode())
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return os.path.join(CACHE_DIR, f"minilm_{h.hexdigest()[:16]}.npy")


def embed(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Normalised embeddings, shape (len(texts), 384), float32."""
    path = _cache_path(texts)
    if os.path.exists(path):
        return np.load(path)
    from sentence_transformers import SentenceTransformer
    import torch
    torch.manual_seed(0)
    model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device="cpu")
    vecs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(path, vecs)
    return vecs


def cosine_matrix(queries: List[str], pool: List[str]) -> np.ndarray:
    """Cosine similarity of every query title to every pool title."""
    q = embed(queries)
    p = embed(pool)
    return q @ p.T
