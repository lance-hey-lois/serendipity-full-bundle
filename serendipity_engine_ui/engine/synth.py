# serendipity_engine_ui/engine/synth.py
from typing import List, Dict
import numpy as np

def make_synthetic(n: int = 5000, d: int = 64, k: int = 8, seed: int = 123) -> List[Dict]:
    """Fast, vectorized synthetic dataset (people with embeddings + features)."""
    rng = np.random.default_rng(seed)

    # k cluster centers on the unit sphere
    centers = rng.normal(size=(k, d))
    centers /= (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-9)

    # assign each row to a cluster and add noise
    assign = rng.integers(0, k, size=n)
    vecs = centers[assign] + 0.25 * rng.normal(size=(n, d))
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)

    # novelty = distance to second-nearest center (normalized)
    dists = np.stack([np.linalg.norm(vecs - c, axis=1) for c in centers], axis=1)
    raw = np.sort(dists, axis=1)[:, 1]
    nov = (raw - raw.min()) / (np.ptp(raw) + 1e-9)

    # availability and pathTrust
    avail = np.clip(rng.normal(0.6, 0.25, size=n), 0, 1)
    trust = np.clip(rng.normal(0.5, 0.25, size=n), 0, 1)

    ids = [f"s{i:05d}" for i in range(n)]
    people = [
        {
            "id": ids[i],
            "vec": vecs[i].astype(float).tolist(),
            "novelty": float(nov[i]),
            "availability": float(avail[i]),
            "pathTrust": float(trust[i]),
        }
        for i in range(n)
    ]
    return people
