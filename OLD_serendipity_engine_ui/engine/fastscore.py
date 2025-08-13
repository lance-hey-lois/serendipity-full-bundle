# engine/fastscore.py
from typing import Dict, List, Tuple
import numpy as np
from .core import weights_for, apply_serendipity_scale

def _to_arrays(people: List[Dict]) -> Dict[str, np.ndarray]:
    ids = [p["id"] for p in people]
    vecs = np.array([p["vec"] for p in people], dtype=float)
    nov  = np.array([float(p.get("novelty", 0.5)) for p in people], dtype=float)
    avail= np.array([float(p.get("availability", 0.5)) for p in people], dtype=float)
    trust= np.array([float(p.get("pathTrust", 0.5)) for p in people], dtype=float)
    # keep original vectors for quantum/GBS; also store normalized copies for cosine
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    vecs_norm = vecs / norms
    return {
        "ids": np.array(ids, dtype=object),
        "vecs": vecs,               # original
        "vecs_norm": vecs_norm,     # unit vectors
        "novelty": nov,
        "availability": avail,
        "trust": trust,
    }

def prepare_arrays(seed: Dict, pool: List[Dict]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    seed_vec = np.array(seed["vec"], dtype=float)
    seed_norm = np.linalg.norm(seed_vec) + 1e-9
    seed_unit = seed_vec / seed_norm
    arrs = _to_arrays(pool)
    return seed_unit, arrs

def score_vectorized(
    seed_unit: np.ndarray,
    arrs: Dict[str, np.ndarray],
    intent: str,
    ser_scale: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return classical scores for the entire pool (vectorized)."""
    if rng is None:
        rng = np.random.default_rng()
    w = apply_serendipity_scale(weights_for(intent), ser_scale)

    # cosine: since both are unit, cos = dot
    cos_sim = arrs["vecs_norm"] @ seed_unit  # shape (N,)

    base = (
        w["fit"]  * np.maximum(0.0, cos_sim) +
        w["trust"]* arrs["trust"] +
        w["avail"]* arrs["availability"] +
        w["div"]  * arrs["novelty"]
    )
    # controlled randomness (same formula as before, but vectorized)
    ser_noise = rng.random(size=base.shape)
    ser = w["ser"] * (0.5*arrs["novelty"] + 0.5*ser_noise)

    return base + ser  # shape (N,)
