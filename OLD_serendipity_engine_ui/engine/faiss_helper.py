# engine/faiss_helper.py
from typing import Tuple
import numpy as np

def have_faiss() -> bool:
    try:
        import faiss  # noqa
        return True
    except Exception:
        return False

def top_m_cosine(seed_unit: np.ndarray, cand_unit: np.ndarray, m: int) -> np.ndarray:
    """
    Return indices (int64) of the top-M candidates by cosine similarity.
    Uses FAISS IndexFlatIP if available; otherwise falls back to NumPy argpartition.
    Inputs must be L2-normalized.
    """
    m = int(min(m, cand_unit.shape[0]))
    if m <= 0:
        return np.empty((0,), dtype=np.int64)

    if have_faiss():
        import faiss
        d = cand_unit.shape[1]
        index = faiss.IndexFlatIP(d)           # inner product == cosine on unit vectors
        index.add(cand_unit.astype(np.float32))
        D, I = index.search(seed_unit.astype(np.float32)[None, :], m)  # (1,m)
        return I[0].astype(np.int64)

    # Fallback: pure NumPy
    sims = cand_unit @ seed_unit
    # partial selection is O(N)
    part = np.argpartition(-sims, m-1)[:m]
    # sort those m by score desc
    order = part[np.argsort(-sims[part])]
    return order.astype(np.int64)
