
from typing import List, Dict, Any
from .core import weights_for, score
from .serendipity import make_bins, pick_bin, update_bin

def suggest(self_state: Dict[str,Any], intent: str, pool: List[Dict[str,Any]], bins=None, k=10):
    if bins is None:
        bins = make_bins(5)
    # choose novelty bin via Thompson-like sampling
    bi = pick_bin(bins)
    b = bins[bi]
    # filter candidates by novelty bucket
    bucket = [c for c in pool if (c.get("novelty",0.0) >= b["lo"] and c.get("novelty",0.0) < b["hi"])]
    w = weights_for(intent)
    ranked = sorted(
        ({"c":c, "s": score(self_state, c, w)} for c in bucket),
        key=lambda x: x["s"],
        reverse=True
    )[:k]
    # fake reward update (in real app, use downstream signals)
    update_bin(b, 0.6 if ranked else 0.0)
    return [r["c"] for r in ranked]
