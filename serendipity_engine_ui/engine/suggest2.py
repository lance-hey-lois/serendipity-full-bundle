
from typing import List, Dict, Any, Tuple
from .core import weights_for, apply_serendipity_scale, score

def score_pool(self_state: Dict[str,Any], intent: str, pool: List[Dict[str,Any]], ser_scale: float):
    w = apply_serendipity_scale(weights_for(intent), ser_scale)
    scored = [{"c": c, "score": score(self_state, c, w)} for c in pool]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored
