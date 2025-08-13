
import math
import random
from typing import List, Dict, Any, Tuple

def cos(a, b):
    # cosine similarity with simple guards
    if not a or not b: return 0.0
    s = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return (s / (na*nb)) if (na and nb) else 0.0

def weights_for(intent: str) -> Dict[str, float]:
    # Intent presets (tweak to taste)
    intent = intent.lower()
    if intent == "deal":
        return {"fit":0.35, "trust":0.35, "avail":0.20, "div":0.05, "ser":0.05}
    if intent == "ship":
        return {"fit":0.25, "trust":0.10, "avail":0.25, "div":0.10, "ser":0.30}
    if intent == "friend":
        return {"fit":0.40, "trust":0.15, "avail":0.20, "div":0.10, "ser":0.15}
    if intent == "mentor":
        return {"fit":0.30, "trust":0.30, "avail":0.15, "div":0.10, "ser":0.15}
    # default
    return {"fit":0.35, "trust":0.20, "avail":0.20, "div":0.10, "ser":0.15}

def score(me, c, w):
    # Multi-objective + controlled randomness (serendipity)
    fit  = max(0.0, cos(me["vec"], c["vec"]))
    base = (w["fit"]*fit
          + w["trust"]*(c.get("pathTrust",0.0) or 0.0)
          + w["avail"]*(c.get("availability",0.0) or 0.0)
          + w["div"]*(c.get("novelty",0.0) or 0.0))
    ser  = w["ser"]*(0.5*(c.get("novelty",0.0) or 0.0) + 0.5*random.random())
    return base + ser


def apply_serendipity_scale(w: dict, scale: float) -> dict:
    # scale serendipity weight by factor; clamp to reasonable bounds
    s = w.copy()
    s["ser"] = max(0.0, min(1.5, s["ser"] * scale))
    return s
