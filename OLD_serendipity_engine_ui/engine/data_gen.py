
import random, math
from typing import List, Dict, Any
import numpy as np

def _unit(x):
    n = np.linalg.norm(x)
    return (x / n) if n else x

def make_people(n=5000, dim=64, seed=42):
    rng = np.random.default_rng(seed)
    # Create some latent "communities" as cluster centers
    k = max(4, int(math.sqrt(n)//2))
    centers = _unit(rng.normal(size=(k, dim)))
    people = []
    for i in range(n):
        c = centers[i % k] + 0.2 * rng.normal(size=(dim,))
        vec = _unit(c + 0.05 * rng.normal(size=(dim,)))
        # novelty ~ distance to person's usual "comfort cluster" (random here)
        novelty = float(np.clip(np.linalg.norm(vec - centers[(i+1)%k]), 0, 2) / 2.0)
        availability = float(np.clip(rng.normal(0.6, 0.25), 0, 1))
        path_trust = float(np.clip(rng.normal(0.5, 0.25), 0, 1))
        people.append({
            "id": f"p{i:05d}",
            "vec": vec.astype(float).tolist(),
            "novelty": novelty,
            "availability": availability,
            "pathTrust": path_trust
        })
    return people

def pick_seed(people):
    # choose a median-ish availability user as seed
    people_sorted = sorted(people, key=lambda x: x["availability"])
    return people_sorted[len(people_sorted)//2]["id"]
