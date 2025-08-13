
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any

def _parse_vec(row, dim_hint=None):
    # Accept either a JSON array in 'vec' or multiple columns vec_0..vec_{d-1}
    if "vec" in row and row["vec"]:
        try:
            v = row["vec"]
            if isinstance(v, str):
                v = json.loads(v)
            return [float(x) for x in v]
        except Exception:
            pass
    # try wide format
    keys = [k for k in row.keys() if k.startswith("vec_")]
    if keys:
        return [float(row[k]) for k in sorted(keys, key=lambda x:int(x.split("_")[1]))]
    return []

def load_embeddings_csv(path: str, id_col="id") -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    out = []
    for _, r in df.iterrows():
        vec = _parse_vec(r)
        d = {
            "id": str(r[id_col]),
            "vec": vec,
            "availability": float(r.get("availability", np.clip(np.random.normal(0.6,0.25),0,1))),
            "pathTrust": float(r.get("pathTrust", np.clip(np.random.normal(0.5,0.25),0,1))),
        }
        # novelty optional; if present use it, else derive a crude proxy by L2 norm from mean
        d["novelty"] = float(r.get("novelty", np.nan))
        out.append(d)
    # Compute novelty if missing
    if any(np.isnan(x["novelty"]) for x in out):
        # mean vector
        mat = np.array([x["vec"] for x in out if x["vec"]])
        mu = mat.mean(axis=0) if len(mat) else None
        for x in out:
            if np.isnan(x["novelty"]):
                if x["vec"] and mu is not None:
                    x["novelty"] = float(np.clip(np.linalg.norm(np.array(x["vec"])-mu)/ (np.linalg.norm(mu)+1e-9), 0, 1))
                else:
                    x["novelty"] = float(np.clip(np.random.rand(),0,1))
    return out
