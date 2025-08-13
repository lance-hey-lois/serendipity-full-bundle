
import argparse, json, random, os
from typing import List, Dict, Any
from engine.io import load_embeddings_csv
from engine.core import weights_for, apply_serendipity_scale, score
from engine.serendipity import make_bins, pick_bin, update_bin

def suggest_with_scale(self_state: Dict[str,Any], intent: str, pool: List[Dict[str,Any]], ser_scale: float, k=10):
    bins = make_bins(5)
    bi = pick_bin(bins); b = bins[bi]
    bucket = [c for c in pool if (c.get("novelty",0.0) >= b["lo"] and c.get("novelty",0.0) < b["hi"])]
    w = apply_serendipity_scale(weights_for(intent), ser_scale)
    ranked = sorted(({"c":c, "s": score(self_state, c, w)} for c in bucket), key=lambda x: x["s"], reverse=True)[:k]
    update_bin(b, 0.6 if ranked else 0.0)
    return [r["c"] for r in ranked]

def main():
    ap = argparse.ArgumentParser(description="Serendipity Engine with real embeddings")
    ap.add_argument("--embeddings", type=str, required=False, help="CSV with columns: id, vec (JSON) or vec_0..vec_d-1; optional availability, pathTrust, novelty")
    ap.add_argument("--intent", type=str, default="ship")
    ap.add_argument("--serendipity", type=float, default=1.0, help="scale factor for serendipity weight (e.g., 0.0..1.5)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=str, default="", help="seed person id")
    args = ap.parse_args()

    if args.embeddings and os.path.exists(args.embeddings):
        people = load_embeddings_csv(args.embeddings)
    else:
        # fallback: synthetic
        from engine.data_gen import make_people, pick_seed
        people = make_people(n=4000, dim=64, seed=123)

    by_id = {p["id"]: p for p in people if p["vec"]}
    seed_id = args.seed or next(iter(by_id.keys()))
    me = by_id[seed_id]
    pool = [p for p in people if p["id"] != seed_id and p["vec"]]

    top = suggest_with_scale(me, args.intent, pool, ser_scale=args.serendipity, k=args.k)
    print(json.dumps({"seed": seed_id, "intent": args.intent, "serendipity": args.serendipity, "top": top}, indent=2))

if __name__ == "__main__":
    main()
