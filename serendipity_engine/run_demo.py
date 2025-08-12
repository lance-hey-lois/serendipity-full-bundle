
import argparse, json, random
from typing import List, Dict, Any
from engine.data_gen import make_people, pick_seed
from engine.suggest import suggest

def main():
    ap = argparse.ArgumentParser(description="Serendipity Engine Demo")
    ap.add_argument("--n", type=int, default=5000, help="number of synthetic people")
    ap.add_argument("--dim", type=int, default=64, help="embedding dimension")
    ap.add_argument("--seed", type=str, default="", help="seed person id (e.g., p00042)")
    ap.add_argument("--intent", type=str, default="ship", help="deal|friend|mentor|ship|collab")
    ap.add_argument("--k", type=int, default=10, help="top-k suggestions")
    args = ap.parse_args()

    people = make_people(n=args.n, dim=args.dim, seed=123)
    by_id = {p["id"]: p for p in people}

    seed_id = args.seed or pick_seed(people)
    me = by_id[seed_id]

    # pool excludes self
    pool = [p for p in people if p["id"] != seed_id]

    picks = suggest(me, args.intent, pool, bins=None, k=args.k)
    print(json.dumps({
        "seed": seed_id,
        "intent": args.intent,
        "top": picks
    }, indent=2))

if __name__ == "__main__":
    main()
