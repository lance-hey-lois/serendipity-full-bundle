
import argparse, math, random, numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from engine.data_gen import make_people
from engine.core import weights_for, apply_serendipity_scale, score
from engine.serendipity import make_bins, pick_bin, update_bin

def simulate(n_users=200, ser_scale=1.0, intent="ship", k=5, steps=200):
    people = make_people(n=5000, dim=64, seed=42)
    bins = make_bins(5)
    accept_counts = [0]*len(bins)
    show_counts = [0]*len(bins)

    def bucket_idx(nv):
        for i,b in enumerate(bins):
            if nv >= b["lo"] and nv < b["hi"]:
                return i
        return len(bins)-1

    for _ in range(steps):
        me = random.choice(people)
        pool = [p for p in people if p["id"] != me["id"]]
        bi = pick_bin(bins); b = bins[bi]
        # serendipity-scaled scoring
        w = apply_serendipity_scale(weights_for(intent), ser_scale)
        bucket = [c for c in pool if c["novelty"] >= b["lo"] and c["novelty"] < b["hi"]]
        ranked = sorted(({"c":c, "s": score(me, c, w)} for c in bucket), key=lambda x:x["s"], reverse=True)[:k]

        # Simulate a reward: higher for better fit and availability
        reward = 0.0
        for r in ranked:
            fit = r["s"]
            # crude reward mapping
            reward += max(0.0, min(1.0, fit))
            show_counts[bi] += 1
            # accept with prob proportional to availability and trust
            prob = 0.3*c["availability"] + 0.3*c["pathTrust"] + 0.4*random.random()
        # aggregate reward likelihood to binary accept for the bin
        # normalize by k
        avg_r = (reward / max(1,k))
        # convert to [0,1]
        accept = 1 if avg_r > 0.75 else 0
        accept_counts[bi] += accept
        update_bin(b, accept)

    rates = [ (accept_counts[i] / show_counts[i]) if show_counts[i] else 0.0 for i in range(len(bins)) ]
    return bins, rates, show_counts, accept_counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serendipity", type=float, default=1.0)
    ap.add_argument("--intent", type=str, default="ship")
    args = ap.parse_args()

    bins, rates, shows, accepts = simulate(ser_scale=args.serendipity, intent=args.intent)

    labels = [f"{b['lo']:.1f}-{b['hi']:.1f}" for b in bins]
    plt.figure()
    plt.bar(labels, rates)
    plt.title("Accept rate by novelty bin")
    plt.xlabel("Novelty bin")
    plt.ylabel("Accept rate")
    plt.tight_layout()
    out = "metrics_accept_rate.png"
    plt.savefig(out)
    print(out)

if __name__ == "__main__":
    main()
