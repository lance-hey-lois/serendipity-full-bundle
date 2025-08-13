
# Serendipity Engine — Minimal Prototype (Quantum-Ready)

This is a small, **runnable Python demo** of the serendipity engine we discussed.
It generates a synthetic population, scores candidates with a multi-objective utility,
and injects **controlled randomness** via a bandit over novelty buckets. There's also a
**quantum-programming example** (PennyLane-style) showing how to compute a tiny quantum
kernel for tie-breaking among top-K candidates.

## Files
- `run_demo.py` — CLI runner
- `engine/core.py` — cosine + multi-objective scoring
- `engine/serendipity.py` — Thompson-like bandit over novelty
- `engine/suggest.py` — orchestrates one-shot suggestions
- `engine/data_gen.py` — synthetic data generator
- `quantum_demo.py` — PennyLane-style quantum kernel tie-breaker (not executed here)

## Quick Start (local)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pandas pennylane  # pennylane only needed for quantum_demo.py
python run_demo.py --intent ship --k 10
```

Sample output:
```json
{
  "seed": "p02500",
  "intent": "ship",
  "top": [ { "id": "...", "novelty": 0.82, "availability": 0.61, "pathTrust": 0.44, ... }, ... ]
}
```

## Quantum tie-breaker (optional)
The `quantum_demo.py` file demonstrates a simple **feature-map kernel** with PennyLane.
You'd feed a **small batch** of PCA-compressed candidate embeddings (e.g., 4–8 dims)
and get a kernel matrix that can help break ties in the final ranking.

## Notes
- This demo uses **synthetic** data; wire your real embeddings and path-trust signals later.
- Serendipity is controlled via the bandit and the "ser" weight in `core.weights_for()`.
- Keep quantum as a **bolt-on** on small ambiguous slices; classical remains the backbone.
