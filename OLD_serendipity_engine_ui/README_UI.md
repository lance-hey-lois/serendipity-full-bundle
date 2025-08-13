
# Serendipity Engine — Streamlit UI + Quantum Tie-Breaker

Run a live UI with a **serendipity slider** and an optional **quantum kernel** tie-breaker (PennyLane).

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then in the browser:
1. (Optional) Upload `sample_embeddings.csv` or your own CSV.
2. Pick an **intent**, adjust **Serendipity**, **Top-K**, **Quantum weight γ**, and **wires** (PCA dimension).
3. Click **Run**. You’ll get a ranked table with classical score + quantum kernel tie-breaker.

### CSV formats
- **Wide:** `id, vec_0, vec_1, ..., vec_D-1, availability?, pathTrust?, novelty?`
- **Compact:** `id, vec` where `vec` is a JSON array string, plus optional columns.

### How the quantum piece works
- We **compress** seed + candidates to a small dimension (2–8) with PCA (`engine/quantum.pca_compress`).
- We encode them into a **feature-map circuit** and compute a **kernel overlap** with PennyLane.
- Final ranking uses `zscore(classical_score) + γ * zscore(quantum_kernel_to_seed)`.

If PennyLane is not installed or a device is unavailable, the quantum score defaults to zeros.

### Files
- `app.py` — Streamlit UI.
- `engine/quantum.py` — PCA + PennyLane kernel (fallback-safe).
- `engine/suggest2.py` — returns classical scores for the pool.
- `engine/` — core logic from the prototype.
- `sample_embeddings.csv` — 300×32 synthetic vectors to test.

Enjoy turning the **serendipity** and **quantum** dials. 🔮
