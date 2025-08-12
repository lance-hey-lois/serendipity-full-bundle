# Serendipity Engine — Full Bundle

This bundle includes:
- `/serendipity_engine` — CLI engine + synthetic generator + quantum demo
- `/serendipity_engine_ui` — Streamlit UI with quantum tie-breaker
- Quick start:
  ```bash
  cd serendipity_engine_ui
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  streamlit run app.py
  ```
  or run the CLI:
  ```bash
  cd serendipity_engine
  python run_demo.py --intent ship --k 10
  python run_real.py --embeddings ../serendipity_engine_ui/sample_embeddings.csv --intent deal --serendipity 1.0 --k 20
  ```
