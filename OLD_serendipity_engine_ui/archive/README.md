# Archive

This folder contains deprecated or experimental code that is no longer in active use.

## Contents

### `/quantum/`
- `discovery_with_learned.py` - Early version of quantum discovery (v1)
- `debug_quantum.py` - Debugging utilities for quantum kernel

### `/enrichment/`
- `quantum_features.py` - Keyword-based feature extraction (deprecated - keywords are bad!)
- `quantum_features_v2.py` - PCA-based feature extraction (replaced by learned reducer)
- `enrich_mongo.py` - Old enrichment script using keywords
- `enrich_with_embeddings.py` - PCA enrichment (replaced by learned reducer)

### `/tests/`
- `test_query_search.py` - Query search with keyword filtering (deprecated)

## Active Code

The current active implementation is in:
- `/quantum_search.py` - Pure quantum search (no keywords!)
- `/quantum/discovery_with_learned_v2.py` - Main quantum discovery
- `/learning/learned_reducer.py` - Neural network for dimension reduction
- `/test_masseyl.py` - Test script for specific users