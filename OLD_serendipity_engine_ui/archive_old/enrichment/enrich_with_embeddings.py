#!/usr/bin/env python3
"""
Enrich profiles with quantum features derived from ACTUAL embeddings
Not keyword nonsense!
"""

import os
import sys
from pymongo import MongoClient
from dotenv import load_dotenv
from tqdm import tqdm
import time
from quantum_features_v2 import QuantumFeatureGeneratorV2, batch_fit_pca
import numpy as np

# Load environment variables
load_dotenv('../.env.dev')
if not os.getenv("MONGODB_URI"):
    load_dotenv('../.env')

def enrich_with_real_embeddings(limit: int = None, batch_size: int = 100):
    """
    Enrich profiles using their actual embeddings
    """
    # Initialize MongoDB
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        print("ERROR: MONGODB_URI not found")
        return
    
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = client[os.getenv("DB_NAME", "MagicCRM")]
    
    print("\n🚀 Quantum Feature Generation from REAL Embeddings")
    print("="*60)
    
    # First, fit PCA on a sample of embeddings
    print("\n📊 Step 1: Learning embedding patterns with PCA...")
    generator = batch_fit_pca(db, sample_size=500)
    
    # Get profiles with embeddings
    query = {"embedding": {"$exists": True}}
    total_count = db["public_profiles"].count_documents(query)
    
    if limit:
        total_count = min(total_count, limit)
    
    print(f"\n🔮 Step 2: Generating quantum features for {total_count} profiles...")
    
    cursor = db["public_profiles"].find(query)
    if limit:
        cursor = cursor.limit(limit)
    
    enriched_count = 0
    no_embedding_count = 0
    error_count = 0
    
    with tqdm(total=total_count, desc="Quantumizing") as pbar:
        batch = []
        
        for profile in cursor:
            batch.append(profile)
            
            if len(batch) >= batch_size:
                # Process batch
                for prof in batch:
                    try:
                        # Check if embedding exists and is valid
                        if not prof.get('embedding') or len(prof['embedding']) != 1536:
                            no_embedding_count += 1
                            pbar.update(1)
                            continue
                        
                        # Generate quantum features from REAL embedding
                        features = generator.generate_quantum_features(prof)
                        
                        # Update profile
                        db["public_profiles"].update_one(
                            {"_id": prof["_id"]},
                            {"$set": features}
                        )
                        
                        enriched_count += 1
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"\n⚠️ Error: {str(e)}")
                        error_count += 1
                        pbar.update(1)
                
                batch = []
                time.sleep(0.05)  # Small delay
        
        # Process remaining
        for prof in batch:
            try:
                if not prof.get('embedding') or len(prof['embedding']) != 1536:
                    no_embedding_count += 1
                    pbar.update(1)
                    continue
                    
                features = generator.generate_quantum_features(prof)
                db["public_profiles"].update_one(
                    {"_id": prof["_id"]},
                    {"$set": features}
                )
                enriched_count += 1
                pbar.update(1)
            except Exception as e:
                error_count += 1
                pbar.update(1)
    
    cursor.close()
    
    print("\n" + "="*60)
    print("✅ Quantum Feature Generation Complete!")
    print(f"   - Profiles enriched: {enriched_count}")
    print(f"   - No embedding: {no_embedding_count}")
    print(f"   - Errors: {error_count}")
    print(f"   - Success rate: {(enriched_count/total_count)*100:.1f}%")
    
    # Show sample
    sample = db["public_profiles"].find_one({
        "quantum_features.embedding_pca": True
    })
    
    if sample:
        print("\n📊 Sample quantum profile:")
        print(f"   Name: {sample.get('name', 'Unknown')}")
        qf = sample.get('quantum_features', {})
        print(f"   Using PCA: {qf.get('embedding_pca', False)}")
        print(f"   Quantum vector (first 3): {qf.get('skills_vector', [])[:3]}")
        print(f"   Network centrality: {sample.get('network_metrics', {}).get('centrality', 0):.2f}")
        print(f"   Uniqueness: {sample.get('serendipity_factors', {}).get('uniqueness', 0):.2f}")

def clear_old_quantum_features():
    """
    Clear the keyword-based quantum features
    """
    mongo_uri = os.getenv("MONGODB_URI")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = client[os.getenv("DB_NAME", "MagicCRM")]
    
    print("\n🧹 Clearing old keyword-based quantum features...")
    
    result = db["public_profiles"].update_many(
        {},
        {"$unset": {
            "quantum_features": "",
            "network_metrics": "",
            "serendipity_factors": ""
        }}
    )
    
    print(f"   Cleared {result.modified_count} profiles")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true", help="Clear old features first")
    parser.add_argument("--limit", type=int, help="Limit profiles to process")
    parser.add_argument("--batch-size", type=int, default=100)
    
    args = parser.parse_args()
    
    if args.clear:
        clear_old_quantum_features()
    
    enrich_with_real_embeddings(limit=args.limit, batch_size=args.batch_size)