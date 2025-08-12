#!/usr/bin/env python3
"""
MongoDB Profile Enrichment Script
Adds quantum features to all profiles for serendipitous discovery
"""

import os
import sys
from pymongo import MongoClient
from dotenv import load_dotenv
from tqdm import tqdm
import time
from quantum_features import QuantumFeatureGenerator
from openai import OpenAI

# Load environment variables
load_dotenv('../.env.dev')
if not os.getenv("MONGODB_URI"):
    load_dotenv('../.env')

def enrich_all_profiles(limit: int = None, batch_size: int = 100):
    """
    Enrich all profiles in MongoDB with quantum features
    
    Args:
        limit: Maximum number of profiles to process (None for all)
        batch_size: Number of profiles to process in each batch
    """
    # Initialize MongoDB
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        print("ERROR: MONGODB_URI not found in environment")
        return
    
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = client[os.getenv("DB_NAME", "MagicCRM")]
    
    # Initialize quantum feature generator
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    generator = QuantumFeatureGenerator(openai_client)
    
    # Get profiles to enrich
    query = {}  # You can add filters here if needed
    
    # Count total profiles
    total_count = db["public_profiles"].count_documents(query)
    if limit:
        total_count = min(total_count, limit)
    
    print(f"\n🚀 Enriching {total_count} profiles with quantum features...")
    print("="*60)
    
    # Process profiles in batches
    enriched_count = 0
    error_count = 0
    
    cursor = db["public_profiles"].find(query, no_cursor_timeout=True)
    if limit:
        cursor = cursor.limit(limit)
    
    with tqdm(total=total_count, desc="Enriching profiles") as pbar:
        batch = []
        for profile in cursor:
            batch.append(profile)
            
            if len(batch) >= batch_size:
                # Process batch
                for prof in batch:
                    try:
                        # Generate quantum features
                        features = generator.generate_quantum_features(prof)
                        
                        # Update profile in database
                        db["public_profiles"].update_one(
                            {"_id": prof["_id"]},
                            {"$set": features}
                        )
                        
                        enriched_count += 1
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"\n⚠️ Error enriching profile {prof.get('name', 'Unknown')}: {str(e)}")
                        error_count += 1
                        pbar.update(1)
                
                batch = []
                time.sleep(0.1)  # Small delay to avoid overwhelming the system
        
        # Process remaining profiles in batch
        for prof in batch:
            try:
                features = generator.generate_quantum_features(prof)
                db["public_profiles"].update_one(
                    {"_id": prof["_id"]},
                    {"$set": features}
                )
                enriched_count += 1
                pbar.update(1)
            except Exception as e:
                print(f"\n⚠️ Error enriching profile {prof.get('name', 'Unknown')}: {str(e)}")
                error_count += 1
                pbar.update(1)
    
    cursor.close()
    
    print("\n" + "="*60)
    print(f"✅ Enrichment complete!")
    print(f"   - Profiles enriched: {enriched_count}")
    print(f"   - Errors: {error_count}")
    print(f"   - Success rate: {(enriched_count/total_count)*100:.1f}%")
    
    # Show sample enriched profile
    sample = db["public_profiles"].find_one({"quantum_features": {"$exists": True}})
    if sample:
        print("\n📊 Sample enriched profile:")
        print(f"   Name: {sample.get('name', 'Unknown')}")
        if 'quantum_features' in sample:
            qf = sample['quantum_features']
            print(f"   Skills Vector: {qf.get('skills_vector', [])[:3]}...")
            print(f"   Availability: {qf.get('availability', 0):.2f}")
            print(f"   Transition Probability: {qf.get('transition_probability', 0):.2f}")
        if 'network_metrics' in sample:
            nm = sample['network_metrics']
            print(f"   Centrality: {nm.get('centrality', 0):.2f}")
            print(f"   Bridge Score: {nm.get('bridge_score', 0):.2f}")
        if 'serendipity_factors' in sample:
            sf = sample['serendipity_factors']
            print(f"   Uniqueness: {sf.get('uniqueness', 0):.2f}")
            print(f"   Timing Score: {sf.get('timing_score', 0):.2f}")

def verify_enrichment():
    """
    Verify that profiles have been enriched with quantum features
    """
    mongo_uri = os.getenv("MONGODB_URI")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = client[os.getenv("DB_NAME", "MagicCRM")]
    
    total = db["public_profiles"].count_documents({})
    enriched = db["public_profiles"].count_documents({"quantum_features": {"$exists": True}})
    
    print(f"\n📈 Enrichment Status:")
    print(f"   Total profiles: {total}")
    print(f"   Enriched profiles: {enriched}")
    print(f"   Coverage: {(enriched/total)*100:.1f}%")
    
    return enriched, total

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Enrich MongoDB profiles with quantum features")
    parser.add_argument("--limit", type=int, help="Limit number of profiles to enrich")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    parser.add_argument("--verify", action="store_true", help="Just verify enrichment status")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_enrichment()
    else:
        enrich_all_profiles(limit=args.limit, batch_size=args.batch_size)
        verify_enrichment()