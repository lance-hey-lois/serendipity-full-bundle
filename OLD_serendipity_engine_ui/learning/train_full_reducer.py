#!/usr/bin/env python3
"""
Train learned reducer on full connection dataset
"""

import os
import sys
sys.path.append('..')
from learned_reducer import train_reducer, load_reducer, reduce_profile_batch
from pymongo import MongoClient
from dotenv import load_dotenv
import numpy as np
import torch

# Load environment
load_dotenv('../.env.dev')
if not os.getenv("MONGODB_URI"):
    load_dotenv('../.env')

def train_full_model():
    """Train on more edges with better parameters"""
    print("\n🚀 Training Learned Reducer on Full Dataset")
    print("=" * 60)
    
    # Train with more data
    print("\n📊 Phase 1: Training on 500 edges...")
    model = train_reducer(output_dim=8, epochs=50)
    
    # Test the model
    print("\n🧪 Testing the trained model...")
    test_model()
    
    return model

def test_model():
    """Test the trained reducer on sample connections"""
    # Load the model
    model = load_reducer('learned_reducer.pt')
    
    # Connect to MongoDB
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client['MagicCRM']
    
    # Get a few test profiles
    test_profiles = list(db['public_profiles'].find(
        {'embedding': {'$exists': True}},
        {'name': 1, 'slug': 1, 'embedding': 1}
    ).limit(10))
    
    print("\n📊 Sample reductions:")
    for profile in test_profiles[:3]:
        reduced = reduce_profile_batch([profile], model)
        print(f"\n{profile.get('name', 'Unknown')}:")
        print(f"  Original: 1536 dims")
        print(f"  Reduced: {reduced.shape[1]} dims")
        print(f"  Values: {reduced[0][:3]}...")
    
    # Check for actual connections
    print("\n🔗 Checking actual connections...")
    
    # Find users with connections
    edges = list(db['private_profiles'].find().limit(10))
    
    for edge in edges[:3]:
        user_slug = edge.get('userId')
        connection_slug = edge.get('slug')
        
        if user_slug and connection_slug:
            user = db['public_profiles'].find_one({'slug': user_slug}, {'name': 1, 'embedding': 1})
            connection = db['public_profiles'].find_one({'slug': connection_slug}, {'name': 1, 'embedding': 1})
            
            if user and connection and user.get('embedding') and connection.get('embedding'):
                # Reduce both
                user_reduced = model.reduce_single(torch.FloatTensor(user['embedding'])).detach().numpy()
                conn_reduced = model.reduce_single(torch.FloatTensor(connection['embedding'])).detach().numpy()
                
                # Calculate cosine similarity in reduced space
                cos_sim = np.dot(user_reduced, conn_reduced) / (
                    np.linalg.norm(user_reduced) * np.linalg.norm(conn_reduced)
                )
                
                print(f"\n✅ Real connection:")
                print(f"  {user.get('name', 'Unknown')} → {connection.get('name', 'Unknown')}")
                print(f"  Cosine similarity in reduced space: {cos_sim:.3f}")

def enrich_with_learned_reducer():
    """Use the learned reducer to enrich profiles"""
    print("\n🔮 Enriching profiles with learned reducer...")
    
    # Load the model
    model = load_reducer('learned_reducer.pt')
    
    # Connect to MongoDB
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client['MagicCRM']
    
    # Batch process profiles
    batch_size = 100
    total_enriched = 0
    
    cursor = db['public_profiles'].find(
        {'embedding': {'$exists': True}},
        {'embedding': 1}
    ).limit(1000)  # Start with 1000
    
    batch = []
    for profile in cursor:
        batch.append(profile)
        
        if len(batch) >= batch_size:
            # Process batch
            reduced_batch = reduce_profile_batch(batch, model)
            
            # Update profiles
            for i, prof in enumerate(batch):
                db['public_profiles'].update_one(
                    {'_id': prof['_id']},
                    {'$set': {
                        'quantum_features_learned': {
                            'reduced_embedding': reduced_batch[i].tolist(),
                            'reducer_version': 'v1.0',
                            'dimensions': 8
                        }
                    }}
                )
                total_enriched += 1
            
            batch = []
            print(f"  Enriched {total_enriched} profiles...")
    
    # Process remaining
    if batch:
        reduced_batch = reduce_profile_batch(batch, model)
        for i, prof in enumerate(batch):
            db['public_profiles'].update_one(
                {'_id': prof['_id']},
                {'$set': {
                    'quantum_features_learned': {
                        'reduced_embedding': reduced_batch[i].tolist(),
                        'reducer_version': 'v1.0',
                        'dimensions': 8
                    }
                }}
            )
            total_enriched += 1
    
    print(f"\n✅ Enriched {total_enriched} profiles with learned reductions!")
    
    # Verify
    sample = db['public_profiles'].find_one({'quantum_features_learned': {'$exists': True}})
    if sample:
        print(f"\n📊 Sample enriched profile: {sample.get('name', 'Unknown')}")
        qfl = sample['quantum_features_learned']
        print(f"  Reduced dims: {qfl['dimensions']}")
        print(f"  First 3 values: {qfl['reduced_embedding'][:3]}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-only", action="store_true", help="Just test existing model")
    parser.add_argument("--enrich", action="store_true", help="Enrich profiles with learned reducer")
    
    args = parser.parse_args()
    
    if args.test_only:
        test_model()
    elif args.enrich:
        enrich_with_learned_reducer()
    else:
        # Train the full model
        model = train_full_model()
        
        # Optionally enrich
        response = input("\n🤔 Would you like to enrich profiles now? (y/n): ")
        if response.lower() == 'y':
            enrich_with_learned_reducer()