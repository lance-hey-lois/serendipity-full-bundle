#!/usr/bin/env python3
"""
Generate OpenAI embeddings for all profiles
"""

import os
import sys
from pymongo import MongoClient
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import time
import numpy as np

# Load environment
load_dotenv('../.env.dev')
if not os.getenv("MONGODB_URI"):
    load_dotenv('../.env')

def generate_embeddings(limit: int = None, batch_size: int = 10):
    """
    Generate embeddings for profiles using OpenAI
    
    Args:
        limit: Maximum number of profiles to process
        batch_size: Number of profiles per batch
    """
    # Initialize
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client['MagicCRM']
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Get profiles without embeddings
    query = {'embedding': {'$exists': False}}
    total = db['public_profiles'].count_documents(query)
    
    if limit:
        total = min(total, limit)
    
    print(f"\n🚀 Generating embeddings for {total} profiles")
    print("=" * 60)
    
    cursor = db['public_profiles'].find(query).limit(limit or total)
    
    success_count = 0
    error_count = 0
    
    with tqdm(total=total, desc="Generating embeddings") as pbar:
        batch = []
        
        for profile in cursor:
            batch.append(profile)
            
            if len(batch) >= batch_size:
                # Process batch
                for prof in batch:
                    try:
                        # Create text representation
                        text_parts = []
                        
                        # Add name
                        if prof.get('name'):
                            text_parts.append(f"Name: {prof['name']}")
                        
                        # Add bio
                        if prof.get('bio'):
                            text_parts.append(f"Bio: {prof['bio']}")
                        
                        # Add skills
                        if prof.get('skills'):
                            skills_str = ', '.join(prof['skills']) if isinstance(prof['skills'], list) else str(prof['skills'])
                            text_parts.append(f"Skills: {skills_str}")
                        
                        # Add location
                        if prof.get('location'):
                            text_parts.append(f"Location: {prof['location']}")
                        
                        # Add any other relevant fields
                        if prof.get('headline'):
                            text_parts.append(f"Headline: {prof['headline']}")
                        
                        if prof.get('summary'):
                            text_parts.append(f"Summary: {prof['summary']}")
                        
                        if prof.get('experience'):
                            text_parts.append(f"Experience: {prof['experience']}")
                        
                        # Combine all text
                        full_text = " | ".join(text_parts)
                        
                        # Fallback if no text
                        if not full_text.strip():
                            full_text = prof.get('name', 'Professional profile')
                        
                        # Generate embedding
                        response = openai_client.embeddings.create(
                            model="text-embedding-ada-002",
                            input=full_text[:8000]  # Limit text length
                        )
                        
                        embedding = response.data[0].embedding
                        
                        # Update profile
                        db['public_profiles'].update_one(
                            {'_id': prof['_id']},
                            {'$set': {
                                'embedding': embedding,
                                'embedding_model': 'text-embedding-ada-002',
                                'embedding_generated_at': time.time()
                            }}
                        )
                        
                        success_count += 1
                        
                    except Exception as e:
                        print(f"\n⚠️ Error for {prof.get('name', 'Unknown')}: {str(e)}")
                        error_count += 1
                    
                    pbar.update(1)
                    time.sleep(0.1)  # Rate limiting
                
                batch = []
        
        # Process remaining
        for prof in batch:
            try:
                # Create text representation
                text_parts = []
                
                if prof.get('name'):
                    text_parts.append(f"Name: {prof['name']}")
                if prof.get('bio'):
                    text_parts.append(f"Bio: {prof['bio']}")
                if prof.get('skills'):
                    skills_str = ', '.join(prof['skills']) if isinstance(prof['skills'], list) else str(prof['skills'])
                    text_parts.append(f"Skills: {skills_str}")
                if prof.get('location'):
                    text_parts.append(f"Location: {prof['location']}")
                
                full_text = " | ".join(text_parts)
                if not full_text.strip():
                    full_text = prof.get('name', 'Professional profile')
                
                # Generate embedding
                response = openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=full_text[:8000]
                )
                
                embedding = response.data[0].embedding
                
                # Update profile
                db['public_profiles'].update_one(
                    {'_id': prof['_id']},
                    {'$set': {
                        'embedding': embedding,
                        'embedding_model': 'text-embedding-ada-002',
                        'embedding_generated_at': time.time()
                    }}
                )
                
                success_count += 1
                
            except Exception as e:
                print(f"\n⚠️ Error: {str(e)}")
                error_count += 1
            
            pbar.update(1)
    
    print("\n" + "=" * 60)
    print(f"✅ Embedding generation complete!")
    print(f"   Successfully generated: {success_count}")
    print(f"   Errors: {error_count}")
    
    # Verify
    total_with_embeddings = db['public_profiles'].count_documents({'embedding': {'$exists': True}})
    print(f"   Total profiles with embeddings: {total_with_embeddings}")
    
    # Show sample
    sample = db['public_profiles'].find_one({'embedding': {'$exists': True}})
    if sample:
        print(f"\n📊 Sample profile with embedding:")
        print(f"   Name: {sample.get('name', 'Unknown')}")
        print(f"   Embedding dimensions: {len(sample['embedding'])}")
        print(f"   First 5 values: {sample['embedding'][:5]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of profiles")
    parser.add_argument("--batch-size", type=int, default=10)
    
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️ OPENAI_API_KEY not found in environment!")
        print("Please set it in .env or .env.dev")
        sys.exit(1)
    
    generate_embeddings(limit=args.limit, batch_size=args.batch_size)