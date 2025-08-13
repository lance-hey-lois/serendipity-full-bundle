#!/usr/bin/env python3
"""
Fast Re-embedding Pipeline
==========================
Re-embeds all profiles with text-embedding-3-large at 1536 dimensions
"""

import os
import sys
from dotenv import load_dotenv
from pymongo import MongoClient
from openai import OpenAI
import numpy as np
from typing import List, Dict, Any
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client[os.getenv("DB_NAME", "MagicCRM")]

def create_profile_text(profile: Dict[str, Any]) -> str:
    """Create rich text representation for embedding"""
    parts = []
    
    # Core profile info
    if profile.get('name'):
        parts.append(f"Name: {profile['name']}")
    
    if profile.get('title'):
        parts.append(f"Title: {profile['title']}")
    
    if profile.get('company'):
        parts.append(f"Company: {profile['company']}")
    
    # Skills and expertise
    if profile.get('areasOfNetworkStrength'):
        skills = ', '.join(profile['areasOfNetworkStrength'])
        parts.append(f"Skills and Expertise: {skills}")
    
    # Bio/description
    if profile.get('blurb'):
        parts.append(f"Bio: {profile['blurb']}")
    
    # Industry/domain context
    if profile.get('industry'):
        parts.append(f"Industry: {profile['industry']}")
    
    return " | ".join(parts)

def get_embedding(text: str, retry_count: int = 3) -> List[float]:
    """Get embedding with retry logic"""
    for attempt in range(retry_count):
        try:
            response = openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=text,
                dimensions=1536
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt < retry_count - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to get embedding after {retry_count} attempts: {e}")
                return None

def process_profile_batch(profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process a batch of profiles"""
    updates = []
    
    for profile in profiles:
        try:
            # Create text representation
            profile_text = create_profile_text(profile)
            
            # Get embedding
            embedding = get_embedding(profile_text)
            
            if embedding:
                updates.append({
                    "_id": profile["_id"],
                    "embedding": embedding,
                    "embedding_version": "text-embedding-3-large-1536",
                    "embedding_updated": int(time.time())
                })
                logger.info(f"✅ Processed: {profile.get('name', 'Unknown')}")
            else:
                logger.error(f"❌ Failed: {profile.get('name', 'Unknown')}")
                
        except Exception as e:
            logger.error(f"Error processing {profile.get('name', 'Unknown')}: {e}")
    
    return updates

def re_embed_collection(collection_name: str, user_id: str = None) -> int:
    """Re-embed all profiles in a collection"""
    collection = db[collection_name]
    
    # Build query
    query = {}
    if user_id:
        query["userId"] = user_id
    
    # Count total profiles
    total_profiles = collection.count_documents(query)
    logger.info(f"Re-embedding {total_profiles} profiles from {collection_name}")
    
    if total_profiles == 0:
        logger.warning(f"No profiles found in {collection_name}")
        return 0
    
    # Get all profiles
    profiles = list(collection.find(query))
    
    # Process in batches with threading
    batch_size = 10  # Small batches for API rate limits
    processed_count = 0
    
    for i in range(0, len(profiles), batch_size):
        batch = profiles[i:i + batch_size]
        
        # Process batch
        updates = process_profile_batch(batch)
        
        # Bulk update MongoDB
        if updates:
            from pymongo import UpdateOne
            bulk_ops = []
            for update in updates:
                profile_id = update.pop("_id")
                bulk_ops.append(
                    UpdateOne(
                        {"_id": profile_id},
                        {"$set": update}
                    )
                )
            
            if bulk_ops:
                result = collection.bulk_write(bulk_ops)
                processed_count += result.modified_count
                logger.info(f"Updated {result.modified_count} profiles. Progress: {processed_count}/{total_profiles}")
        
        # Rate limiting
        time.sleep(0.5)
    
    logger.info(f"✅ Re-embedded {processed_count} profiles in {collection_name}")
    return processed_count

def main():
    """Main execution"""
    print("🚀 Starting Fast Re-embedding Pipeline")
    print(f"Database: {db.name}")
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found")
        sys.exit(1)
    
    start_time = time.time()
    total_processed = 0
    
    # Re-embed private profiles for tconnors first
    print("\n📊 Re-embedding tconnors private_profiles...")
    count = re_embed_collection("private_profiles", "tconnors")
    total_processed += count
    
    # For now, skip public profiles prompt - just do tconnors
    print(f"✅ Completed tconnors re-embedding. Skipping public profiles for now.")
    
    # Summary
    elapsed_time = time.time() - start_time
    print(f"\n✅ Complete! Re-embedded {total_processed} profiles in {elapsed_time:.1f} seconds")
    print(f"⚡ Rate: {total_processed / elapsed_time:.1f} profiles/second")

if __name__ == "__main__":
    main()