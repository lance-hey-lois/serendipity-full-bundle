#!/usr/bin/env python3
"""
Two-Tower Learned Reducer for Quantum Discovery
Fixes the training/inference mismatch with proper Siamese architecture
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Any, Optional
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import json

load_dotenv('../.env.dev')
if not os.getenv("MONGODB_URI"):
    load_dotenv('../.env')


class Tower(nn.Module):
    """Single tower encoder - shared weights for both inputs"""
    
    def __init__(self, in_dim: int = 1536, out_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class TwoTowerModel(nn.Module):
    """Two-tower model with shared encoder and training head"""
    
    def __init__(self, in_dim: int = 1536, out_dim: int = 8):
        super().__init__()
        self.tower = Tower(in_dim, out_dim)  # Shared weights
        self.output_dim = out_dim
        
        # Classifier head for training only
        self.head = nn.Sequential(
            nn.Linear(out_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Single-tower encoding for inference"""
        return self.tower(x)
    
    def forward(self, a, b):
        """Two-tower forward for training"""
        za = self.tower(a)
        zb = self.tower(b)
        concatenated = torch.cat([za, zb], dim=-1)
        return self.head(concatenated)


class EdgeDataset:
    """Creates training data from MongoDB edges"""
    
    def __init__(self, db):
        self.db = db
        self.embeddings_cache = {}
        
    def get_profile_embedding(self, slug: str) -> Optional[np.ndarray]:
        """Get embedding for a profile by slug with caching"""
        if slug not in self.embeddings_cache:
            profile = self.db['public_profiles'].find_one(
                {'slug': slug},
                {'embedding': 1}
            )
            if profile and 'embedding' in profile:
                # Handle both 1536 and 3072 dims (truncate if needed)
                emb = np.array(profile['embedding'])
                if len(emb) > 1536:
                    emb = emb[:1536]  # Truncate text-embedding-3-large to 1536
                self.embeddings_cache[slug] = emb
            else:
                return None
        return self.embeddings_cache.get(slug)
    
    def generate_training_data(self, limit: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate positive and negative examples
        Returns: (X_a, X_b, y) where X_a and X_b are paired embeddings, y is 0/1 labels
        """
        X_a = []
        X_b = []
        y = []
        
        # Get edges from the correct collection
        # Check if we have an edges collection or use private_profiles
        if 'edges' in self.db.list_collection_names():
            edges = list(self.db['edges'].find({}, {'userId': 1, 'slug': 1}).limit(limit or 500))
        else:
            # Fallback to private_profiles
            edges = list(self.db['private_profiles'].find({}, {'userId': 1, 'slug': 1}).limit(limit or 500))
        
        print(f"Processing {len(edges)} edges...")
        
        positive_pairs = set()
        all_slugs = set()
        
        # Collect all unique slugs first
        for edge in edges:
            user_slug = edge.get('userId')
            connection_slug = edge.get('slug')
            if user_slug and connection_slug:
                all_slugs.add(user_slug)
                all_slugs.add(connection_slug)
        
        print(f"Found {len(all_slugs)} unique users")
        
        # Batch fetch all embeddings
        print("Fetching embeddings...")
        profiles = list(self.db['public_profiles'].find(
            {'slug': {'$in': list(all_slugs)}},
            {'slug': 1, 'embedding': 1}
        ))
        
        for profile in profiles:
            if profile.get('embedding'):
                emb = np.array(profile['embedding'])
                if len(emb) > 1536:
                    emb = emb[:1536]
                self.embeddings_cache[profile['slug']] = emb
        
        print(f"Cached {len(self.embeddings_cache)} embeddings")
        
        # Process edges for positive examples
        for edge in edges:
            user_slug = edge.get('userId')
            connection_slug = edge.get('slug')
            
            if user_slug and connection_slug:
                emb_a = self.embeddings_cache.get(user_slug)
                emb_b = self.embeddings_cache.get(connection_slug)
                
                if emb_a is not None and emb_b is not None:
                    # L2 normalize
                    emb_a = emb_a / (np.linalg.norm(emb_a) + 1e-12)
                    emb_b = emb_b / (np.linalg.norm(emb_b) + 1e-12)
                    
                    X_a.append(emb_a)
                    X_b.append(emb_b)
                    y.append(1)
                    positive_pairs.add((user_slug, connection_slug))
                    positive_pairs.add((connection_slug, user_slug))
        
        print(f"Loaded {len(y)} positive examples")
        
        # Generate hard negative examples
        all_slugs = list(all_slugs)
        n_negatives = min(len(y), 500)  # Balance dataset
        
        print(f"Generating {n_negatives} negative examples...")
        
        negative_count = 0
        attempts = 0
        max_attempts = n_negatives * 10
        
        while negative_count < n_negatives and attempts < max_attempts:
            attempts += 1
            
            # Pick two random users
            slug_a = np.random.choice(all_slugs)
            slug_b = np.random.choice(all_slugs)
            
            if slug_a == slug_b:
                continue
                
            if (slug_a, slug_b) not in positive_pairs:
                emb_a = self.embeddings_cache.get(slug_a)
                emb_b = self.embeddings_cache.get(slug_b)
                
                if emb_a is not None and emb_b is not None:
                    # L2 normalize
                    emb_a = emb_a / (np.linalg.norm(emb_a) + 1e-12)
                    emb_b = emb_b / (np.linalg.norm(emb_b) + 1e-12)
                    
                    # Prefer hard negatives (high similarity but not connected)
                    cos_sim = np.dot(emb_a, emb_b)
                    
                    if cos_sim > 0.7 or np.random.rand() < 0.3:
                        X_a.append(emb_a)
                        X_b.append(emb_b)
                        y.append(0)
                        negative_count += 1
        
        print(f"Generated {negative_count} negative examples")
        
        return np.array(X_a), np.array(X_b), np.array(y)


def train_two_tower_reducer(output_dim: int = 8, epochs: int = 50):
    """Train the two-tower reducer on actual connection data"""
    
    # Connect to MongoDB
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client['MagicCRM']
    
    # Create dataset
    print("Initializing dataset...")
    dataset = EdgeDataset(db)
    X_a, X_b, y = dataset.generate_training_data(limit=1000)
    
    print(f"\nTraining data shape: A={X_a.shape}, B={X_b.shape}")
    print(f"Positive examples: {np.sum(y)}")
    print(f"Negative examples: {len(y) - np.sum(y)}")
    
    # Convert to PyTorch tensors
    X_a_tensor = torch.FloatTensor(X_a)
    X_b_tensor = torch.FloatTensor(X_b)
    y_tensor = torch.FloatTensor(y)
    
    # Split into train/val
    n_train = int(0.8 * len(X_a))
    indices = np.random.permutation(len(X_a))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_a_train, X_b_train, y_train = X_a_tensor[train_idx], X_b_tensor[train_idx], y_tensor[train_idx]
    X_a_val, X_b_val, y_val = X_a_tensor[val_idx], X_b_tensor[val_idx], y_tensor[val_idx]
    
    # Create model
    model = TwoTowerModel(in_dim=1536, out_dim=output_dim)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    print("\nTraining two-tower reducer...")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        
        # Batch training
        batch_size = 32
        train_losses = []
        
        for i in range(0, len(X_a_train), batch_size):
            batch_a = X_a_train[i:i+batch_size]
            batch_b = X_b_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch_a, batch_b).squeeze()
            loss = criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        
        with torch.no_grad():
            val_pred = model(X_a_val, X_b_val).squeeze()
            val_loss = criterion(val_pred, y_val)
            
            # Calculate accuracy
            val_acc = ((val_pred > 0.5) == y_val).float().mean()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss={np.mean(train_losses):.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.3f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save only the tower weights
            torch.save({
                'tower_state': model.tower.state_dict(),
                'output_dim': output_dim,
                'val_acc': val_acc.item(),
                'val_loss': val_loss.item()
            }, 'two_tower_reducer.pt')
    
    print(f"\n✅ Training complete! Best validation accuracy: {val_acc:.3f}")
    print("Model saved to two_tower_reducer.pt")
    
    return model


def load_two_tower_reducer(path: str = 'two_tower_reducer.pt') -> Tower:
    """Load a trained two-tower reducer (returns just the tower for inference)"""
    
    checkpoint = torch.load(path, weights_only=True)
    output_dim = checkpoint['output_dim']
    
    tower = Tower(in_dim=1536, out_dim=output_dim)
    tower.load_state_dict(checkpoint['tower_state'])
    tower.eval()
    
    print(f"Loaded two-tower reducer: Val Acc={checkpoint['val_acc']:.3f}, Output Dim={output_dim}")
    
    return tower


def reduce_profiles_batch(profiles: List[Dict], tower: Tower) -> np.ndarray:
    """Reduce a batch of profiles using the trained tower"""
    
    tower.eval()
    reduced = []
    
    with torch.no_grad():
        for profile in profiles:
            if 'embedding' in profile:
                emb = np.array(profile['embedding'])
                # Truncate if needed
                if len(emb) > 1536:
                    emb = emb[:1536]
                # L2 normalize
                emb = emb / (np.linalg.norm(emb) + 1e-12)
                emb_tensor = torch.FloatTensor(emb)
                reduced_emb = tower(emb_tensor)
                reduced.append(reduced_emb.numpy())
            else:
                reduced.append(np.zeros(tower.net[-1].out_features))
    
    return np.array(reduced)


if __name__ == "__main__":
    # Train the two-tower reducer
    print("🚀 Starting two-tower reducer training...")
    model = train_two_tower_reducer(output_dim=8, epochs=50)
    
    # Test on a sample
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client['MagicCRM']
    
    sample = db['public_profiles'].find_one({'embedding': {'$exists': True}})
    if sample:
        tower = load_two_tower_reducer()
        reduced = reduce_profiles_batch([sample], tower)
        print(f"\nSample reduction: {sample.get('name', 'Unknown')}")
        print(f"Original: 1536 dims")
        print(f"Reduced: {reduced.shape[1]} dims")
        print(f"Values: {reduced[0][:5]}...")