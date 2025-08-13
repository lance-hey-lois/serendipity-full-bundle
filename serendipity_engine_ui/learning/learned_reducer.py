"""
Learned Reducer for Quantum Discovery
Trains on actual connection data to learn what embedding dimensions matter for real connections
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Any
from pymongo import MongoClient
import os
from dotenv import load_dotenv
# from tqdm import tqdm  # May not be installed
import json

load_dotenv('../.env.dev')
if not os.getenv("MONGODB_URI"):
    load_dotenv('../.env')

class ConnectionReducer(nn.Module):
    """
    Learns to reduce 1536-dim embeddings to quantum-compatible size
    by training on actual connection data
    """
    
    def __init__(self, input_dim: int = 1536, output_dim: int = 8):
        super().__init__()
        
        # Simple but effective architecture
        # We want to learn what matters for connections
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, 512),  # *2 because we concatenate pairs
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim * 2),  # *2 for both users
        )
        
        # Output: 8 dims for user A, 8 dims for user B
        self.output_dim = output_dim
        
    def forward(self, x):
        """
        Input: concatenated embeddings of two users [emb_a || emb_b]
        Output: reduced features [reduced_a || reduced_b]
        """
        return self.encoder(x)
    
    def reduce_single(self, embedding):
        """
        Reduce a single embedding (for inference)
        We'll use a trick: pair with itself and take first half
        """
        doubled = torch.cat([embedding, embedding])
        reduced = self.encoder(doubled.unsqueeze(0))
        return reduced[0, :self.output_dim]


class EdgeDataset:
    """
    Creates training data from MongoDB edges
    """
    
    def __init__(self, db):
        self.db = db
        self.profiles_cache = {}
        
    def get_profile_embedding(self, slug: str) -> np.ndarray:
        """Get embedding for a profile by slug"""
        if slug not in self.profiles_cache:
            profile = self.db['public_profiles'].find_one(
                {'slug': slug},
                {'embedding': 1}
            )
            if profile and 'embedding' in profile:
                self.profiles_cache[slug] = np.array(profile['embedding'])
            else:
                return None
        return self.profiles_cache.get(slug)
    
    def generate_training_data(self, limit: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate positive and negative examples
        Returns: (X, y) where X is pairs of embeddings, y is 0/1 labels
        """
        X = []
        y = []
        
        # Get all edges (positive examples) - start MUCH smaller
        edges = list(self.db['private_profiles'].find().limit(limit or 100))  # Start with just 100
        
        print(f"Processing {len(edges)} edges...")
        
        positive_pairs = set()
        all_slugs = set()
        
        # First pass: collect all slugs
        for edge in edges:
            user_slug = edge.get('userId')
            connection_slug = edge.get('slug')
            if user_slug and connection_slug:
                all_slugs.add(user_slug)
                all_slugs.add(connection_slug)
        
        print(f"Found {len(all_slugs)} unique users")
        
        # Batch fetch ALL embeddings at once
        print("Fetching all embeddings in one batch...")
        profiles_batch = list(self.db['public_profiles'].find(
            {'slug': {'$in': list(all_slugs)}},
            {'slug': 1, 'embedding': 1}
        ))
        
        # Build cache from batch
        for profile in profiles_batch:
            if profile.get('embedding'):
                self.profiles_cache[profile['slug']] = np.array(profile['embedding'])
        
        print(f"Cached {len(self.profiles_cache)} embeddings")
        
        # Now process edges with cached embeddings
        for i, edge in enumerate(edges):
            if i % 50 == 0:
                print(f"Processing edge {i}/{len(edges)}...")
            user_slug = edge.get('userId')
            connection_slug = edge.get('slug')
            
            if user_slug and connection_slug:
                # Get embeddings from cache
                emb_a = self.profiles_cache.get(user_slug)
                emb_b = self.profiles_cache.get(connection_slug)
                
                if emb_a is not None and emb_b is not None:
                    # Positive example (they ARE connected)
                    X.append(np.concatenate([emb_a, emb_b]))
                    y.append(1)
                    positive_pairs.add((user_slug, connection_slug))
                    positive_pairs.add((connection_slug, user_slug))  # Bidirectional
        
        print(f"Loaded {len(X)} positive examples")
        
        # Generate negative examples (pairs that AREN'T connected)
        # Focus on hard negatives: high cosine similarity but no edge
        all_slugs = list(all_slugs)
        n_negatives = min(len(X), 50)  # Limit negatives for testing
        
        print(f"Generating {n_negatives} negative examples...")
        
        # Use cached embeddings
        slug_embeddings = self.profiles_cache
        
        # Generate hard negatives (high similarity but not connected)
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
                emb_a = slug_embeddings.get(slug_a)
                emb_b = slug_embeddings.get(slug_b)
                
                if emb_a is not None and emb_b is not None:
                    # Compute cosine similarity
                    cos_sim = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
                    
                    # Prefer hard negatives (high similarity but not connected)
                    if cos_sim > 0.7 or np.random.rand() < 0.3:  # 30% random negatives too
                        X.append(np.concatenate([emb_a, emb_b]))
                        y.append(0)
                        negative_count += 1
        
        print(f"Generated {negative_count} negative examples")
        
        return np.array(X), np.array(y)


def train_reducer(output_dim: int = 8, epochs: int = 10):  # Further reduced for testing
    """
    Train the reducer on actual connection data
    """
    # Connect to MongoDB
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client['MagicCRM']
    
    # Create dataset
    print("Initializing dataset...")
    dataset = EdgeDataset(db)
    print("Generating training data...")
    X, y = dataset.generate_training_data(limit=50)  # Start with just 50 edges
    
    print(f"\nTraining data shape: {X.shape}")
    print(f"Positive examples: {np.sum(y)}")
    print(f"Negative examples: {len(y) - np.sum(y)}")
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Split into train/val
    n_train = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
    X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]
    
    # Create model
    model = ConnectionReducer(input_dim=1536, output_dim=output_dim)
    
    # Add a classifier head for training
    classifier = nn.Sequential(
        nn.Linear(output_dim * 2, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )
    
    # Training setup
    optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=0.001)
    criterion = nn.BCELoss()
    
    print("\nTraining reducer...")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        classifier.train()
        
        # Batch training
        batch_size = 32
        train_losses = []
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            
            # Forward pass
            reduced = model(batch_X)
            predictions = classifier(reduced).squeeze()
            loss = criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        classifier.eval()
        
        with torch.no_grad():
            val_reduced = model(X_val)
            val_pred = classifier(val_reduced).squeeze()
            val_loss = criterion(val_pred, y_val)
            
            # Calculate accuracy
            val_acc = ((val_pred > 0.5) == y_val).float().mean()
        
        if epoch % 2 == 0:  # Print more frequently
            print(f"Epoch {epoch}: Train Loss={np.mean(train_losses):.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.3f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state': model.state_dict(),
                'classifier_state': classifier.state_dict(),
                'output_dim': output_dim,
                'val_acc': val_acc.item(),
                'val_loss': val_loss.item()
            }, 'learned_reducer.pt')
    
    print(f"\n✅ Training complete! Best validation accuracy: {val_acc:.3f}")
    print("Model saved to learned_reducer.pt")
    
    return model


def load_reducer(path: str = 'learned_reducer.pt') -> ConnectionReducer:
    """
    Load a trained reducer
    """
    checkpoint = torch.load(path)
    output_dim = checkpoint['output_dim']
    
    model = ConnectionReducer(input_dim=1536, output_dim=output_dim)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print(f"Loaded reducer: Val Acc={checkpoint['val_acc']:.3f}, Output Dim={output_dim}")
    
    return model


def reduce_profile_batch(profiles: List[Dict], model: ConnectionReducer) -> np.ndarray:
    """
    Reduce a batch of profiles using the learned model
    """
    model.eval()
    reduced = []
    
    with torch.no_grad():
        for profile in profiles:
            if 'embedding' in profile:
                emb = torch.FloatTensor(profile['embedding'])
                reduced_emb = model.reduce_single(emb)
                reduced.append(reduced_emb.numpy())
            else:
                # No embedding - return zeros
                reduced.append(np.zeros(model.output_dim))
    
    return np.array(reduced)


if __name__ == "__main__":
    # Train the reducer
    print("🚀 Starting learned reducer training...")
    print("This will train on actual connection data from MongoDB")
    print("-" * 60)
    model = train_reducer(output_dim=8, epochs=10)
    
    # Test on a sample
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client['MagicCRM']
    
    sample = db['public_profiles'].find_one({'embedding': {'$exists': True}})
    if sample:
        reduced = reduce_profile_batch([sample], model)
        print(f"\nSample reduction: {sample['name']}")
        print(f"Original: 1536 dims")
        print(f"Reduced: {reduced.shape[1]} dims")
        print(f"Values: {reduced[0][:5]}...")  # First 5 values