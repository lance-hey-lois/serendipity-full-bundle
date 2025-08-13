#!/usr/bin/env python3
"""
Debug quantum kernel computation
"""

import numpy as np
import pennylane as qml
import torch
import sys
import os
sys.path.append('..')
from learning.learned_reducer import load_reducer

# Load reducer
reducer = load_reducer('../learning/learned_reducer.pt')

# Create simple test embeddings
emb1 = np.random.randn(1536)
emb2 = np.random.randn(1536)

# Reduce them
with torch.no_grad():
    reduced1 = reducer.reduce_single(torch.FloatTensor(emb1)).detach().numpy()
    reduced2 = reducer.reduce_single(torch.FloatTensor(emb2)).detach().numpy()

print("Raw reduced values:")
print(f"  reduced1: {reduced1[:4]}...")
print(f"  reduced2: {reduced2[:4]}...")

# Try different normalizations
print("\nNormalization tests:")

# Method 1: Simple scaling
norm1 = (reduced1 - reduced1.min()) / (reduced1.max() - reduced1.min() + 1e-10)
norm1 = norm1 * np.pi
print(f"  Simple scaling: {norm1[:4]}...")

# Method 2: Tanh normalization
norm2 = (np.tanh(reduced1) + 1) * np.pi / 2
print(f"  Tanh scaling: {norm2[:4]}...")

# Method 3: Sigmoid normalization  
norm3 = 1 / (1 + np.exp(-reduced1)) * np.pi
print(f"  Sigmoid scaling: {norm3[:4]}...")

# Test quantum kernel with different values
n_qubits = 8
dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev)
def test_kernel(f1, f2):
    # Encode first feature vector
    for i in range(n_qubits):
        qml.RY(f1[i], wires=i)
        qml.RZ(f1[i], wires=i)
    
    # Entanglement
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    
    # Encode second (inverse)
    for i in range(n_qubits):
        qml.RZ(-f2[i], wires=i)
        qml.RY(-f2[i], wires=i)
    
    return qml.probs(wires=list(range(n_qubits)))

# Test with different normalizations
print("\nQuantum kernel tests:")

# Test 1: Identical vectors (should be high)
f1 = norm2
probs = test_kernel(f1, f1)
print(f"  Identical vectors: prob[0] = {probs[0]:.4f}")

# Test 2: Random different vectors
f2 = (np.tanh(reduced2) + 1) * np.pi / 2
probs = test_kernel(f1, f2)
print(f"  Different vectors: prob[0] = {probs[0]:.4f}")

# Test 3: Simple test values
test1 = np.ones(8) * np.pi/4
test2 = np.ones(8) * np.pi/4
probs = test_kernel(test1, test2)
print(f"  All π/4 vectors: prob[0] = {probs[0]:.4f}")

# Test 4: Orthogonal
test3 = np.array([np.pi/2] * 4 + [0] * 4)
test4 = np.array([0] * 4 + [np.pi/2] * 4)
probs = test_kernel(test3, test4)
print(f"  Orthogonal vectors: prob[0] = {probs[0]:.4f}")

print("\nDiagnostics:")
print(f"  Device: {dev.name}")
print(f"  Qubits: {dev.num_wires}")
print(f"  Probs shape: {probs.shape}")
print(f"  Sum of probs: {np.sum(probs):.6f}")