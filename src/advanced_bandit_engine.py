"""
Advanced Bandit Engine for Serendipity Discovery
Implements state-of-the-art contextual bandits with neural networks and quantum exploration.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from collections import deque
import math

# Optional quantum imports
try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BanditConfig:
    """Configuration for bandit algorithms."""
    exploration_factor: float = 1.0
    learning_rate: float = 0.001
    batch_size: int = 64
    memory_size: int = 10000
    update_frequency: int = 100
    confidence_decay: float = 0.99
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995


class BanditAlgorithm(ABC):
    """Abstract base class for bandit algorithms."""
    
    @abstractmethod
    def select_arm(self, context: np.ndarray, available_arms: List[int]) -> int:
        """Select an arm given context and available arms."""
        pass
    
    @abstractmethod
    def update(self, context: np.ndarray, arm: int, reward: float):
        """Update the algorithm with observed reward."""
        pass
    
    @abstractmethod
    def get_confidence(self, context: np.ndarray, arm: int) -> float:
        """Get confidence in the expected reward for an arm."""
        pass


class ThompsonSamplingBandit(BanditAlgorithm):
    """
    Advanced Thompson Sampling with beta-binomial conjugate priors.
    Enhanced with dynamic bin allocation and confidence tracking.
    """
    
    def __init__(self, n_arms: int, n_bins: int = 10, config: BanditConfig = None):
        self.n_arms = n_arms
        self.n_bins = n_bins
        self.config = config or BanditConfig()
        
        # Initialize bins for each arm
        self.arm_bins = []
        for arm in range(n_arms):
            bins = []
            for i in range(n_bins):
                bins.append({
                    'lo': i / n_bins,
                    'hi': (i + 1) / n_bins,
                    'alpha': 1.0,
                    'beta': 1.0,
                    'count': 0,
                    'recent_rewards': deque(maxlen=100)
                })
            self.arm_bins.append(bins)
        
        self.total_pulls = 0
        self.arm_pulls = np.zeros(n_arms)
        
    def select_arm(self, context: np.ndarray, available_arms: List[int]) -> int:
        """Select arm using Thompson sampling across bins."""
        best_score = -float('inf')
        best_arm = available_arms[0]
        
        # Get novelty score from context (assume last element)
        novelty = context[-1] if len(context) > 0 else 0.5
        
        for arm in available_arms:
            # Find appropriate bin for this novelty level
            bin_idx = min(int(novelty * self.n_bins), self.n_bins - 1)
            bin_data = self.arm_bins[arm][bin_idx]
            
            # Sample from beta distribution
            sample = np.random.beta(bin_data['alpha'], bin_data['beta'])
            
            # Add exploration bonus
            exploration_bonus = self.config.exploration_factor * np.sqrt(
                np.log(self.total_pulls + 1) / (self.arm_pulls[arm] + 1)
            )
            
            score = sample + exploration_bonus
            
            if score > best_score:
                best_score = score
                best_arm = arm
        
        return best_arm
    
    def update(self, context: np.ndarray, arm: int, reward: float):
        """Update bin statistics with observed reward."""
        novelty = context[-1] if len(context) > 0 else 0.5
        bin_idx = min(int(novelty * self.n_bins), self.n_bins - 1)
        
        bin_data = self.arm_bins[arm][bin_idx]
        
        # Update beta parameters
        bin_data['alpha'] += reward
        bin_data['beta'] += (1.0 - reward)
        bin_data['count'] += 1
        bin_data['recent_rewards'].append(reward)
        
        # Update global counters
        self.total_pulls += 1
        self.arm_pulls[arm] += 1
    
    def get_confidence(self, context: np.ndarray, arm: int) -> float:
        """Get confidence based on bin statistics."""
        novelty = context[-1] if len(context) > 0 else 0.5
        bin_idx = min(int(novelty * self.n_bins), self.n_bins - 1)
        
        bin_data = self.arm_bins[arm][bin_idx]
        
        # Confidence based on number of observations
        confidence = min(1.0, bin_data['count'] / 100.0)
        return confidence
    
    def get_bin_statistics(self) -> Dict:
        """Get detailed statistics for all bins."""
        stats = {}
        for arm in range(self.n_arms):
            arm_stats = []
            for bin_idx, bin_data in enumerate(self.arm_bins[arm]):
                mean_reward = bin_data['alpha'] / (bin_data['alpha'] + bin_data['beta'])
                arm_stats.append({
                    'bin_range': f"{bin_data['lo']:.2f}-{bin_data['hi']:.2f}",
                    'count': bin_data['count'],
                    'mean_reward': mean_reward,
                    'alpha': bin_data['alpha'],
                    'beta': bin_data['beta']
                })
            stats[f'arm_{arm}'] = arm_stats
        return stats


class NeuralContextualBandit(BanditAlgorithm, nn.Module):
    """
    Deep contextual bandit using neural networks with uncertainty estimation.
    """
    
    def __init__(self, context_dim: int, n_arms: int, 
                 hidden_dims: List[int] = None, config: BanditConfig = None):
        super().__init__()
        
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.config = config or BanditConfig()
        
        hidden_dims = hidden_dims or [128, 64, 32]
        
        # Build feature network
        layers = []
        input_dim = context_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        
        # Arm-specific heads for expected rewards
        self.reward_heads = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(n_arms)
        ])
        
        # Uncertainty estimation heads
        self.uncertainty_heads = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(n_arms)
        ])
        
        # Experience replay buffer
        self.memory = deque(maxlen=config.memory_size)
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        
        # Training statistics
        self.update_count = 0
        self.epsilon = 1.0
        
        logger.info(f"Initialized NeuralContextualBandit: {context_dim}D context, {n_arms} arms")
    
    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning expected rewards and uncertainties."""
        features = self.feature_net(context)
        
        # Compute expected rewards
        rewards = torch.stack([
            head(features).squeeze(-1) for head in self.reward_heads
        ], dim=-1)
        
        # Compute uncertainties (log-variance)
        log_variances = torch.stack([
            head(features).squeeze(-1) for head in self.uncertainty_heads
        ], dim=-1)
        
        uncertainties = torch.exp(log_variances)
        
        return rewards, uncertainties
    
    def select_arm(self, context: np.ndarray, available_arms: List[int]) -> int:
        """Select arm using Upper Confidence Bound with neural estimates."""
        self.eval()
        
        with torch.no_grad():
            context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
            rewards, uncertainties = self.forward(context_tensor)
            
            # UCB selection
            exploration_bonus = self.config.exploration_factor * torch.sqrt(uncertainties)
            ucb_scores = rewards + exploration_bonus
            
            # Only consider available arms
            available_scores = ucb_scores[0, available_arms]
            best_idx = available_scores.argmax().item()
            
            # Epsilon-greedy exploration
            if np.random.random() < self.epsilon:
                return np.random.choice(available_arms)
            
            return available_arms[best_idx]
    
    def update(self, context: np.ndarray, arm: int, reward: float):
        """Update network with observed reward."""
        # Add to memory
        self.memory.append((context.copy(), arm, reward))
        
        # Update every N steps
        if len(self.memory) >= self.config.batch_size and \
           self.update_count % self.config.update_frequency == 0:
            self._train_step()
        
        self.update_count += 1
        
        # Decay epsilon
        self.epsilon = max(self.config.epsilon_min, 
                          self.epsilon * self.config.epsilon_decay)
    
    def _train_step(self):
        """Perform a training step using experience replay."""
        self.train()
        
        # Sample batch from memory
        batch_size = min(self.config.batch_size, len(self.memory))
        batch_indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]
        
        contexts = torch.tensor([item[0] for item in batch], dtype=torch.float32)
        arms = torch.tensor([item[1] for item in batch], dtype=torch.long)
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32)
        
        # Forward pass
        predicted_rewards, predicted_uncertainties = self.forward(contexts)
        
        # Extract predictions for selected arms
        batch_indices = torch.arange(batch_size)
        selected_rewards = predicted_rewards[batch_indices, arms]
        selected_uncertainties = predicted_uncertainties[batch_indices, arms]
        
        # Compute losses
        reward_loss = nn.MSELoss()(selected_rewards, rewards)
        
        # Uncertainty loss (negative log-likelihood)
        uncertainty_loss = 0.5 * (torch.log(selected_uncertainties) + 
                                 (rewards - selected_rewards).pow(2) / selected_uncertainties)
        uncertainty_loss = uncertainty_loss.mean()
        
        total_loss = reward_loss + 0.1 * uncertainty_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        if self.update_count % 1000 == 0:
            logger.info(f"Neural bandit update {self.update_count}: "
                       f"reward_loss={reward_loss:.4f}, uncertainty_loss={uncertainty_loss:.4f}")
    
    def get_confidence(self, context: np.ndarray, arm: int) -> float:
        """Get confidence based on uncertainty estimates."""
        self.eval()
        
        with torch.no_grad():
            context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
            _, uncertainties = self.forward(context_tensor)
            
            # Convert uncertainty to confidence (inverse relationship)
            uncertainty = uncertainties[0, arm].item()
            confidence = 1.0 / (1.0 + uncertainty)
            
            return confidence


class QuantumExplorationBandit(BanditAlgorithm):
    """
    Quantum-enhanced bandit using quantum circuits for exploration.
    """
    
    def __init__(self, context_dim: int, n_arms: int, 
                 n_qubits: int = 4, config: BanditConfig = None):
        
        if not HAS_PENNYLANE:
            raise ImportError("PennyLane required for QuantumExplorationBandit")
        
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.n_qubits = min(n_qubits, int(np.log2(n_arms)) + 2)
        self.config = config or BanditConfig()
        
        # Quantum device
        self.device = qml.device("default.qubit", wires=self.n_qubits)
        
        # Quantum circuit for exploration
        self.quantum_circuit = self._build_quantum_circuit()
        
        # Classical components
        self.classical_bandit = ThompsonSamplingBandit(n_arms, config=config)
        
        # Quantum parameters
        self.quantum_params = np.random.normal(0, 0.1, size=(2, self.n_qubits))
        
        logger.info(f"Initialized QuantumExplorationBandit: {n_qubits} qubits, {n_arms} arms")
    
    def _build_quantum_circuit(self):
        """Build quantum circuit for exploration."""
        
        @qml.qnode(self.device)
        def circuit(context, params):
            # Encode context
            for i in range(min(len(context), self.n_qubits)):
                qml.RY(context[i] * np.pi, wires=i)
            
            # Variational layer
            for i in range(self.n_qubits):
                qml.RY(params[0, i], wires=i)
                qml.RZ(params[1, i], wires=i)
            
            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            
            # Measurement
            return qml.probs(wires=range(self.n_qubits))
        
        return circuit
    
    def select_arm(self, context: np.ndarray, available_arms: List[int]) -> int:
        """Select arm using quantum-enhanced exploration."""
        
        # Classical arm selection
        classical_arm = self.classical_bandit.select_arm(context, available_arms)
        
        # Quantum exploration decision
        normalized_context = context / (np.linalg.norm(context) + 1e-8)
        quantum_probs = self.quantum_circuit(normalized_context, self.quantum_params)
        
        # Use quantum measurement to decide exploration
        quantum_sample = np.random.choice(len(quantum_probs), p=quantum_probs)
        exploration_threshold = quantum_sample / len(quantum_probs)
        
        if exploration_threshold > 0.7:  # High quantum entropy -> explore
            return np.random.choice(available_arms)
        else:
            return classical_arm
    
    def update(self, context: np.ndarray, arm: int, reward: float):
        """Update both classical and quantum components."""
        # Update classical bandit
        self.classical_bandit.update(context, arm, reward)
        
        # Adapt quantum parameters based on reward
        if reward > 0.5:  # Good reward -> adjust parameters
            learning_rate = 0.01
            gradient = np.random.normal(0, 0.1, self.quantum_params.shape)
            self.quantum_params += learning_rate * gradient * reward
        
        # Keep parameters in reasonable range
        self.quantum_params = np.clip(self.quantum_params, -np.pi, np.pi)
    
    def get_confidence(self, context: np.ndarray, arm: int) -> float:
        """Get confidence from classical component."""
        return self.classical_bandit.get_confidence(context, arm)


class HybridBanditEnsemble:
    """
    Ensemble of multiple bandit algorithms with dynamic weighting.
    """
    
    def __init__(self, context_dim: int, n_arms: int, config: BanditConfig = None):
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.config = config or BanditConfig()
        
        # Initialize different bandit algorithms
        self.bandits = {
            'thompson': ThompsonSamplingBandit(n_arms, config=config),
            'neural': NeuralContextualBandit(context_dim, n_arms, config=config)
        }
        
        # Add quantum bandit if available
        if HAS_PENNYLANE:
            self.bandits['quantum'] = QuantumExplorationBandit(
                context_dim, n_arms, config=config
            )
        
        # Bandit weights (start equal)
        self.bandit_weights = {name: 1.0 for name in self.bandits.keys()}
        
        # Performance tracking
        self.bandit_performance = {name: deque(maxlen=1000) for name in self.bandits.keys()}
        
        logger.info(f"Initialized HybridBanditEnsemble with {len(self.bandits)} algorithms")
    
    def select_arm(self, context: np.ndarray, available_arms: List[int]) -> Tuple[int, str]:
        """Select arm using weighted ensemble of algorithms."""
        
        # Select which bandit to use based on weights
        bandit_names = list(self.bandit_weights.keys())
        weights = np.array([self.bandit_weights[name] for name in bandit_names])
        weights = weights / weights.sum()
        
        selected_bandit_name = np.random.choice(bandit_names, p=weights)
        selected_bandit = self.bandits[selected_bandit_name]
        
        # Get arm selection
        selected_arm = selected_bandit.select_arm(context, available_arms)
        
        return selected_arm, selected_bandit_name
    
    def update(self, context: np.ndarray, arm: int, reward: float, bandit_used: str):
        """Update the specific bandit that was used and adjust weights."""
        
        # Update the bandit that made the selection
        self.bandits[bandit_used].update(context, arm, reward)
        
        # Track performance
        self.bandit_performance[bandit_used].append(reward)
        
        # Update weights based on recent performance
        self._update_weights()
    
    def _update_weights(self):
        """Update bandit weights based on recent performance."""
        for name in self.bandits.keys():
            if len(self.bandit_performance[name]) > 10:
                recent_performance = np.mean(list(self.bandit_performance[name])[-100:])
                # Exponential weighting toward better performing bandits
                self.bandit_weights[name] = np.exp(recent_performance * 2.0)
        
        # Normalize weights
        total_weight = sum(self.bandit_weights.values())
        if total_weight > 0:
            self.bandit_weights = {
                name: weight / total_weight 
                for name, weight in self.bandit_weights.items()
            }
    
    def get_ensemble_stats(self) -> Dict:
        """Get comprehensive statistics for the ensemble."""
        stats = {
            'bandit_weights': self.bandit_weights.copy(),
            'bandit_performance': {
                name: {
                    'mean_reward': np.mean(list(perf)) if perf else 0.0,
                    'n_selections': len(perf)
                }
                for name, perf in self.bandit_performance.items()
            }
        }
        
        # Add algorithm-specific stats
        if 'thompson' in self.bandits:
            stats['thompson_bins'] = self.bandits['thompson'].get_bin_statistics()
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    # Set up test environment
    context_dim = 32
    n_arms = 100
    n_episodes = 5000
    
    # Create test data
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Simulate different user contexts and arm rewards
    def simulate_reward(context: np.ndarray, arm: int) -> float:
        """Simulate reward based on context and arm."""
        # Simple reward model: dot product + noise
        arm_features = np.random.randn(context_dim)  # Fixed for each arm
        base_reward = np.dot(context, arm_features) / context_dim
        noise = np.random.normal(0, 0.1)
        return max(0.0, min(1.0, 0.5 + base_reward + noise))
    
    # Test different bandit algorithms
    algorithms = {
        'Thompson Sampling': ThompsonSamplingBandit(n_arms),
        'Neural Bandit': NeuralContextualBandit(context_dim, n_arms),
        'Hybrid Ensemble': HybridBanditEnsemble(context_dim, n_arms)
    }
    
    # Add quantum if available
    if HAS_PENNYLANE:
        algorithms['Quantum Bandit'] = QuantumExplorationBandit(context_dim, n_arms)
    
    # Run simulation
    results = {name: {'rewards': [], 'regrets': []} for name in algorithms.keys()}
    
    logger.info(f"Running bandit simulation: {n_episodes} episodes, {n_arms} arms")
    
    for episode in range(n_episodes):
        # Generate random context
        context = np.random.randn(context_dim)
        available_arms = list(range(n_arms))
        
        # Test each algorithm
        for name, algorithm in algorithms.items():
            # Select arm
            if isinstance(algorithm, HybridBanditEnsemble):
                selected_arm, bandit_used = algorithm.select_arm(context, available_arms)
            else:
                selected_arm = algorithm.select_arm(context, available_arms)
                bandit_used = None
            
            # Get reward
            reward = simulate_reward(context, selected_arm)
            
            # Update algorithm
            if isinstance(algorithm, HybridBanditEnsemble):
                algorithm.update(context, selected_arm, reward, bandit_used)
            else:
                algorithm.update(context, selected_arm, reward)
            
            # Track performance
            results[name]['rewards'].append(reward)
            
            # Compute regret (simplified)
            optimal_reward = max(simulate_reward(context, arm) for arm in available_arms[:10])
            regret = optimal_reward - reward
            results[name]['regrets'].append(regret)
        
        # Log progress
        if (episode + 1) % 1000 == 0:
            logger.info(f"Episode {episode + 1}/{n_episodes}")
            for name in algorithms.keys():
                recent_reward = np.mean(results[name]['rewards'][-100:])
                logger.info(f"  {name}: {recent_reward:.3f} avg reward")
    
    # Final results
    logger.info("Final Results:")
    for name in algorithms.keys():
        avg_reward = np.mean(results[name]['rewards'])
        avg_regret = np.mean(results[name]['regrets'])
        logger.info(f"{name}:")
        logger.info(f"  Average Reward: {avg_reward:.4f}")
        logger.info(f"  Average Regret: {avg_regret:.4f}")
    
    # Show ensemble stats if available
    if 'Hybrid Ensemble' in algorithms:
        ensemble_stats = algorithms['Hybrid Ensemble'].get_ensemble_stats()
        logger.info(f"Ensemble Stats: {ensemble_stats}")