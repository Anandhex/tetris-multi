# powerup_dqn_agent.py - PyTorch version with CUDA support
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np
import random
import os
from typing import Dict, List, Tuple

class DQNNetwork(nn.Module):
    """PyTorch neural network for DQN"""
    
    def __init__(self, feature_size: int = 13, action_size: int = 4):
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(feature_size, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation for Q-values
        return x


class PowerupDQNAgent:
    """DQN agent for powerup selection using PyTorch with CUDA support"""
    
    def __init__(self, feature_size: int = 13, action_size: int = 4,
                 learning_rate: float = 0.001, epsilon: float = 1.0,
                 epsilon_min: float = 0.01, epsilon_decay: float = 0.995,
                 memory_size: int = 10000, batch_size: int = 32,
                 gamma: float = 0.95, tau: float = 0.005):
        
        self.feature_size = feature_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.gamma = gamma  # Discount factor
        self.tau = tau      # Soft update parameter
        
        # Check for CUDA availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self.memory = deque(maxlen=memory_size)
        
        # Build neural networks
        self.q_network = DQNNetwork(feature_size, action_size).to(self.device)
        self.target_network = DQNNetwork(feature_size, action_size).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize target network with same weights
        self.hard_update()
        
        # Import action space here to avoid circular imports
        from action_space import ActionSpace
        self.action_space = ActionSpace()
        
        print(f"PowerupDQNAgent initialized:")
        print(f"  - Features: {feature_size}")
        print(f"  - Actions: {action_size}")
        print(f"  - Device: {self.device}")
        print(f"  - Memory size: {memory_size}")
        print(f"  - Batch size: {batch_size}")
    
    def hard_update(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def soft_update(self):
        """Soft update target network: θ_target = τ*θ_local + (1-τ)*θ_target"""
        for target_param, local_param in zip(self.target_network.parameters(), 
                                           self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + 
                                  (1.0 - self.tau) * target_param.data)
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, environment) -> Dict:
        """Choose action using epsilon-greedy policy"""
        features = environment.get_features()
        board = environment.get_board_state()
        powerups = environment.get_powerup_availability()
        
        if np.random.random() <= self.epsilon:
            # Random action from valid actions
            valid_actions = self._get_valid_actions(powerups, board)
            action_id = random.choice(valid_actions)
        else:
            # Greedy action using neural network
            state_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor).cpu().numpy()[0]
            
            masked_q_values = self._mask_invalid_actions(q_values, powerups, board)
            action_id = np.argmax(masked_q_values)
        
        return self.action_space.decode_action(action_id, board)
    
    def _get_valid_actions(self, powerups: Dict[str, bool], board: np.ndarray) -> List[int]:
        """Get list of valid action IDs"""
        valid_actions = [0]  # 'none' is always valid
        
        if powerups.get('bottom_clear', False):
            valid_actions.append(1)
        if powerups.get('gravity', False):
            valid_actions.append(2)
        if powerups.get('bomb', False) and self._has_valid_bomb_positions(board):
            valid_actions.append(3)
        
        return valid_actions
    
    def _has_valid_bomb_positions(self, board: np.ndarray) -> bool:
        """Check if there are valid bomb positions (only surface blocks)"""
        surface_blocks = self._get_surface_blocks(board)
        
        for hit_row, hit_col in surface_blocks:
            effectiveness = self._calculate_bomb_effectiveness_on_surface(board, hit_row, hit_col)
            if effectiveness >= 3:  # At least 3 effectiveness to be worth it
                return True
        return False
    
    def _get_surface_blocks(self, board: np.ndarray) -> List[Tuple[int, int]]:
        """Get all surface blocks (blocks that can be hit by bomb from above)"""
        rows, cols = board.shape
        surface_blocks = []
        
        for col in range(cols):
            for row in range(rows):
                if board[row, col] == 1:  # Found a block
                    # This is a surface block (first block from top in this column)
                    surface_blocks.append((row, col))
                    break  # Only the topmost block in each column is reachable
        
        return surface_blocks
    
    def _calculate_bomb_effectiveness_on_surface(self, board: np.ndarray, hit_row: int, hit_col: int) -> float:
        """Calculate bomb effectiveness when bomb hits a surface block using relative offsets"""
        rows, cols = board.shape
        
        # Define 3x3 explosion pattern around hit point
        explosion_offsets = [
            (-1, -1), (-1, 0), (-1, 1),  # Row above
            (0, -1),  (0, 0),  (0, 1),   # Same row as hit
            (1, -1),  (1, 0),  (1, 1)    # Row below
        ]
        
        # Count blocks that would be destroyed in explosion
        blocks_destroyed = 0
        
        for row_offset, col_offset in explosion_offsets:
            explosion_row = hit_row + row_offset
            explosion_col = hit_col + col_offset
            
            # Check if position is valid (within board bounds)
            if 0 <= explosion_row < rows and 0 <= explosion_col < cols:
                if board[explosion_row, explosion_col] == 1:
                    blocks_destroyed += 1
        
        return float(blocks_destroyed)  # Simple effectiveness for validation
    
    def _mask_invalid_actions(self, q_values: np.ndarray, powerups: Dict[str, bool], 
                             board: np.ndarray) -> np.ndarray:
        """Mask invalid actions with very low values"""
        masked = q_values.copy()
        
        if not powerups.get('bottom_clear', False):
            masked[1] = -np.inf
        if not powerups.get('gravity', False):
            masked[2] = -np.inf
        if not powerups.get('bomb', False) or not self._has_valid_bomb_positions(board):
            masked[3] = -np.inf
        
        return masked
    
    def train(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Convert batch to tensors
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Soft update target network
        self.soft_update()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'feature_size': self.feature_size,
            'action_size': self.action_size,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'tau': self.tau
        }
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load network states
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training parameters
        self.epsilon = checkpoint.get('epsilon', 0.0)  # Set to 0 for inference
        
        # Verify model architecture matches
        if (checkpoint.get('feature_size', self.feature_size) != self.feature_size or
            checkpoint.get('action_size', self.action_size) != self.action_size):
            print("Warning: Model architecture mismatch!")
        
        print(f"Model loaded from {filepath}")
        print(f"Epsilon: {self.epsilon}")
    
    def set_eval_mode(self):
        """Set networks to evaluation mode"""
        self.q_network.eval()
        self.target_network.eval()
        self.epsilon = 0.0  # No exploration during evaluation
    
    def set_train_mode(self):
        """Set networks to training mode"""
        self.q_network.train()
        self.target_network.train()
    
    def get_model_info(self):
        """Get information about the model"""
        total_params = sum(p.numel() for p in self.q_network.parameters())
        trainable_params = sum(p.numel() for p in self.q_network.parameters() if p.requires_grad)
        
        info = {
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'memory_size': len(self.memory),
            'epsilon': self.epsilon,
            'feature_size': self.feature_size,
            'action_size': self.action_size
        }
        
        return info
    
    def print_model_info(self):
        """Print detailed model information"""
        info = self.get_model_info()
        print("\n" + "="*50)
        print("POWERUP DQN MODEL INFO")
        print("="*50)
        print(f"Device: {info['device']}")
        print(f"Total Parameters: {info['total_parameters']:,}")
        print(f"Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"Memory Usage: {info['memory_size']:,} / {self.memory.maxlen:,}")
        print(f"Current Epsilon: {info['epsilon']:.4f}")
        print(f"Feature Size: {info['feature_size']}")
        print(f"Action Size: {info['action_size']}")
        print("="*50 + "\n")


# Additional utility functions for PyTorch compatibility
def count_parameters(model):
    """Count total and trainable parameters in model"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def check_cuda_memory():
    """Check CUDA memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        print(f"CUDA Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
    else:
        print("CUDA not available")

def clear_cuda_cache():
    """Clear CUDA cache to free memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared")