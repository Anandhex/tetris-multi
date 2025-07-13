# optimized_cnn_dqn_agent.py - Surface-only bomb targeting + WildBlocks
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np
import random
import os
from typing import Dict, List, Tuple, Optional
from powerup_training_visualizer import TrainingVisualizer, TrainingLogger

class OptimizedCNNDQN(nn.Module):
    """
    Optimized CNN with surface-only bomb targeting + WildBlocks
    Output: 5 action types + 10 bomb columns + 8 wildblock columns = 23 total outputs
    """
    
    def __init__(self, board_height=20, board_width=10):
        super(OptimizedCNNDQN, self).__init__()
        
        self.board_height = board_height
        self.board_width = board_width
        
        # Shared convolutional backbone (6 channels: own + opponent + 4 powerups)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Global features for action type
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Action type branch (5 outputs: none, bottom_clear, gravity, bomb, wildblocks)
        self.action_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # Added wildblocks
        )
        
        # Bomb column branch (10 outputs: one per column)
        self.bomb_column_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 10)),  # Pool to (1, 10) - one value per column
            nn.Flatten(),
            nn.Linear(128 * 10, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 10)  # Q-value for bombing each column
        )
        
        # WildBlocks column branch (8 outputs: columns 1-8)
        self.wildblock_column_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 8)),  # Pool to (1, 8) - middle columns
            nn.Flatten(),
            nn.Linear(128 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 8)  # Q-value for wildblock columns 1-8
        )
        
    def forward(self, x):
        # Shared features from conv layers
        features = self.conv_layers(x)  # (batch, 128, 20, 10)
        
        # Action type Q-values
        pooled_features = self.global_pool(features)
        action_q = self.action_branch(pooled_features)  # (batch, 5)
        
        # Bomb column Q-values
        bomb_col_q = self.bomb_column_branch(features)  # (batch, 10)
        
        # WildBlocks column Q-values
        wild_col_q = self.wildblock_column_branch(features)  # (batch, 8)
        
        # Concatenate outputs: [action_q, bomb_col_q, wild_col_q]
        # Output shape: (batch, 23) = 5 actions + 10 bomb columns + 8 wildblock columns
        output = torch.cat([action_q, bomb_col_q, wild_col_q], dim=1)
        
        return output


class OptimizedBombAgent:
    """
    Optimized agent with surface-only bomb targeting + WildBlocks
    """
    
    def __init__(self, board_height=20, board_width=10, **kwargs):
        self.board_height = board_height
        self.board_width = board_width
        
        self.learning_rate = kwargs.get('learning_rate', 0.0001)
        self.epsilon = kwargs.get('epsilon', 1.0)
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)
        self.batch_size = kwargs.get('batch_size', 32)
        self.gamma = kwargs.get('gamma', 0.99)
        self.tau = kwargs.get('tau', 0.005)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Optimized CNN DQN using device: {self.device}")
        
        self.memory = deque(maxlen=kwargs.get('memory_size', 10000))
        
        # Networks
        self.q_network = OptimizedCNNDQN(board_height, board_width).to(self.device)
        self.target_network = OptimizedCNNDQN(board_height, board_width).to(self.device)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.hard_update()
        
        print(f"Model parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
    
    def prepare_state(self, own_board: np.ndarray, opponent_board: np.ndarray, powerups: Dict[str, bool]) -> torch.Tensor:
        """Convert dual boards + powerups to 6-channel tensor"""
        own_board_channel = own_board.astype(np.float32)
        opponent_board_channel = opponent_board.astype(np.float32)
        bottom_clear_channel = np.full_like(own_board, 1.0 if powerups.get('bottom_clear', False) else 0.0, dtype=np.float32)
        gravity_channel = np.full_like(own_board, 1.0 if powerups.get('gravity', False) else 0.0, dtype=np.float32)
        bomb_channel = np.full_like(own_board, 1.0 if powerups.get('bomb', False) else 0.0, dtype=np.float32)
        wildblocks_channel = np.full_like(own_board, 1.0 if powerups.get('wildblocks', False) else 0.0, dtype=np.float32)
        
        state = np.stack([own_board_channel, opponent_board_channel, bottom_clear_channel, 
                         gravity_channel, bomb_channel, wildblocks_channel])
        return torch.FloatTensor(state).to(self.device)
    
    def find_surface_blocks(self, board: np.ndarray) -> Dict[int, Optional[int]]:
        """Find surface block (topmost block) in each column"""
        surface_blocks = {}
        
        for col in range(self.board_width):
            surface_row = None
            for row in range(self.board_height):
                if board[row, col] == 1:  # Found first block from top
                    surface_row = row
                    break
            surface_blocks[col] = surface_row
        
        return surface_blocks
    
    def find_wildblock_surface(self, opponent_board: np.ndarray, center_col: int) -> int:
        """Find where 3×3 WildBlocks will land when centered at center_col"""
        left_col = center_col - 1
        right_col = center_col + 1
        
        # Find surface blocks in all three columns
        surfaces = []
        for col in [left_col, center_col, right_col]:
            if 0 <= col < self.board_width:
                for row in range(self.board_height):
                    if opponent_board[row, col] == 1:
                        surfaces.append(row)
                        break
        
        if not surfaces:
            return -1  # No surface blocks
        
        # Return the highest surface (minimum row number)
        return min(surfaces)
    
    def find_valid_wildblock_columns(self, opponent_board: np.ndarray) -> List[int]:
        """Find valid center columns for 3×3 WildBlocks placement"""
        valid_columns = []
        
        for center_col in range(1, 9):  # Columns 1-8
            surface_row = self.find_wildblock_surface(opponent_board, center_col)
            if surface_row != -1 and surface_row >= 1:  # Has surface and some space
                valid_columns.append(center_col)
        
        return valid_columns
    
    def predict_unity(self, own_board: np.ndarray, opponent_board: np.ndarray, powerups: Dict[str, bool]) -> Dict:
        """MAIN METHOD FOR UNITY: Dual board prediction"""
        
        # Prepare input
        state = self.prepare_state(own_board, opponent_board, powerups).unsqueeze(0)
        
        # Single forward pass
        with torch.no_grad():
            output = self.q_network(state).cpu().numpy()[0]  # Shape: (23,)
        
        # Split output
        action_q = output[:5]       # Actions: [none, bottom_clear, gravity, bomb, wildblocks]
        bomb_col_q = output[5:15]   # Bomb columns
        wild_col_q = output[15:23]  # WildBlock columns
        
        # Find surface blocks and valid columns
        surface_blocks = self.find_surface_blocks(own_board)
        valid_bomb_columns = [col for col, row in surface_blocks.items() if row is not None]
        valid_wild_columns = self.find_valid_wildblock_columns(opponent_board)
        
        # LOG: Show available powerups and valid placements
        available_powerups = [k for k, v in powerups.items() if v]
        print(f"PREDICT: Available powerups: {available_powerups}")
        print(f"PREDICT: Valid bomb columns: {valid_bomb_columns}")
        print(f"PREDICT: Valid wildblock columns: {valid_wild_columns}")
        
        # Mask invalid actions
        masked_action_q = self._mask_actions(action_q, powerups, valid_bomb_columns, valid_wild_columns)
        
        # Select best action
        best_action_id = np.argmax(masked_action_q)
        action_names = ['none', 'bottom_clear', 'gravity', 'bomb', 'wildblocks']
        action_name = action_names[best_action_id]
        
        # Calculate confidence
        action_probs = self._softmax(masked_action_q)
        confidence = action_probs[best_action_id]
        
        result = {
            'action_type': int(best_action_id),
            'action_name': action_name,
            'confidence': float(confidence),
            'valid_bomb_columns': valid_bomb_columns,
            'valid_wild_columns': valid_wild_columns
        }
        
        # Handle bomb placement (on own board)
        if best_action_id == 3:  # bomb action
            if valid_bomb_columns:
                masked_bomb_col_q = self._mask_bomb_columns(bomb_col_q, valid_bomb_columns)
                best_col = np.argmax(masked_bomb_col_q)
                bomb_row = surface_blocks[best_col] if best_col in surface_blocks else 0
                
                result.update({
                    'bomb_column': int(best_col),
                    'bomb_row': int(bomb_row) if bomb_row is not None else -1,
                    'bomb_confidence': float(self._softmax(masked_bomb_col_q)[best_col])
                })
                print(f"PREDICT: Bomb selected -> Column {best_col}, Row {bomb_row}")
            else:
                result.update({'bomb_column': -1, 'bomb_row': -1, 'bomb_confidence': 0.0})
        
        # Handle wildblocks placement (on opponent board)
        elif best_action_id == 4:  # wildblocks action
            if valid_wild_columns:
                masked_wild_col_q = self._mask_wildblock_columns(wild_col_q, valid_wild_columns)
                best_wild_col_idx = np.argmax(masked_wild_col_q)
                actual_column = best_wild_col_idx + 1  # Convert to actual column (1-8)
                wild_row = self.find_wildblock_surface(opponent_board, actual_column)
                
                result.update({
                    'wildblock_column': int(actual_column),
                    'wildblock_row': int(wild_row),
                    'wildblock_confidence': float(self._softmax(masked_wild_col_q)[best_wild_col_idx])
                })
                print(f"PREDICT: WildBlocks selected -> Column {actual_column}, Row {wild_row}")
            else:
                result.update({'wildblock_column': -1, 'wildblock_row': -1, 'wildblock_confidence': 0.0})
        
        else:
            # No bomb or wildblocks
            result.update({
                'bomb_column': -1, 'bomb_row': -1, 'bomb_confidence': 0.0,
                'wildblock_column': -1, 'wildblock_row': -1, 'wildblock_confidence': 0.0
            })
        
        print(f"PREDICT: Final decision -> {action_name} (confidence: {confidence:.3f})")
        return result
    
    def _mask_actions(self, action_q: np.ndarray, powerups: Dict[str, bool], 
                     valid_bomb_columns: List[int], valid_wild_columns: List[int]) -> np.ndarray:
        """Mask invalid actions"""
        masked = action_q.copy()
        
        # none (0) always valid
        if not powerups.get('bottom_clear', False):
            masked[1] = -np.inf
        if not powerups.get('gravity', False):
            masked[2] = -np.inf
        if not powerups.get('bomb', False) or len(valid_bomb_columns) == 0:
            masked[3] = -np.inf
        if not powerups.get('wildblocks', False) or len(valid_wild_columns) == 0:
            masked[4] = -np.inf
        
        return masked
    
    def _mask_bomb_columns(self, bomb_col_q: np.ndarray, valid_columns: List[int]) -> np.ndarray:
        """Mask invalid bomb columns"""
        masked = bomb_col_q.copy()
        
        for col in range(len(bomb_col_q)):
            if col not in valid_columns:
                masked[col] = -np.inf
        
        return masked
    
    def _mask_wildblock_columns(self, wild_col_q: np.ndarray, valid_columns: List[int]) -> np.ndarray:
        """Mask invalid wildblock columns"""
        masked = wild_col_q.copy()
        
        for i in range(len(wild_col_q)):
            actual_col = i + 1  # Convert index to actual column (1-8)
            if actual_col not in valid_columns:
                masked[i] = -np.inf
        
        return masked
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities"""
        valid_mask = x != -np.inf
        if not np.any(valid_mask):
            return np.ones_like(x) / len(x)
        
        x_valid = x[valid_mask]
        exp_x = np.exp(x_valid - np.max(x_valid))
        probs = np.zeros_like(x)
        probs[valid_mask] = exp_x / np.sum(exp_x)
        
        return probs
    
    def calculate_bomb_impact(self, board: np.ndarray, bomb_row: int, bomb_col: int) -> int:
        """Calculate how many blocks would be destroyed by bomb at position"""
        blocks_destroyed = 0
        
        # 3x3 area around bomb
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r, c = bomb_row + dr, bomb_col + dc
                if 0 <= r < self.board_height and 0 <= c < self.board_width:
                    if board[r, c] == 1:
                        blocks_destroyed += 1
        
        return blocks_destroyed
    
    def choose_action_training(self, own_board: np.ndarray, opponent_board: np.ndarray, powerups: Dict[str, bool]) -> Dict:
        """Training version with epsilon-greedy exploration"""
        if np.random.random() <= self.epsilon:
            return self._random_action(own_board, opponent_board, powerups)
        else:
            return self.predict_unity(own_board, opponent_board, powerups)
    
    def _random_action(self, own_board: np.ndarray, opponent_board: np.ndarray, powerups: Dict[str, bool]) -> Dict:
        """Random valid action for training"""
        surface_blocks = self.find_surface_blocks(own_board)
        valid_bomb_columns = [col for col, row in surface_blocks.items() if row is not None]
        valid_wild_columns = self.find_valid_wildblock_columns(opponent_board)
        
        valid_actions = [0]  # none always valid
        if powerups.get('bottom_clear', False):
            valid_actions.append(1)
        if powerups.get('gravity', False):
            valid_actions.append(2)
        if powerups.get('bomb', False) and len(valid_bomb_columns) > 0:
            valid_actions.append(3)
        if powerups.get('wildblocks', False) and len(valid_wild_columns) > 0:
            valid_actions.append(4)
        
        action_type = random.choice(valid_actions)
        action_names = ['none', 'bottom_clear', 'gravity', 'bomb', 'wildblocks']
        
        result = {
            'action_type': action_type,
            'action_name': action_names[action_type],
            'confidence': 1.0,
            'valid_bomb_columns': valid_bomb_columns,
            'valid_wild_columns': valid_wild_columns
        }
        
        if action_type == 3:  # bomb
            bomb_col = random.choice(valid_bomb_columns)
            bomb_row = surface_blocks[bomb_col]
            
            result.update({
                'bomb_column': bomb_col,
                'bomb_row': bomb_row if bomb_row is not None else -1,
                'bomb_confidence': 1.0
            })
        elif action_type == 4:  # wildblocks
            wild_col = random.choice(valid_wild_columns)
            wild_row = self.find_wildblock_surface(opponent_board, wild_col)
            
            result.update({
                'wildblock_column': wild_col,
                'wildblock_row': wild_row,
                'wildblock_confidence': 1.0
            })
        else:
            result.update({
                'bomb_column': -1, 'bomb_row': -1, 'bomb_confidence': 0.0,
                'wildblock_column': -1, 'wildblock_row': -1, 'wildblock_confidence': 0.0
            })
        
        return result
    
    def remember(self, own_board: np.ndarray, opponent_board: np.ndarray, powerups: Dict[str, bool], action: Dict, 
                 reward: float, next_own_board: np.ndarray, next_opponent_board: np.ndarray, 
                 next_powerups: Dict[str, bool], done: bool):
        """Store experience for training"""
        state = self.prepare_state(own_board, opponent_board, powerups).cpu().numpy()
        next_state = self.prepare_state(next_own_board, next_opponent_board, next_powerups).cpu().numpy()
        
        action_encoded = {
            'action_type': action['action_type'],
            'bomb_column': action.get('bomb_column', -1),
            'wildblock_column': action.get('wildblock_column', -1)
        }
        
        self.memory.append((state, action_encoded, reward, next_state, done))
    
    def train(self):
        """Train the network"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = [e[1] for e in batch]
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states)  # (batch, 23)
        
        # Action type Q-values
        action_types = torch.LongTensor([a['action_type'] for a in actions]).to(self.device)
        current_action_q = current_q[:, :5].gather(1, action_types.unsqueeze(1)).squeeze(1)
        
        # Target Q-values for actions
        with torch.no_grad():
            next_q = self.target_network(next_states)
            next_action_q = next_q[:, :5].max(1)[0]
            target_action_q = rewards + (self.gamma * next_action_q * ~dones)
        
        # Action loss
        action_loss = F.mse_loss(current_action_q, target_action_q)
        
        # Bomb column loss (only for bomb actions)
        bomb_loss = 0.0
        bomb_indices = [i for i, a in enumerate(actions) if a['action_type'] == 3 and a['bomb_column'] >= 0]
        
        if bomb_indices:
            bomb_columns = torch.LongTensor([actions[i]['bomb_column'] for i in bomb_indices]).to(self.device)
            current_bomb_q = current_q[bomb_indices, 5:15].gather(1, bomb_columns.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                next_bomb_q = next_q[bomb_indices, 5:15].max(1)[0]
                target_bomb_q = rewards[bomb_indices] + (self.gamma * next_bomb_q * ~dones[bomb_indices])
            
            bomb_loss = F.mse_loss(current_bomb_q, target_bomb_q)
        
        # WildBlocks column loss (only for wildblocks actions)
        wild_loss = 0.0
        wild_indices = [i for i, a in enumerate(actions) if a['action_type'] == 4 and a['wildblock_column'] >= 1]
        
        if wild_indices:
            wild_columns = torch.LongTensor([actions[i]['wildblock_column'] - 1 for i in wild_indices]).to(self.device)
            current_wild_q = current_q[wild_indices, 15:23].gather(1, wild_columns.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                next_wild_q = next_q[wild_indices, 15:23].max(1)[0]
                target_wild_q = rewards[wild_indices] + (self.gamma * next_wild_q * ~dones[wild_indices])
            
            wild_loss = F.mse_loss(current_wild_q, target_wild_q)
        
        # Total loss
        total_loss = action_loss + bomb_loss + wild_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Soft update
        self.soft_update()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return total_loss.item()
    
    def hard_update(self):
        """Hard update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def soft_update(self):
        """Soft update target network"""
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def save_model(self, filepath: str):
        """Save model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.q_network.state_dict(),
            'board_height': self.board_height,
            'board_width': self.board_width,
            'model_type': 'optimized_surface_bomb_wildblocks'
        }
        
        torch.save(checkpoint, filepath)
        print(f"Optimized WildBlocks model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.epsilon = 0.0
        print(f"Optimized WildBlocks model loaded from {filepath}")
    
    def set_eval_mode(self):
        """Set to evaluation mode"""
        self.q_network.eval()
        self.epsilon = 0.0


# Updated Training Environment to handle WildBlocks
class OptimizedTrainingEnvironment:
    """Your original training environment + WildBlocks support"""
    
    def __init__(self, dataset_path: str):
        import pickle
        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)
        
        from feature_extractor import UniversalFeatureExtractor
        self.feature_extractor = UniversalFeatureExtractor()
        
        self.current_own_board = None
        self.current_opponent_board = None
        self.current_powerups = None
        
        print(f"Loaded dataset with {len(self.dataset)} board configurations")
    
    def reset(self):
        """Load random boards"""
        sample = random.choice(self.dataset)
        
        if isinstance(sample, dict):
            board_data = np.array(sample['board'], dtype=np.int32)
            if 'powerups' in sample:
                self.current_powerups = sample['powerups'].copy()
            else:
                self.current_powerups = {
                    'bottom_clear': random.choice([True, False]),
                    'gravity': random.choice([True, False]),
                    'bomb': random.choice([True, False]),
                    'wildblocks': random.choice([True, False])
                }
        else:
            board_data = np.array(sample, dtype=np.int32)
            self.current_powerups = {
                'bottom_clear': random.choice([True, False]),
                'gravity': random.choice([True, False]),
                'bomb': random.choice([True, False]),
                'wildblocks': random.choice([True, False])
            }
        
        # Use same board for both (for training)
        self.current_own_board = board_data.copy()
        self.current_opponent_board = board_data.copy()
        
        # Ensure at least one powerup is available
        if not any(self.current_powerups.values()):
            powerup_to_enable = random.choice(['bottom_clear', 'gravity', 'bomb', 'wildblocks'])
            self.current_powerups[powerup_to_enable] = True
        
        # LOG: Show powerup availability
        available_powerups = [k for k, v in self.current_powerups.items() if v]
        print(f"RESET: Powerups available: {available_powerups}")
        
        return self.current_own_board, self.current_opponent_board
    
    def get_state(self):
        """Get current state"""
        return self.current_own_board, self.current_opponent_board, self.current_powerups
    
    def apply_powerup(self, action: Dict):
        """Apply powerup action and return new states + reward"""
        old_own_board = self.current_own_board.copy()
        old_opponent_board = self.current_opponent_board.copy()
        
        new_own_board = old_own_board.copy()
        new_opponent_board = old_opponent_board.copy()
        
        action_type = action.get('action_name', 'none')
        
        print(f"APPLY: Action {action_type} being applied...")
        
        if action_type == 'bottom_clear':
            new_own_board[-1, :] = 0
            self.current_powerups['bottom_clear'] = False
            
        elif action_type == 'gravity':
            new_own_board = self._apply_gravity(new_own_board)
            self.current_powerups['gravity'] = False
            
        elif action_type == 'bomb':
            if action.get('bomb_row', -1) != -1 and action.get('bomb_column', -1) != -1:
                new_own_board = self._apply_bomb(new_own_board, action['bomb_row'], action['bomb_column'])
                self.current_powerups['bomb'] = False
                print(f"APPLY: Bomb applied at ({action['bomb_row']}, {action['bomb_column']})")
        
        elif action_type == 'wildblocks':
            if action.get('wildblock_row', -1) != -1 and action.get('wildblock_column', -1) != -1:
                new_opponent_board = self._apply_wildblocks(new_opponent_board, 
                                                          action['wildblock_row'], 
                                                          action['wildblock_column'])
                self.current_powerups['wildblocks'] = False
        elif action_type == 'wildblocks':
            if action.get('wildblock_row', -1) != -1 and action.get('wildblock_column', -1) != -1:
                new_opponent_board = self._apply_wildblocks(new_opponent_board, 
                                                          action['wildblock_row'], 
                                                          action['wildblock_column'])
                self.current_powerups['wildblocks'] = False
                print(f"APPLY: WildBlocks applied at ({action['wildblock_row']}, {action['wildblock_column']})")
        
        # Update current state
        self.current_own_board = new_own_board
        self.current_opponent_board = new_opponent_board
        
        # Calculate reward
        reward = self._calculate_reward(old_own_board, new_own_board, 
                                      old_opponent_board, new_opponent_board, action)
        
        print(f"APPLY: Reward calculated: {reward:.2f}")
        
        return new_own_board, new_opponent_board, reward
    
    def _apply_gravity(self, board: np.ndarray) -> np.ndarray:
        """Apply gravity effect - move blocks down to fill holes"""
        new_board = board.copy()
        rows, cols = new_board.shape
        
        for col in range(cols):
            # Extract all blocks in this column
            blocks = []
            for row in range(rows-1, -1, -1):  # Bottom to top
                if new_board[row, col] == 1:
                    blocks.append(1)
                    new_board[row, col] = 0
            
            # Place blocks at bottom
            for i, block in enumerate(blocks):
                new_board[rows-1-i, col] = block
        
        return new_board
    
    def _apply_bomb(self, board: np.ndarray, bomb_row: int, bomb_col: int) -> np.ndarray:
        """Apply bomb effect - clear 3x3 area"""
        new_board = board.copy()
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r, c = bomb_row + dr, bomb_col + dc
                if 0 <= r < board.shape[0] and 0 <= c < board.shape[1]:
                    new_board[r, c] = 0
        
        return new_board
    
    def _apply_wildblocks(self, board: np.ndarray, surface_row: int, center_col: int) -> np.ndarray:
        """Apply wildblocks effect - place 3x3 block on opponent board"""
        new_board = board.copy()
        
        # Place 3×3 block starting from surface_row going up
        for dr in range(3):  # 3 rows
            for dc in range(-1, 2):  # 3 columns: -1, 0, +1
                place_row = surface_row - dr  # Go upward from surface
                place_col = center_col + dc
                
                if 0 <= place_row < board.shape[0] and 0 <= place_col < board.shape[1]:
                    new_board[place_row, place_col] = 1
        
        return new_board
    
    def _calculate_reward(self, old_own: np.ndarray, new_own: np.ndarray,
                         old_opp: np.ndarray, new_opp: np.ndarray, action: Dict) -> float:
        """Calculate reward - using your original reward structure + wildblocks"""
        
        action_type = action.get('action_name', 'none')
        
        if action_type == 'bottom_clear':
            bottom_blocks = np.sum(old_own[-1, :])
            reward = 4.0 + bottom_blocks * 0.4
            
        elif action_type == 'gravity':
            blocks_removed = np.sum(old_own) - np.sum(new_own)
            reward = 3.0 + blocks_removed * 0.3
            
        elif action_type == 'bomb':
            blocks_destroyed = np.sum(old_own) - np.sum(new_own)
            reward = 5.0 + blocks_destroyed * 0.5
            
        elif action_type == 'wildblocks':
            # Calculate opponent damage (similar to bomb reward structure)
            opponent_damage = self._evaluate_opponent_damage(old_opp, new_opp)
            reward = 5.0 + opponent_damage * 0.5
            
        elif action_type == 'none':
            reward = -0.5  # Same as your original
            
        return np.clip(reward, -5, 20)
    
    def _evaluate_opponent_damage(self, old_board: np.ndarray, new_board: np.ndarray) -> float:
        """Evaluate damage dealt to opponent board"""
        old_holes = self._count_holes(old_board)
        new_holes = self._count_holes(new_board)
        holes_increase = new_holes - old_holes
        
        old_bumpiness = self._calculate_bumpiness(old_board)
        new_bumpiness = self._calculate_bumpiness(new_board)
        bumpiness_increase = new_bumpiness - old_bumpiness
        
        damage = holes_increase * 3.0 + bumpiness_increase * 1.0
        return damage
    
    def _count_holes(self, board: np.ndarray) -> int:
        """Count holes in board"""
        holes = 0
        for col in range(board.shape[1]):
            found_block = False
            for row in range(board.shape[0]):
                if board[row, col] == 1:
                    found_block = True
                elif found_block and board[row, col] == 0:
                    holes += 1
        return holes
    
    def _calculate_bumpiness(self, board: np.ndarray) -> int:
        """Calculate bumpiness"""
        heights = []
        for col in range(board.shape[1]):
            height = 0
            for row in range(board.shape[0]):
                if board[row, col] == 1:
                    height = board.shape[0] - row
                    break
            heights.append(height)
        
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        
        return bumpiness


# Updated trainer with minimal changes to your original
class OptimizedBombTrainer:
    """Trainer for optimized surface-bomb model + WildBlocks"""
    
    def __init__(self, dataset_path: str, save_dir: str = "optimized_models"):
        self.environment = OptimizedTrainingEnvironment(dataset_path)
        self.agent = OptimizedBombAgent(
            learning_rate=0.0001,
            epsilon=1.0,
            epsilon_min=0.02,
            epsilon_decay=0.9995,
            memory_size=20000,
            batch_size=32
        )
        
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.episode_rewards = []
        self.action_usage = {'none': 0, 'bottom_clear': 0, 'gravity': 0, 'bomb': 0, 'wildblocks': 0}
        self.bomb_column_usage = [0] * 10  # Track which columns are bombed
        self.wildblock_column_usage = [0] * 8  # Track wildblock column usage (1-8)

        ## visualization code 
        self.visualizer = TrainingVisualizer()
        self.logger = TrainingLogger(self.visualizer)
    
    def enhanced_reward_function(self, old_own_board: np.ndarray, new_own_board: np.ndarray, 
                                old_opp_board: np.ndarray, new_opp_board: np.ndarray, action: Dict) -> float:
        """Enhanced reward function - keeping your original structure + wildblocks"""
        
        action_type = action.get('action_name', 'none')
        
        if action_type == 'bomb':
            blocks_removed = np.sum(old_own_board) - np.sum(new_own_board)
            base_reward = 5.0 + blocks_removed * 0.5
            
            # Bonus for strategic bomb placement (your original logic)
            if action.get('bomb_row', -1) != -1:
                bomb_impact = self.agent.calculate_bomb_impact(old_own_board, action['bomb_row'], action['bomb_column'])
                efficiency_bonus = bomb_impact * 0.3
                column_blocks = np.sum(old_own_board[:, action['bomb_column']])
                column_bonus = column_blocks * 0.1
                reward = base_reward + efficiency_bonus + column_bonus
            else:
                reward = base_reward
                
        elif action_type == 'bottom_clear':
            bottom_blocks = np.sum(old_own_board[-1, :])
            reward = 4.0 + bottom_blocks * 0.4
            
        elif action_type == 'gravity':
            blocks_removed = np.sum(old_own_board) - np.sum(new_own_board)
            reward = 3.0 + blocks_removed * 0.3
            
        elif action_type == 'wildblocks':
            # Calculate opponent damage like bomb logic
            opponent_damage = self.environment._evaluate_opponent_damage(old_opp_board, new_opp_board)
            base_reward = 5.0 + opponent_damage * 0.5
            
            # Bonus for strategic wildblock placement
            if action.get('wildblock_row', -1) != -1:
                strategic_bonus = opponent_damage * 0.3
                reward = base_reward + strategic_bonus
            else:
                reward = base_reward
            
        else:  # 'none'
            reward = -0.5  # Your original
        
        return np.clip(reward, -5, 20)
    
    def train(self, episodes: int = 5000):
        """Train optimized model with WildBlocks"""
        print(f"Training optimized surface-bomb + WildBlocks model for {episodes} episodes...")
        
        for episode in range(episodes):
            own_board, opponent_board = self.environment.reset()
            episode_reward = 0
            
            for step in range(8):
                current_own, current_opponent, current_powerups = self.environment.get_state()
                
                # LOG: Show available powerups before choosing action
                available_powerups = [k for k, v in current_powerups.items() if v]
                if episode % 100 == 0 and step == 0:
                    print(f"TRAIN Episode {episode}: Available powerups: {available_powerups}")
                
                # Choose action
                action = self.agent.choose_action_training(current_own, current_opponent, current_powerups)
                
                # Apply action
                old_own_board = current_own.copy()
                old_opponent_board = current_opponent.copy()
                
                new_own, new_opponent, _ = self.environment.apply_powerup(action)
                new_own_state, new_opponent_state, new_powerups = self.environment.get_state()
                
                # Calculate reward using enhanced function
                reward = self.enhanced_reward_function(old_own_board, new_own_state, 
                                                     old_opponent_board, new_opponent_state, action)
                done = not any(new_powerups.values())
                
                # Store experience
                self.agent.remember(current_own, current_opponent, current_powerups, action,
                                  reward, new_own_state, new_opponent_state, new_powerups, done)
                
                episode_reward += reward
                self.action_usage[action['action_name']] += 1
                
                # Track column usage
                if action['action_name'] == 'bomb' and action.get('bomb_column', -1) >= 0:
                    self.bomb_column_usage[action['bomb_column']] += 1
                elif action['action_name'] == 'wildblocks' and action.get('wildblock_column', -1) >= 1:
                    col_idx = action['wildblock_column'] - 1  # Convert to 0-7 index
                    if 0 <= col_idx < 8:
                        self.wildblock_column_usage[col_idx] += 1
                
                if done:
                    break
            
            # Train
            if len(self.agent.memory) > self.agent.batch_size:
                loss = self.agent.train()

                # Log metrics for visualization
                self.logger.log_episode(
                    episode=episode,
                    episode_reward=episode_reward,
                    loss=loss,
                    action_usage=self.action_usage,
                    epsilon=self.agent.epsilon,
                    bomb_column_usage=self.bomb_column_usage
                )
            
            self.episode_rewards.append(episode_reward)
            
            # Enhanced logging
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                total_actions = sum(self.action_usage.values())
                action_dist = {k: (v/total_actions)*100 for k, v in self.action_usage.items()}
                
                print(f"Episode {episode}: Avg Reward: {avg_reward:.2f}")
                print(f"  Actions: {action_dist}")
                print(f"  Epsilon: {self.agent.epsilon:.3f}")
                
                # Show column preferences
                total_bombs = sum(self.bomb_column_usage)
                if total_bombs > 0:
                    bomb_prefs = [f"Col{i}:{(count/total_bombs)*100:.1f}%" 
                                 for i, count in enumerate(self.bomb_column_usage) if count > 0]
                    print(f"  Bomb columns: {bomb_prefs[:5]}")
                
                total_wildblocks = sum(self.wildblock_column_usage)
                if total_wildblocks > 0:
                    wild_prefs = [f"Col{i+1}:{(count/total_wildblocks)*100:.1f}%" 
                                 for i, count in enumerate(self.wildblock_column_usage) if count > 0]
                    print(f"  WildBlock columns: {wild_prefs}")
            
            # Save periodically
            if episode % 500 == 0 and episode > 0:
                model_path = os.path.join(self.save_dir, f"optimized_wildblocks_model_ep{episode}.pth")
                self.agent.save_model(model_path)
        
        # Final save
        final_path = os.path.join(self.save_dir, "optimized_wildblocks_model_final.pth")
        self.agent.save_model(final_path)

        # Final Visualization dashboard
        self.visualizer.create_training_dashboard("final_wildblocks_training_dashboard.png")
        self.visualizer.plot_bomb_column_analysis("final_wildblocks_analysis.png")
        
        return final_path
    
    def export_for_unity(self, model_path: str):
        """Export trained model for Unity"""
        self.agent.load_model(model_path)
        
        # ONNX export
        onnx_path = model_path.replace('.pth', '.onnx')
        
        dummy_input = torch.randn(1, 6, self.agent.board_height, self.agent.board_width).to(self.agent.device)
        
        torch.onnx.export(
            self.agent.q_network,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=['dual_board_state'],
            output_names=['action_and_position_q_values']
        )
        
        print(f"Unity WildBlocks model exported: {onnx_path}")
        print("Unity integration info:")
        print("- Input: (1, 6, 20, 10) tensor = 1200 floats")
        print("- Output: (1, 23) tensor = 23 floats")
        print("- Actions: [none, bottom_clear, gravity, bomb, wildblocks] (5)")
        print("- Bomb columns: [col0, col1, ..., col9] (10)")
        print("- WildBlock columns: [col1, col2, ..., col8] (8)")
        
        return onnx_path


# Usage example
if __name__ == "__main__":
    # Train model
    trainer = OptimizedBombTrainer("tetris_boards.pkl")
    model_path = trainer.train(episodes=3000)
    
    # Export for Unity
    onnx_path = trainer.export_for_unity(model_path)
    
    print(f"\nWildBlocks training complete!")
    print(f"PyTorch model: {model_path}")
    print(f"Unity ONNX model: {onnx_path}")
    
    # Demo prediction
    agent = OptimizedBombAgent()
    agent.load_model(model_path)
    
    # Create test boards
    own_test_board = np.zeros((20, 10))
    own_test_board[15:, [2, 5, 7]] = 1  # Own board
    
    opponent_test_board = np.zeros((20, 10))
    opponent_test_board[12:, [1, 3, 6, 8]] = 1  # Opponent board
    
    test_powerups = {'bottom_clear': True, 'gravity': False, 'bomb': True, 'wildblocks': True}
    
    result = agent.predict_unity(own_test_board, opponent_test_board, test_powerups)
    print(f"\nDemo WildBlocks prediction: {result}")