# optimized_cnn_dqn_wildblock_agent.py - Multi-board with wildblock support
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np
import random
import os
from typing import Dict, List, Tuple, Optional
from powerup_training_visualizer2 import EnhancedTrainingVisualizer, EnhancedTrainingLogger

class OptimizedCNNDQNWildblock(nn.Module):
    """
    Optimized CNN with wildblock support for dual-board input
    Input: 8 channels (self_board + opponent_board + powerups)
    Output: 5 action types + 10 bomb columns + 8 wildblock columns = 23 total outputs
    """
    
    def __init__(self, board_height=20, board_width=10):
        super(OptimizedCNNDQNWildblock, self).__init__()
        
        self.board_height = board_height
        self.board_width = board_width
        
        # Shared convolutional backbone (8 input channels for dual boards)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Global features for action type
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Action type branch (5 outputs: none, bottom_clear, gravity, bomb, wildblock)
        self.action_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )
        
        # Bomb column branch (10 outputs: single cell placement, any column)
        self.bomb_column_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 10)),  # Pool to (1, 10) - one value per column
            nn.Flatten(),
            nn.Linear(256 * 10, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # Q-value for bombing each column (0-9)
        )
        
        # Wildblock column branch (8 outputs: columns 1-8 only, since 3x3 needs center space)
        self.wildblock_column_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 8)),  # Pool to (1, 8) - valid wildblock columns
            nn.Flatten(),
            nn.Linear(256 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8)  # Q-value for wildblock placement in each valid column (1-8)
        )
        
    def forward(self, x):
        # Shared features from conv layers
        features = self.conv_layers(x)  # (batch, 256, 20, 10)
        
        # Action type Q-values
        pooled_features = self.global_pool(features)
        action_q = self.action_branch(pooled_features)  # (batch, 5)
        
        # Bomb column Q-values (all 10 columns)
        bomb_col_q = self.bomb_column_branch(features)  # (batch, 10)
        
        # Wildblock column Q-values (columns 1-8 only for 3x3 placement)
        valid_features = features[:, :, :, 1:9]  # (batch, 256, 20, 8)
        wildblock_col_q = self.wildblock_column_branch(valid_features)  # (batch, 8)
        
        # Concatenate outputs: [action_q, bomb_col_q, wildblock_col_q]
        # Output shape: (batch, 23) = 5 actions + 10 bomb columns + 8 wildblock columns
        output = torch.cat([action_q, bomb_col_q, wildblock_col_q], dim=1)
        
        return output


class OptimizedWildblockAgent:
    """
    Optimized agent with wildblock support for dual-board gameplay
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
        print(f"Optimized CNN DQN with Wildblock using device: {self.device}")
        
        self.memory = deque(maxlen=kwargs.get('memory_size', 10000))
        
        # Networks
        self.q_network = OptimizedCNNDQNWildblock(board_height, board_width).to(self.device)
        self.target_network = OptimizedCNNDQNWildblock(board_height, board_width).to(self.device)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.hard_update()
        
        print(f"Model parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
    
    def prepare_dual_state(self, self_board: np.ndarray, opponent_board: np.ndarray, 
                          powerups: Dict[str, bool]) -> torch.Tensor:
        """Convert dual boards + powerups to 8-channel tensor"""
        # Board channels
        self_board_channel = self_board.astype(np.float32)
        opponent_board_channel = opponent_board.astype(np.float32)
        
        # Powerup channels (applied to both boards for context)
        bottom_clear_channel = np.full_like(self_board, 1.0 if powerups.get('bottom_clear', False) else 0.0, dtype=np.float32)
        gravity_channel = np.full_like(self_board, 1.0 if powerups.get('gravity', False) else 0.0, dtype=np.float32)
        bomb_channel = np.full_like(self_board, 1.0 if powerups.get('bomb', False) else 0.0, dtype=np.float32)
        wildblock_channel = np.full_like(self_board, 1.0 if powerups.get('wildblock', False) else 0.0, dtype=np.float32)
        
        # Additional context channels
        height_diff_channel = self._calculate_height_advantage(self_board, opponent_board)
        threat_level_channel = self._calculate_threat_level(opponent_board)
        
        state = np.stack([
            self_board_channel, opponent_board_channel, 
            bottom_clear_channel, gravity_channel, bomb_channel, wildblock_channel,
            height_diff_channel, threat_level_channel
        ])
        
        return torch.FloatTensor(state).to(self.device)
    
    def _calculate_height_advantage(self, self_board: np.ndarray, opponent_board: np.ndarray) -> np.ndarray:
        """Calculate height advantage map"""
        self_heights = self._get_column_heights(self_board)
        opp_heights = self._get_column_heights(opponent_board)
        
        # Positive values = we have advantage, negative = opponent has advantage
        height_diff = opp_heights - self_heights
        height_diff_normalized = np.tanh(height_diff / 10.0)  # Normalize to [-1, 1]
        
        return np.broadcast_to(height_diff_normalized.reshape(1, -1), self_board.shape).astype(np.float32)
    
    def _calculate_threat_level(self, opponent_board: np.ndarray) -> np.ndarray:
        """Calculate threat level based on opponent board state"""
        heights = self._get_column_heights(opponent_board)
        max_height = np.max(heights)
        
        # Higher threat when opponent has tall columns (closer to losing)
        threat_level = max_height / self.board_height
        
        return np.full_like(opponent_board, threat_level, dtype=np.float32)
    
    def _get_column_heights(self, board: np.ndarray) -> np.ndarray:
        """Get height of each column"""
        heights = np.zeros(self.board_width)
        for col in range(self.board_width):
            for row in range(self.board_height):
                if board[row, col] == 1:
                    heights[col] = self.board_height - row
                    break
        return heights
    
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
    
    def find_valid_wildblock_positions(self, board: np.ndarray) -> Dict[int, Optional[int]]:
        """
        Find valid positions for 3x3 wildblock placement
        Only columns 1-8 are valid (to avoid edge overflow)
        """
        valid_positions = {}
        
        for center_col in range(1, 9):  # Columns 1-8 only
            # Check surface blocks in the 3 columns that the wildblock will span
            left_col = center_col - 1
            right_col = center_col + 1
            
            # Find surface blocks in all three columns
            surface_rows = []
            for col in [left_col, center_col, right_col]:
                for row in range(self.board_height):
                    if board[row, col] == 1:
                        surface_rows.append(row)
                        break
                else:
                    surface_rows.append(self.board_height)  # Empty column
            
            # The wildblock should be placed on top of the highest surface block
            # in the three columns (to ensure it sits on top of existing blocks)
            highest_surface = min(surface_rows)  # Lowest row index = highest position
            
            # Place wildblock on top of the highest surface
            placement_row = max(0, highest_surface - 1)
            
            # Check if placement is valid (not too high)
            if placement_row >= 0 and placement_row < self.board_height - 1:
                valid_positions[center_col] = placement_row
            else:
                valid_positions[center_col] = None
        
        return valid_positions
    
    def calculate_wildblock_damage(self, board: np.ndarray, placement_row: int, placement_col: int) -> float:
        """
        Calculate damage score for placing 3x3 wildblock centered at given position
        Higher score = more damage to opponent
        """
        damage_score = 0.0
        
        # Simulate 3x3 wildblock placement
        temp_board = board.copy()
        
        # Place 3x3 wildblock centered at (placement_row, placement_col)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r, c = placement_row + dr, placement_col + dc
                if 0 <= r < self.board_height and 0 <= c < self.board_width:
                    temp_board[r, c] = 1
        
        # Calculate damage metrics
        
        # 1. Height increase (primary factor)
        original_heights = self._get_column_heights(board)
        new_heights = self._get_column_heights(temp_board)
        height_increase = np.sum(new_heights - original_heights)
        damage_score += height_increase * 2.0
        
        # 2. Maximum height penalty (dangerous for opponent)
        max_height = np.max(new_heights)
        if max_height > 15:  # Close to losing
            damage_score += (max_height - 15) * 5.0
        
        # 3. Surface roughness (harder to clear lines)
        height_variance = np.var(new_heights)
        damage_score += height_variance * 0.8
        
        # 4. Hole creation above existing blocks
        holes_created = 0
        for col in range(max(0, placement_col-1), min(self.board_width, placement_col+2)):
            for row in range(placement_row + 1, self.board_height-1):
                if temp_board[row, col] == 0 and temp_board[row+1, col] == 1:
                    holes_created += 1
        damage_score += holes_created * 2.0
        
        # 5. Line blocking potential (blocks that prevent line clearing)
        blocked_lines = 0
        for row in range(max(0, placement_row-1), min(self.board_height, placement_row+2)):
            blocks_in_row = np.sum(temp_board[row, :])
            if 7 <= blocks_in_row < 10:  # Almost full lines that are now harder to clear
                blocked_lines += 1
        damage_score += blocked_lines * 3.0
        
        # 6. Strategic placement bonus (targeting center columns)
        center_bonus = 0.0
        if 3 <= placement_col <= 6:  # Center columns are more disruptive
            center_bonus = 1.0
        damage_score += center_bonus
        
        return damage_score
    
    def calculate_bomb_impact(self, board: np.ndarray, bomb_row: int, bomb_col: int) -> int:
        """Calculate how many blocks would be destroyed by bomb at position (unchanged)"""
        blocks_destroyed = 0
        
        # 3x3 area around bomb
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r, c = bomb_row + dr, bomb_col + dc
                if 0 <= r < self.board_height and 0 <= c < self.board_width:
                    if board[r, c] == 1:
                        blocks_destroyed += 1
        
        return blocks_destroyed
    
    def predict_unity(self, self_board: np.ndarray, opponent_board: np.ndarray, 
                     powerups: Dict[str, bool]) -> Dict:
        """
        MAIN METHOD FOR UNITY: Optimized dual-board prediction with wildblock
        
        Returns:
            {
                'action_type': 0-4,
                'action_name': string,
                'bomb_column': 0-9 (if bomb selected),
                'bomb_row': actual row (if bomb selected),
                'wildblock_column': 1-8 (if wildblock selected),
                'wildblock_row': actual row (if wildblock selected),
                'confidence': float,
                'valid_bomb_columns': list,
                'valid_wildblock_columns': list
            }
        """
        
        # Prepare input
        state = self.prepare_dual_state(self_board, opponent_board, powerups).unsqueeze(0)
        
        # Single forward pass
        with torch.no_grad():
            output = self.q_network(state).cpu().numpy()[0]  # Shape: (23,)
        
        # Split output
        action_q = output[:5]        # Action types [none, bottom_clear, gravity, bomb, wildblock]
        bomb_col_q = output[5:15]    # Bomb columns (0-9)
        wildblock_col_q = output[15:] # Wildblock columns (1-8)
        
        # Find valid columns
        self_surface_blocks = self.find_surface_blocks(self_board)
        valid_bomb_columns = [col for col, row in self_surface_blocks.items() if row is not None]
        
        opp_wildblock_positions = self.find_valid_wildblock_positions(opponent_board)
        valid_wildblock_columns = [col for col, row in opp_wildblock_positions.items() if row is not None]
        
        # Mask invalid actions
        masked_action_q = self._mask_actions(action_q, powerups, valid_bomb_columns, valid_wildblock_columns)
        
        # Select best action
        best_action_id = np.argmax(masked_action_q)
        action_names = ['none', 'bottom_clear', 'gravity', 'bomb', 'wildblock']
        action_name = action_names[best_action_id]
        
        # Calculate confidence
        action_probs = self._softmax(masked_action_q)
        confidence = action_probs[best_action_id]
        
        result = {
            'action_type': int(best_action_id),
            'action_name': action_name,
            'confidence': float(confidence),
            'valid_bomb_columns': valid_bomb_columns,
            'valid_wildblock_columns': valid_wildblock_columns
        }
        
        # Handle bomb action
        if best_action_id == 3:  # bomb action
            masked_bomb_col_q = self._mask_bomb_columns(bomb_col_q, valid_bomb_columns)
            best_bomb_col = np.argmax(masked_bomb_col_q)
            bomb_row = self_surface_blocks[best_bomb_col] if best_bomb_col in self_surface_blocks else 0
            
            result.update({
                'bomb_column': int(best_bomb_col),
                'bomb_row': int(bomb_row) if bomb_row is not None else -1,
                'bomb_confidence': float(self._softmax(masked_bomb_col_q)[best_bomb_col]),
                'wildblock_column': -1,
                'wildblock_row': -1,
                'wildblock_confidence': 0.0
            })
            
        elif best_action_id == 4:  # wildblock action
            if valid_wildblock_columns:
                # Evaluate damage for each valid position and combine with neural network output
                damage_scores = np.full(8, -np.inf)  # 8 possible columns (1-8)
                
                for i, center_col in enumerate(range(1, 9)):  # Columns 1-8
                    if center_col in valid_wildblock_columns:
                        placement_row = opp_wildblock_positions[center_col]
                        if placement_row is not None:
                            damage = self.calculate_wildblock_damage(opponent_board, placement_row, center_col)
                            damage_scores[i] = damage
                
                # Combine neural network output with damage calculation
                combined_scores = wildblock_col_q + damage_scores * 0.1  # Weight damage calculation
                masked_wildblock_q = self._mask_wildblock_columns(combined_scores, valid_wildblock_columns)
                best_wildblock_idx = np.argmax(masked_wildblock_q)
                best_wildblock_col = best_wildblock_idx + 1  # Convert back to actual column (1-8)
                wildblock_row = opp_wildblock_positions.get(best_wildblock_col, 0)
                
                result.update({
                    'bomb_column': -1,
                    'bomb_row': -1,
                    'bomb_confidence': 0.0,
                    'wildblock_column': int(best_wildblock_col),
                    'wildblock_row': int(wildblock_row) if wildblock_row is not None else -1,
                    'wildblock_confidence': float(self._softmax(masked_wildblock_q)[best_wildblock_idx]),
                    'expected_damage': float(damage_scores[best_wildblock_idx]) if best_wildblock_idx < len(damage_scores) else 0.0
                })
            else:
                result.update({
                    'bomb_column': -1,
                    'bomb_row': -1,
                    'bomb_confidence': 0.0,
                    'wildblock_column': -1,
                    'wildblock_row': -1,
                    'wildblock_confidence': 0.0,
                    'expected_damage': 0.0
                })
        else:
            result.update({
                'bomb_column': -1,
                'bomb_row': -1,
                'bomb_confidence': 0.0,
                'wildblock_column': -1,
                'wildblock_row': -1,
                'wildblock_confidence': 0.0
            })
        
        return result
    
    def _mask_actions(self, action_q: np.ndarray, powerups: Dict[str, bool], 
                     valid_bomb_columns: List[int], valid_wildblock_columns: List[int]) -> np.ndarray:
        """Mask invalid actions"""
        masked = action_q.copy()
        
        # none (0) always valid
        if not powerups.get('bottom_clear', False):
            masked[1] = -np.inf
        if not powerups.get('gravity', False):
            masked[2] = -np.inf
        if not powerups.get('bomb', False) or len(valid_bomb_columns) == 0:
            masked[3] = -np.inf
        if not powerups.get('wildblock', False) or len(valid_wildblock_columns) == 0:
            masked[4] = -np.inf
        
        return masked
    
    def _mask_bomb_columns(self, bomb_col_q: np.ndarray, valid_bomb_columns: List[int]) -> np.ndarray:
        """Mask invalid bomb columns (those without surface blocks)"""
        masked = bomb_col_q.copy()
        
        for col in range(len(bomb_col_q)):
            if col not in valid_bomb_columns:
                masked[col] = -np.inf
        
        return masked
    
    def _mask_wildblock_columns(self, wildblock_col_q: np.ndarray, valid_wildblock_columns: List[int]) -> np.ndarray:
        """Mask invalid wildblock columns"""
        masked = wildblock_col_q.copy()
        
        for i, center_col in enumerate(range(1, 9)):  # Columns 1-8
            if center_col not in valid_wildblock_columns:
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
    
    def choose_action_training(self, self_board: np.ndarray, opponent_board: np.ndarray, 
                             powerups: Dict[str, bool]) -> Dict:
        """Training version with epsilon-greedy exploration"""
        if np.random.random() <= self.epsilon:
            return self._random_action(self_board, opponent_board, powerups)
        else:
            return self.predict_unity(self_board, opponent_board, powerups)
    
    def _random_action(self, self_board: np.ndarray, opponent_board: np.ndarray, 
                      powerups: Dict[str, bool]) -> Dict:
        """Random valid action for training"""
        self_surface_blocks = self.find_surface_blocks(self_board)
        valid_bomb_columns = [col for col, row in self_surface_blocks.items() if row is not None]
        
        opp_wildblock_positions = self.find_valid_wildblock_positions(opponent_board)
        valid_wildblock_columns = [col for col, row in opp_wildblock_positions.items() if row is not None]
        
        valid_actions = [0]  # none always valid
        if powerups.get('bottom_clear', False):
            valid_actions.append(1)
        if powerups.get('gravity', False):
            valid_actions.append(2)
        if powerups.get('bomb', False) and len(valid_bomb_columns) > 0:
            valid_actions.append(3)
        if powerups.get('wildblock', False) and len(valid_wildblock_columns) > 0:
            valid_actions.append(4)
        
        action_type = random.choice(valid_actions)
        action_names = ['none', 'bottom_clear', 'gravity', 'bomb', 'wildblock']
        
        result = {
            'action_type': action_type,
            'action_name': action_names[action_type],
            'confidence': 1.0,
            'valid_bomb_columns': valid_bomb_columns,
            'valid_wildblock_columns': valid_wildblock_columns
        }
        
        if action_type == 3:  # bomb
            bomb_col = random.choice(valid_bomb_columns)
            bomb_row = self_surface_blocks[bomb_col]
            
            result.update({
                'bomb_column': bomb_col,
                'bomb_row': bomb_row if bomb_row is not None else -1,
                'bomb_confidence': 1.0,
                'wildblock_column': -1,
                'wildblock_row': -1,
                'wildblock_confidence': 0.0
            })
        elif action_type == 4:  # wildblock
            wildblock_col = random.choice(valid_wildblock_columns)
            wildblock_row = opp_wildblock_positions[wildblock_col]
            
            result.update({
                'bomb_column': -1,
                'bomb_row': -1,
                'bomb_confidence': 0.0,
                'wildblock_column': wildblock_col,
                'wildblock_row': wildblock_row if wildblock_row is not None else -1,
                'wildblock_confidence': 1.0
            })
        else:
            result.update({
                'bomb_column': -1,
                'bomb_row': -1,
                'bomb_confidence': 0.0,
                'wildblock_column': -1,
                'wildblock_row': -1,
                'wildblock_confidence': 0.0
            })
        
        return result
    
    def remember(self, self_board: np.ndarray, opponent_board: np.ndarray, powerups: Dict[str, bool], 
                action: Dict, reward: float, next_self_board: np.ndarray, next_opponent_board: np.ndarray, 
                next_powerups: Dict[str, bool], done: bool):
        """Store experience for training"""
        state = self.prepare_dual_state(self_board, opponent_board, powerups).cpu().numpy()
        next_state = self.prepare_dual_state(next_self_board, next_opponent_board, next_powerups).cpu().numpy()
        
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
        
        # Wildblock column loss (only for wildblock actions)
        wildblock_loss = 0.0
        wildblock_indices = [i for i, a in enumerate(actions) if a['action_type'] == 4 and a['wildblock_column'] >= 1]
        
        if wildblock_indices:
            # Convert wildblock columns (1-8) to indices (0-7) for tensor indexing
            wildblock_columns = torch.LongTensor([actions[i]['wildblock_column'] - 1 for i in wildblock_indices]).to(self.device)
            current_wildblock_q = current_q[wildblock_indices, 15:23].gather(1, wildblock_columns.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                next_wildblock_q = next_q[wildblock_indices, 15:23].max(1)[0]
                target_wildblock_q = rewards[wildblock_indices] + (self.gamma * next_wildblock_q * ~dones[wildblock_indices])
            
            wildblock_loss = F.mse_loss(current_wildblock_q, target_wildblock_q)
        
        # Total loss
        total_loss = action_loss + bomb_loss + wildblock_loss
        
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
            'model_type': 'optimized_wildblock_dual_board'
        }
        
        torch.save(checkpoint, filepath)
        print(f"Optimized wildblock model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.epsilon = 0.0
        print(f"Optimized wildblock model loaded from {filepath}")
    
    def set_eval_mode(self):
        """Set to evaluation mode"""
        self.q_network.eval()
        self.epsilon = 0.0


# Optimized trainer with wildblock support
class OptimizedWildblockTrainer:
    """Trainer for optimized wildblock model with dual-board support"""
    
    def __init__(self, dataset_path: str, save_dir: str = "optimized_wildblock_models"):
        from environments import TrainingEnvironment
        
        self.environment = TrainingEnvironment(dataset_path)
        self.agent = OptimizedWildblockAgent(
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
        self.action_usage = {'none': 0, 'bottom_clear': 0, 'gravity': 0, 'bomb': 0, 'wildblock': 0}
        self.bomb_column_usage = [0] * 10  # Track which columns are bombed
        self.wildblock_column_usage = [0] * 8  # Track which wildblock columns are used (1-8)

        # Visualization code 
        self.visualizer = EnhancedTrainingVisualizer()
        self.logger = EnhancedTrainingLogger(self.visualizer)
    
    def enhanced_reward_function(self, old_self_board: np.ndarray, new_self_board: np.ndarray,
                                old_opponent_board: np.ndarray, new_opponent_board: np.ndarray, 
                                action: Dict) -> float:
        """Enhanced reward function for dual-board with wildblock"""
        
        if action['action_name'] == 'wildblock':
            # Reward based on damage inflicted on opponent
            if action['wildblock_row'] != -1:
                damage_score = self.agent.calculate_wildblock_damage(
                    old_opponent_board, action['wildblock_row'], action['wildblock_column']
                )
                
                # Base reward for wildblock usage
                base_reward = 8.0
                
                # Bonus for effective damage
                damage_bonus = damage_score * 0.5
                
                # Bonus for strategic timing (when opponent is vulnerable)
                opp_heights = self.agent._get_column_heights(old_opponent_board)
                vulnerability_bonus = np.max(opp_heights) * 0.2
                
                reward = base_reward + damage_bonus + vulnerability_bonus
            else:
                reward = 2.0  # Small reward for attempting wildblock
                
        elif action['action_name'] == 'bomb':
            # Existing bomb logic (unchanged)
            blocks_removed = np.sum(old_self_board) - np.sum(new_self_board)
            base_reward = 5.0 + blocks_removed * 0.5
            
            if action['bomb_row'] != -1:
                bomb_impact = self.agent.calculate_bomb_impact(old_self_board, action['bomb_row'], action['bomb_column'])
                efficiency_bonus = bomb_impact * 0.3
                column_blocks = np.sum(old_self_board[:, action['bomb_column']])
                column_bonus = column_blocks * 0.1
                reward = base_reward + efficiency_bonus + column_bonus
            else:
                reward = base_reward
                
        elif action['action_name'] == 'bottom_clear':
            blocks_removed = np.sum(old_self_board) - np.sum(new_self_board)
            bottom_blocks = np.sum(old_self_board[-1, :])
            reward = 4.0 + bottom_blocks * 0.4 + blocks_removed * 0.2
            
        elif action['action_name'] == 'gravity':
            blocks_removed = np.sum(old_self_board) - np.sum(new_self_board)
            reward = 3.0 + blocks_removed * 0.3
            
        else:  # 'none'
            reward = -0.5
        
        return np.clip(reward, -5, 25)
    
    def train(self, episodes: int = 5000):
        """Train optimized wildblock model"""
        print(f"Training optimized wildblock model for {episodes} episodes...")
        
        for episode in range(episodes):
            self.environment.reset()
            episode_reward = 0
            
            for step in range(8):
                current_board = self.environment.get_board_state()
                current_powerups = self.environment.get_powerup_availability()
                
                # For training, use same board as both self and opponent
                # In actual gameplay, these would be different
                opponent_board = current_board.copy()
                
                # Choose action
                action = self.agent.choose_action_training(current_board, opponent_board, current_powerups)
                
                # Apply action
                old_self_board = current_board.copy()
                old_opponent_board = opponent_board.copy()
                
                # Format action for environment compatibility
                if action['action_name'] == 'bomb' and action['bomb_row'] != -1:
                    action_for_env = {
                        'type': action['action_name'],
                        'row': action['bomb_row'],
                        'col': action['bomb_column']
                    }
                elif action['action_name'] == 'wildblock' and action['wildblock_row'] != -1:
                    # For training, apply wildblock to same board (simulating opponent effect)
                    action_for_env = {
                        'type': 'wildblock',
                        'row': action['wildblock_row'],
                        'col': action['wildblock_column']
                    }
                else:
                    action_for_env = {
                        'type': action['action_name']
                    }
                
                # Apply action to environment
                new_self_board, _ = self.environment.apply_powerup(action_for_env)
                new_opponent_board = opponent_board.copy()  # In training, opponent board doesn't change
                
                # If wildblock was used, simulate its effect on opponent board
                if action['action_name'] == 'wildblock' and action['wildblock_row'] != -1:
                    # Apply 3x3 wildblock to opponent board
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            r, c = action['wildblock_row'] + dr, action['wildblock_column'] + dc
                            if 0 <= r < self.agent.board_height and 0 <= c < self.agent.board_width:
                                new_opponent_board[r, c] = 1
                
                new_powerups = self.environment.get_powerup_availability()
                
                # Calculate reward
                reward = self.enhanced_reward_function(old_self_board, new_self_board, 
                                                     old_opponent_board, new_opponent_board, action)
                done = not any(new_powerups.values())
                
                # Store experience
                self.agent.remember(current_board, opponent_board, current_powerups, action, 
                                  reward, new_self_board, new_opponent_board, new_powerups, done)
                
                episode_reward += reward
                self.action_usage[action['action_name']] += 1
                
                # Track usage statistics
                if action['action_name'] == 'bomb' and action['bomb_column'] >= 0:
                    self.bomb_column_usage[action['bomb_column']] += 1
                elif action['action_name'] == 'wildblock' and action['wildblock_column'] >= 1:
                    self.wildblock_column_usage[action['wildblock_column'] - 1] += 1  # Convert to 0-7 index
                
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
                    bomb_column_usage=self.bomb_column_usage,
                    wildblock_column_usage=self.wildblock_column_usage
                )
            
            self.episode_rewards.append(episode_reward)
            
            # Enhanced logging
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                total_actions = sum(self.action_usage.values())
                action_dist = {k: (v/total_actions)*100 for k, v in self.action_usage.items()}
                
                print(f"Episode {episode}: Avg Reward: {avg_reward:.2f}")
                print(f"  Actions: {action_dist}")
                
                # Show bomb column preferences
                total_bombs = sum(self.bomb_column_usage)
                if total_bombs > 0:
                    bomb_prefs = [f"Col{i}:{(count/total_bombs)*100:.1f}%" 
                                 for i, count in enumerate(self.bomb_column_usage) if count > 0]
                    print(f"  Bomb columns: {bomb_prefs[:5]}")
                
                # Show wildblock column preferences
                total_wildblocks = sum(self.wildblock_column_usage)
                if total_wildblocks > 0:
                    wildblock_prefs = [f"Col{i+1}:{(count/total_wildblocks)*100:.1f}%" 
                                     for i, count in enumerate(self.wildblock_column_usage) if count > 0]
                    print(f"  Wildblock columns: {wildblock_prefs[:5]}")
            
            # Save periodically
            if episode % 500 == 0 and episode > 0:
                model_path = os.path.join(self.save_dir, f"wildblock_model_ep{episode}.pth")
                self.agent.save_model(model_path)
        
        # Final save
        final_path = os.path.join(self.save_dir, "wildblock_model_final.pth")
        self.agent.save_model(final_path)

        # Final Visualization dashboard
        self.visualizer.create_enhanced_dashboard("final_wildblock_training_dashboard.png")
        # self.visualizer.plot_bomb_column_analysis("final_wildblock_bomb_analysis.png")
        
        return final_path
    
    def export_for_unity(self, model_path: str):
        """Export trained model for Unity"""
        self.agent.load_model(model_path)
        
        # ONNX export
        onnx_path = model_path.replace('.pth', '.onnx')
        
        # Dummy input: 8 channels for dual-board input
        dummy_input = torch.randn(1, 8, self.agent.board_height, self.agent.board_width).to(self.agent.device)
        
        torch.onnx.export(
            self.agent.q_network,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=['dual_board_state'],
            output_names=['action_bomb_wildblock_q_values']
        )
        
        print(f"Unity wildblock model exported: {onnx_path}")
        print("Unity integration info:")
        print("- Input: (1, 8, 20, 10) tensor")
        print("  - Channels: [self_board, opponent_board, bottom_clear, gravity, bomb, wildblock, height_diff, threat_level]")
        print("- Output: (1, 23) tensor")
        print("  - First 5 values: [none, bottom_clear, gravity, bomb, wildblock] Q-values")
        print("  - Next 10 values: bomb column Q-values [col0, col1, ..., col9]")
        print("  - Last 8 values: wildblock column Q-values [col1, col2, ..., col8]")
        
        return onnx_path


# Usage example
if __name__ == "__main__":
    # Train model
    trainer = OptimizedWildblockTrainer("tetris_boards3.pkl")
    model_path = trainer.train(episodes=3000)
    
    # Export for Unity
    onnx_path = trainer.export_for_unity(model_path)
    
    print(f"\nWildblock training complete!")
    print(f"PyTorch model: {model_path}")
    print(f"Unity ONNX model: {onnx_path}")
    
    # Demo prediction
    agent = OptimizedWildblockAgent()
    agent.load_model(model_path)
    
    # Create test boards
    self_test_board = np.zeros((20, 10))
    self_test_board[15:, [2, 5, 7]] = 1  # Add blocks in columns 2, 5, 7
    
    opponent_test_board = np.zeros((20, 10))
    opponent_test_board[12:, [1, 3, 6, 8]] = 1  # Opponent has blocks in columns 1, 3, 6, 8
    
    test_powerups = {'bottom_clear': True, 'gravity': False, 'bomb': True, 'wildblock': True}
    
    result = agent.predict_unity(self_test_board, opponent_test_board, test_powerups)
    print(f"\nDemo prediction: {result}")