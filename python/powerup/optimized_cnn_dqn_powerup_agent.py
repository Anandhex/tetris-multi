# optimized_cnn_dqn_agent.py - Surface-only bomb targeting
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np
import random
import os
from typing import Dict, List, Tuple, Optional
from powerup_training_visualizer2 import TrainingVisualizer, TrainingLogger

class OptimizedCNNDQN(nn.Module):
    """
    Optimized CNN with surface-only bomb targeting
    Output: 4 action types + 10 bomb columns = 14 total outputs
    """
    
    def __init__(self, board_height=20, board_width=10):
        super(OptimizedCNNDQN, self).__init__()
        
        self.board_height = board_height
        self.board_width = board_width
        
        # Shared convolutional backbone
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Global features for action type
        self.global_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Action type branch (4 outputs: none, bottom_clear, gravity, bomb)
        self.action_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        
        # Bomb column branch (10 outputs: one per column)
        # Uses column-wise pooling to focus on vertical patterns
        self.bomb_column_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 10)),  # Pool to (1, 10) - one value per column
            nn.Flatten(),
            nn.Linear(128 * 10, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 10)  # Q-value for bombing each column
        )
        
    def forward(self, x):
        # Shared features from conv layers
        features = self.conv_layers(x)  # (batch, 128, 20, 10)
        
        # Action type Q-values
        pooled_features = self.global_pool(features)
        action_q = self.action_branch(pooled_features)  # (batch, 4)
        
        # Bomb column Q-values
        bomb_col_q = self.bomb_column_branch(features)  # (batch, 10)
        
        # Concatenate outputs: [action_q, bomb_col_q]
        # Output shape: (batch, 14) = 4 actions + 10 columns
        output = torch.cat([action_q, bomb_col_q], dim=1)
        
        return output


class OptimizedBombAgent:
    """
    Optimized agent with surface-only bomb targeting
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
    
    def prepare_state(self, board: np.ndarray, powerups: Dict[str, bool]) -> torch.Tensor:
        """Convert board + powerups to 4-channel tensor"""
        board_channel = board.astype(np.float32)
        bottom_clear_channel = np.full_like(board, 1.0 if powerups.get('bottom_clear', False) else 0.0, dtype=np.float32)
        gravity_channel = np.full_like(board, 1.0 if powerups.get('gravity', False) else 0.0, dtype=np.float32)
        bomb_channel = np.full_like(board, 1.0 if powerups.get('bomb', False) else 0.0, dtype=np.float32)
        
        state = np.stack([board_channel, bottom_clear_channel, gravity_channel, bomb_channel])
        return torch.FloatTensor(state).to(self.device)
    
    def find_surface_blocks(self, board: np.ndarray) -> Dict[int, Optional[int]]:
        """
        Find surface block (topmost block) in each column
        
        Returns:
            Dict[column -> row] where row is the topmost block, None if column empty
        """
        surface_blocks = {}
        
        for col in range(self.board_width):
            surface_row = None
            for row in range(self.board_height):
                if board[row, col] == 1:  # Found first block from top
                    surface_row = row
                    break
            surface_blocks[col] = surface_row
        
        return surface_blocks
    
    def predict_unity(self, board: np.ndarray, powerups: Dict[str, bool]) -> Dict:
        """
        MAIN METHOD FOR UNITY: Optimized single prediction
        
        Returns:
            {
                'action_type': 0-3,
                'action_name': string,
                'bomb_column': 0-9 (if bomb selected),
                'bomb_row': actual row of surface block (if bomb selected),
                'confidence': float,
                'valid_columns': list of columns with surface blocks
            }
        """
        
        # Prepare input
        state = self.prepare_state(board, powerups).unsqueeze(0)
        
        # Single forward pass
        with torch.no_grad():
            output = self.q_network(state).cpu().numpy()[0]  # Shape: (14,)
        
        # Split output
        action_q = output[:4]      # Action types
        bomb_col_q = output[4:]    # Bomb columns
        
        # Find surface blocks
        surface_blocks = self.find_surface_blocks(board)
        valid_columns = [col for col, row in surface_blocks.items() if row is not None]
        
        # Mask invalid actions
        masked_action_q = self._mask_actions(action_q, powerups, valid_columns)
        
        # Select best action
        best_action_id = np.argmax(masked_action_q)
        action_names = ['none', 'bottom_clear', 'gravity', 'bomb']
        action_name = action_names[best_action_id]
        
        # Calculate confidence
        action_probs = self._softmax(masked_action_q)
        confidence = action_probs[best_action_id]
        
        result = {
            'action_type': int(best_action_id),
            'action_name': action_name,
            'confidence': float(confidence),
            'valid_columns': valid_columns
        }
        
        # If bomb selected, find best column
        if best_action_id == 3:  # bomb action
            masked_bomb_col_q = self._mask_bomb_columns(bomb_col_q, valid_columns)
            best_col = np.argmax(masked_bomb_col_q)
            
            # Get actual surface block position
            bomb_row = surface_blocks[best_col] if best_col in surface_blocks else 0
            
            result.update({
                'bomb_column': int(best_col),
                'bomb_row': int(bomb_row) if bomb_row is not None else -1,
                'bomb_confidence': float(self._softmax(masked_bomb_col_q)[best_col])
            })
        else:
            result.update({
                'bomb_column': -1,
                'bomb_row': -1,
                'bomb_confidence': 0.0
            })
        
        return result
    
    def _mask_actions(self, action_q: np.ndarray, powerups: Dict[str, bool], valid_columns: List[int]) -> np.ndarray:
        """Mask invalid actions"""
        masked = action_q.copy()
        
        # none (0) always valid
        if not powerups.get('bottom_clear', False):
            masked[1] = -np.inf
        if not powerups.get('gravity', False):
            masked[2] = -np.inf
        if not powerups.get('bomb', False) or len(valid_columns) == 0:
            masked[3] = -np.inf
        
        return masked
    
    def _mask_bomb_columns(self, bomb_col_q: np.ndarray, valid_columns: List[int]) -> np.ndarray:
        """Mask invalid bomb columns (those without surface blocks)"""
        masked = bomb_col_q.copy()
        
        for col in range(len(bomb_col_q)):
            if col not in valid_columns:
                masked[col] = -np.inf
        
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
    
    def choose_action_training(self, board: np.ndarray, powerups: Dict[str, bool]) -> Dict:
        """Training version with epsilon-greedy exploration"""
        if np.random.random() <= self.epsilon:
            return self._random_action(board, powerups)
        else:
            return self.predict_unity(board, powerups)
    
    def _random_action(self, board: np.ndarray, powerups: Dict[str, bool]) -> Dict:
        """Random valid action for training"""
        surface_blocks = self.find_surface_blocks(board)
        valid_columns = [col for col, row in surface_blocks.items() if row is not None]
        
        valid_actions = [0]  # none always valid
        if powerups.get('bottom_clear', False):
            valid_actions.append(1)
        if powerups.get('gravity', False):
            valid_actions.append(2)
        if powerups.get('bomb', False) and len(valid_columns) > 0:
            valid_actions.append(3)
        
        action_type = random.choice(valid_actions)
        action_names = ['none', 'bottom_clear', 'gravity', 'bomb']
        
        result = {
            'action_type': action_type,
            'action_name': action_names[action_type],
            'confidence': 1.0,
            'valid_columns': valid_columns
        }
        
        if action_type == 3:  # bomb
            bomb_col = random.choice(valid_columns)
            bomb_row = surface_blocks[bomb_col]
            
            result.update({
                'bomb_column': bomb_col,
                'bomb_row': bomb_row if bomb_row is not None else -1,
                'bomb_confidence': 1.0
            })
        else:
            result.update({
                'bomb_column': -1,
                'bomb_row': -1,
                'bomb_confidence': 0.0
            })
        
        return result
    
    def remember(self, board: np.ndarray, powerups: Dict[str, bool], action: Dict, 
                 reward: float, next_board: np.ndarray, next_powerups: Dict[str, bool], done: bool):
        """Store experience for training"""
        state = self.prepare_state(board, powerups).cpu().numpy()
        next_state = self.prepare_state(next_board, next_powerups).cpu().numpy()
        
        action_encoded = {
            'action_type': action['action_type'],
            'bomb_column': action.get('bomb_column', -1)
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
        current_q = self.q_network(states)  # (batch, 14)
        
        # Action type Q-values
        action_types = torch.LongTensor([a['action_type'] for a in actions]).to(self.device)
        current_action_q = current_q[:, :4].gather(1, action_types.unsqueeze(1)).squeeze(1)
        
        # Target Q-values for actions
        with torch.no_grad():
            next_q = self.target_network(next_states)
            next_action_q = next_q[:, :4].max(1)[0]
            target_action_q = rewards + (self.gamma * next_action_q * ~dones)
        
        # Action loss
        action_loss = F.mse_loss(current_action_q, target_action_q)
        
        # Bomb column loss (only for bomb actions)
        bomb_loss = 0.0
        bomb_indices = [i for i, a in enumerate(actions) if a['action_type'] == 3 and a['bomb_column'] >= 0]
        
        if bomb_indices:
            bomb_columns = torch.LongTensor([actions[i]['bomb_column'] for i in bomb_indices]).to(self.device)
            current_bomb_q = current_q[bomb_indices, 4:].gather(1, bomb_columns.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                next_bomb_q = next_q[bomb_indices, 4:].max(1)[0]
                target_bomb_q = rewards[bomb_indices] + (self.gamma * next_bomb_q * ~dones[bomb_indices])
            
            bomb_loss = F.mse_loss(current_bomb_q, target_bomb_q)
        
        # Total loss
        total_loss = action_loss + bomb_loss
        
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
            'model_type': 'optimized_surface_bomb'
        }
        
        torch.save(checkpoint, filepath)
        print(f"Optimized model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.epsilon = 0.0
        print(f"Optimized model loaded from {filepath}")
    
    def set_eval_mode(self):
        """Set to evaluation mode"""
        self.q_network.eval()
        self.epsilon = 0.0


# Optimized trainer
class OptimizedBombTrainer:
    """Trainer for optimized surface-bomb model"""
    
    def __init__(self, dataset_path: str, save_dir: str = "optimized_models"):
        from python.powerup.environments import TrainingEnvironment
        
        self.environment = TrainingEnvironment(dataset_path)
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
        self.action_usage = {'none': 0, 'bottom_clear': 0, 'gravity': 0, 'bomb': 0}
        self.bomb_column_usage = [0] * 10  # Track which columns are bombed

        ## visualization code 
        self.visualizer = TrainingVisualizer()
        self.logger = TrainingLogger(self.visualizer)
    
    def enhanced_reward_function(self, old_board: np.ndarray, new_board: np.ndarray, action: Dict) -> float:
        """Enhanced reward function considering bomb effectiveness"""
        
        blocks_removed = np.sum(old_board) - np.sum(new_board)
        
        if action['action_name'] == 'bomb':
            base_reward = 5.0 + blocks_removed * 0.5
            
            # Bonus for strategic bomb placement
            if action['bomb_row'] != -1:
                bomb_impact = self.agent.calculate_bomb_impact(old_board, action['bomb_row'], action['bomb_column'])
                
                # Reward efficiency: more blocks destroyed = better
                efficiency_bonus = bomb_impact * 0.3
                
                # Reward for targeting columns with many blocks above
                column_blocks = np.sum(old_board[:, action['bomb_column']])
                column_bonus = column_blocks * 0.1
                
                reward = base_reward + efficiency_bonus + column_bonus
            else:
                reward = base_reward
                
        elif action['action_name'] == 'bottom_clear':
            bottom_blocks = np.sum(old_board[-1, :])
            reward = 4.0 + bottom_blocks * 0.4
            
        elif action['action_name'] == 'gravity':
            reward = 3.0 + blocks_removed * 0.3
            
        else:  # 'none'
            reward = -0.5
        
        return np.clip(reward, -5, 20)
    
    def train(self, episodes: int = 5000):
        """Train optimized model"""
        print(f"Training optimized surface-bomb model for {episodes} episodes...")
        
        for episode in range(episodes):
            self.environment.reset()
            episode_reward = 0
            
            for step in range(8):
                current_board = self.environment.get_board_state()
                current_powerups = self.environment.get_powerup_availability()
                
                # Choose action
                action = self.agent.choose_action_training(current_board, current_powerups)
                
                # Apply action
                old_board = current_board.copy()
                
                # Format action for environment compatibility
                if action['action_name'] == 'bomb' and action['bomb_row'] != -1:
                    action_for_env = {
                        'type': action['action_name'],
                        'row': action['bomb_row'],
                        'col': action['bomb_column']
                    }
                else:
                    action_for_env = {
                        'type': action['action_name']
                    }
                
                new_board, _ = self.environment.apply_powerup(action_for_env)
                new_powerups = self.environment.get_powerup_availability()
                
                # Calculate reward
                reward = self.enhanced_reward_function(old_board, new_board, action)
                done = not any(new_powerups.values())
                
                # Store experience
                self.agent.remember(current_board, current_powerups, action, 
                                  reward, new_board, new_powerups, done)
                
                episode_reward += reward
                self.action_usage[action['action_name']] += 1
                
                # Track bomb column usage
                if action['action_name'] == 'bomb' and action['bomb_column'] >= 0:
                    self.bomb_column_usage[action['bomb_column']] += 1
                
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
                
                # Show bomb column preferences
                total_bombs = sum(self.bomb_column_usage)
                if total_bombs > 0:
                    bomb_prefs = [f"Col{i}:{(count/total_bombs)*100:.1f}%" 
                                 for i, count in enumerate(self.bomb_column_usage) if count > 0]
                    print(f"  Bomb columns: {bomb_prefs[:5]}")  # Show top 5
            
            # Save periodically
            if episode % 500 == 0 and episode > 0:
                model_path = os.path.join(self.save_dir, f"optimized_model_ep{episode}.pth")
                self.agent.save_model(model_path)
        
        # Final save
        final_path = os.path.join(self.save_dir, "optimized_model_final.pth")
        self.agent.save_model(final_path)

        # Final Visualization dashboard
        self.visualizer.create_training_dashboard("final_training_dashboard.png")
        self.visualizer.plot_bomb_column_analysis("final_bomb_analysis.png")
        
        return final_path
    
    def export_for_unity(self, model_path: str):
        """Export trained model for Unity"""
        self.agent.load_model(model_path)
        
        # ONNX export
        onnx_path = model_path.replace('.pth', '.onnx')
        
        dummy_input = torch.randn(1, 4, self.agent.board_height, self.agent.board_width).to(self.agent.device)
        
        torch.onnx.export(
            self.agent.q_network,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=['board_state'],
            output_names=['action_and_column_q_values']
        )
        
        print(f"Unity model exported: {onnx_path}")
        print("Unity integration info:")
        print("- Input: (1, 4, 20, 10) tensor")
        print("- Output: (1, 14) tensor")
        print("- First 4 values: [none, bottom_clear, gravity, bomb] Q-values")
        print("- Next 10 values: bomb column Q-values [col0, col1, ..., col9]")
        
        return onnx_path


# Usage example
if __name__ == "__main__":
    # Train model
    trainer = OptimizedBombTrainer("tetris_boards.pkl")
    model_path = trainer.train(episodes=3000)
    
    # Export for Unity
    onnx_path = trainer.export_for_unity(model_path)
    
    print(f"\nTraining complete!")
    print(f"PyTorch model: {model_path}")
    print(f"Unity ONNX model: {onnx_path}")
    
    # Demo prediction
    agent = OptimizedBombAgent()
    agent.load_model(model_path)
    
    # Create test board with some surface blocks
    test_board = np.zeros((20, 10))
    test_board[15:, [2, 5, 7]] = 1  # Add blocks in columns 2, 5, 7
    test_powerups = {'bottom_clear': True, 'gravity': False, 'bomb': True}
    
    result = agent.predict_unity(test_board, test_powerups)
    print(f"\nDemo prediction: {result}")