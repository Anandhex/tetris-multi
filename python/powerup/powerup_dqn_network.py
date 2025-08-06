import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
import numpy as np
import os
from datetime import datetime

class PowerUpDQNNetwork(nn.Module):
    """Neural Network for PowerUp placement evaluation"""
    
    def __init__(self, state_size=7, hidden_sizes=[128, 64, 32]):
        super(PowerUpDQNNetwork, self).__init__()
        
        # Input: [lines, holes, bumpiness, height, placement_impact, blocks_since_powerup, powerup_type]
        # Output: Single Q-value for this specific placement
        
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_size = hidden_size
            
        layers.append(nn.Linear(input_size, 1))  # Single Q-value output
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class FinalPowerUpDQNAgent:
    """Final PowerUp DQN Agent with complete decision tracking"""
    
    def __init__(self, 
                 state_size=7,
                 learning_rate=0.001,
                 gamma=0.95,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 memory_size=10000,
                 batch_size=32,
                 target_update_freq=1000,
                 tensorboard_log_dir=None):
        
        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Final PowerUp DQN Agent initialized on {self.device}")
        
        # Network parameters
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Exploration parameters
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Memory and training parameters
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Networks
        self.q_network = PowerUpDQNNetwork(state_size).to(self.device)
        self.target_network = PowerUpDQNNetwork(state_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize target network
        self.update_target_network()
        
        # Training tracking
        self.steps = 0
        self.episodes = 0
        
        # TensorBoard
        if tensorboard_log_dir is None:
            tensorboard_log_dir = f"runs/final_powerup_dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(tensorboard_log_dir)
        
        # PowerUp types
        self.powerup_types = ['bottom_line_clear', 'gravity', 'bomb']
        
        print(f"TensorBoard logs: {tensorboard_log_dir}")
    
    def get_placement_state(self, board_features, placement_impact, blocks_since_powerup, powerup_type):
        """Create state representation for a specific placement"""
        lines, holes, bumpiness, height = board_features
        
        # Normalize features
        normalized_lines = lines / 4.0
        normalized_holes = min(holes / 20.0, 1.0)
        normalized_bumpiness = min(bumpiness / 50.0, 1.0)
        normalized_height = height / 20.0
        normalized_impact = min(abs(placement_impact) / 100.0, 1.0)
        normalized_blocks = min(blocks_since_powerup / 10.0, 1.0)
        
        # PowerUp type encoding
        powerup_encoding = 0
        if powerup_type == 'bottom_line_clear':
            powerup_encoding = 0.33
        elif powerup_type == 'gravity':
            powerup_encoding = 0.66
        elif powerup_type == 'bomb':
            powerup_encoding = 1.0
        
        state = [normalized_lines, normalized_holes, normalized_bumpiness, 
                normalized_height, normalized_impact, normalized_blocks, powerup_encoding]
        
        return np.array(state, dtype=np.float32)
    
    def find_bomb_landing_position(self, board_2d, col):
        """Find where bomb lands when dropped in this column"""
        if board_2d is None or board_2d.size == 0:
            return 19  # Bottom of empty board
        
        height, width = board_2d.shape
        
        # Drop from top until hitting something
        for row in range(height):
            if board_2d[row, col] != 0:  # Hit a block
                return max(0, row - 1)  # Land on top of the block
        
        return height - 1  # Hit bottom of board
    
    def calculate_bomb_impact_at_position(self, board_2d, landing_row, col):
        """Calculate impact of bomb exploding at specific position"""
        if board_2d is None or board_2d.size == 0:
            return 0
        
        height, width = board_2d.shape
        impact_score = 0
        blocks_destroyed = 0
        holes_filled = 0
        
        # Analyze 3x3 bomb area around landing position
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r, c = landing_row + dr, col + dc
                
                if 0 <= r < height and 0 <= c < width:
                    if board_2d[r, c] != 0:  # Block will be destroyed
                        blocks_destroyed += 1
                        impact_score += 10
                    
                    # Check if destroying this block helps fill holes below
                    if r < height - 1:
                        for check_r in range(r + 1, height):
                            if board_2d[check_r, c] == 0:  # Found hole below
                                holes_filled += 1
                                impact_score += 15
                                break
                            elif board_2d[check_r, c] != 0:  # Hit filled cell
                                break
        
        # Bonus for targeting areas with high local density
        density_bonus = 0
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r, c = landing_row + dr, col + dc
                if 0 <= r < height and 0 <= c < width and board_2d[r, c] != 0:
                    density_bonus += 1
        
        impact_score += density_bonus * 2
        
        # Penalty for bombing areas with few blocks
        if blocks_destroyed < 2:
            impact_score -= 20
        
        return impact_score
    
    def generate_all_powerup_options(self, powerup_type, board_2d, board_features, blocks_since_powerup):
        """
        Generate all possible powerup options with complete decision tracking
        Returns: list of (decision_data, state_vector) tuples
        """
        options = []
        
        if powerup_type == 'bomb':
            # Generate all possible bomb drop columns
            for col in range(10):
                # Find landing position
                landing_row = self.find_bomb_landing_position(board_2d, col)
                
                # Calculate impact for this column
                impact = self.calculate_bomb_impact_at_position(board_2d, landing_row, col)
                
                # Create state features for this placement
                state_features = self.get_placement_state(
                    board_features, impact, blocks_since_powerup, 'bomb'
                )
                
                # Complete decision data with position tracking
                decision_data = {
                    'action': 'use_bomb',
                    'powerup_type': 'bomb',
                    'column': col,
                    'landing_row': landing_row,
                    'impact': impact,
                    'option_key': f"bomb_col_{col}"
                }
                
                options.append((decision_data, state_features))
            
            # Add wait option
            wait_state = self.get_placement_state(
                board_features, -5, blocks_since_powerup, 'bomb'
            )
            
            wait_decision = {
                'action': 'wait',
                'powerup_type': 'bomb',
                'impact': -5,
                'option_key': 'wait'
            }
            
            options.append((wait_decision, wait_state))
            
        elif powerup_type == 'gravity':
            # Calculate gravity impact
            total_holes = board_features[1] if len(board_features) > 1 else 0
            impact = total_holes * 15
            
            # Use option
            use_state = self.get_placement_state(
                board_features, impact, blocks_since_powerup, 'gravity'
            )
            
            use_decision = {
                'action': 'use_gravity',
                'powerup_type': 'gravity',
                'impact': impact,
                'option_key': 'use_gravity'
            }
            
            # Wait option
            wait_state = self.get_placement_state(
                board_features, -5, blocks_since_powerup, 'gravity'
            )
            
            wait_decision = {
                'action': 'wait',
                'powerup_type': 'gravity',
                'impact': -5,
                'option_key': 'wait'
            }
            
            options.append((use_decision, use_state))
            options.append((wait_decision, wait_state))
            
        elif powerup_type == 'bottom_line_clear':
            # Calculate bottom line clear impact
            if board_2d is not None and board_2d.size > 0:
                bottom_row_blocks = sum(1 for cell in board_2d[-1] if cell != 0)
                impact = bottom_row_blocks * 20
            else:
                impact = 0
            
            # Use option
            use_state = self.get_placement_state(
                board_features, impact, blocks_since_powerup, 'bottom_line_clear'
            )
            
            use_decision = {
                'action': 'use_bottom_clear',
                'powerup_type': 'bottom_line_clear',
                'impact': impact,
                'option_key': 'use_bottom_clear'
            }
            
            # Wait option
            wait_state = self.get_placement_state(
                board_features, -5, blocks_since_powerup, 'bottom_line_clear'
            )
            
            wait_decision = {
                'action': 'wait',
                'powerup_type': 'bottom_line_clear',
                'impact': -5,
                'option_key': 'wait'
            }
            
            options.append((use_decision, use_state))
            options.append((wait_decision, wait_state))
        
        return options
    
    def make_powerup_decision(self, powerup_type, board_2d, board_features, blocks_since_powerup, episode_step):
        """
        Make complete powerup decision with position tracking
        Returns: complete decision with Q-value and position information
        """
        # Generate all possible options
        all_options = self.generate_all_powerup_options(
            powerup_type, board_2d, board_features, blocks_since_powerup
        )
        
        if not all_options:
            return None
        
        # Log epsilon for this step
        self.writer.add_scalar("powerup/epsilon", self.epsilon, episode_step)
        
        # Evaluate all options
        option_evaluations = []
        
        for decision_data, state_features in all_options:
            # Get Q-value for this option
            state_tensor = torch.FloatTensor(state_features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_value = self.q_network(state_tensor).item()
            
            # Create complete evaluation
            evaluation = {
                'decision_data': decision_data,
                'state_features': state_features,
                'q_value': q_value
            }
            
            option_evaluations.append(evaluation)
        
        # Choose best option (epsilon-greedy)
        if random.random() <= self.epsilon:
            # Random exploration
            chosen_evaluation = random.choice(option_evaluations)
            decision_type = 'exploration'
        else:
            # Choose best Q-value
            chosen_evaluation = max(option_evaluations, key=lambda x: x['q_value'])
            decision_type = 'exploitation'
        
        # Create complete decision result
        decision_result = {
            'decision_data': chosen_evaluation['decision_data'],
            'state_features': chosen_evaluation['state_features'],
            'q_value': chosen_evaluation['q_value'],
            'decision_type': decision_type,
            'all_evaluations': option_evaluations,  # For analysis
            'powerup_type': powerup_type
        }
        
        return decision_result
    
    def remember(self, state, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, reward, next_state, done))
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[1] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[3] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).squeeze()
        next_q_values = self.target_network(next_states).squeeze().detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Log training metrics
        self.writer.add_scalar('Training/Loss', loss.item(), self.steps)
        self.writer.add_scalar('Training/Epsilon', self.epsilon, self.steps)
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.steps += 1
        
        # Update target network
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def calculate_placement_reward(self, board_before, board_after, powerup_type, decision_data):
        """Calculate reward based on board improvement after powerup usage"""
        lines_before, holes_before, bumpiness_before, height_before = board_before
        lines_after, holes_after, bumpiness_after, height_after = board_after
        
        # Calculate improvements
        lines_cleared = lines_after - lines_before
        holes_reduced = holes_before - holes_after
        bumpiness_reduced = bumpiness_before - bumpiness_after
        height_reduced = height_before - height_after
        
        # Base rewards
        line_reward = lines_cleared * 100
        hole_reward = holes_reduced * 50
        bumpiness_reward = bumpiness_reduced * 10
        height_reward = height_reduced * 20
        
        # Efficiency bonus based on total improvement
        total_improvement = lines_cleared + holes_reduced + bumpiness_reduced + height_reduced
        efficiency_bonus = 0
        
        if total_improvement > 0:
            if powerup_type == 'bomb':
                efficiency_bonus = min(total_improvement * 15, 150)
            elif powerup_type == 'gravity':
                efficiency_bonus = min(total_improvement * 10, 100)
            elif powerup_type == 'bottom_line_clear':
                efficiency_bonus = min(total_improvement * 8, 75)
        
        # Penalties
        waste_penalty = 0
        if total_improvement <= 0 and decision_data.get('action') != 'wait':
            waste_penalty = -100
        
        waiting_penalty = 0
        if decision_data.get('action') == 'wait':
            waiting_penalty = -1
        
        total_reward = line_reward + hole_reward + bumpiness_reward + height_reward + efficiency_bonus + waste_penalty + waiting_penalty
        
        return total_reward
    
    def save_model(self, filepath):
        """Save the model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes
        }, filepath)
        print(f"Final PowerUp DQN model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load the model"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.steps = checkpoint.get('steps', 0)
            self.episodes = checkpoint.get('episodes', 0)
            print(f"Final PowerUp DQN model loaded from {filepath}")
            return True
        return False
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()