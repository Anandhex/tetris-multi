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
    """Neural Network for PowerUp DQN"""
    
    def __init__(self, state_size=5, action_size=2, hidden_sizes=[128, 64, 32]):
        super(PowerUpDQNNetwork, self).__init__()
        
        # Input: [lines, holes, bumpiness, height, blocks_since_powerup]
        # Output: [use_now, wait]
        
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_size = hidden_size
            
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class PowerUpDQNAgent:
    """DQN Agent for PowerUp decision making"""
    
    def __init__(self, 
                 state_size=5, 
                 action_size=2,
                 hidden_sizes=[128, 64, 32],
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
        print(f"PowerUp DQN Agent initialized on {self.device}")
        
        # Network parameters
        self.state_size = state_size
        self.action_size = action_size
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
        self.q_network = PowerUpDQNNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.target_network = PowerUpDQNNetwork(state_size, action_size, hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize target network
        self.update_target_network()
        
        # Training tracking
        self.steps = 0
        self.episodes = 0
        
        # TensorBoard
        if tensorboard_log_dir is None:
            tensorboard_log_dir = f"runs/powerup_dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(tensorboard_log_dir)
        
        # PowerUp types
        self.powerup_types = ['bottom_line_clear', 'gravity', 'bomb']
        
        print(f"TensorBoard logs: {tensorboard_log_dir}")
        
    def get_state(self, board_features, blocks_since_powerup, powerup_type):
        """
        Create state representation for powerup decision
        Args:
            board_features: [lines, holes, bumpiness, height]
            blocks_since_powerup: number of blocks placed since powerup received
            powerup_type: type of powerup ('bottom_line_clear', 'gravity', 'bomb')
        """
        # Normalize board features
        lines, holes, bumpiness, height = board_features
        
        # Normalize features (you might want to adjust these based on your game)
        normalized_lines = lines / 4.0  # Tetris typically clears 1-4 lines
        normalized_holes = min(holes / 20.0, 1.0)  # Cap at 20 holes
        normalized_bumpiness = min(bumpiness / 50.0, 1.0)  # Cap at 50 bumpiness
        normalized_height = height / 20.0  # Assuming 20 row board
        normalized_blocks = min(blocks_since_powerup / 10.0, 1.0)  # Cap at 10 blocks
        
        # One-hot encode powerup type
        powerup_encoding = [0, 0, 0]
        if powerup_type in self.powerup_types:
            powerup_encoding[self.powerup_types.index(powerup_type)] = 1
        
        state = [normalized_lines, normalized_holes, normalized_bumpiness, 
                normalized_height, normalized_blocks] + powerup_encoding
        
        return np.array(state, dtype=np.float32)
    
    def act(self, state):
        """
        Choose action using epsilon-greedy policy
        Returns: 0 for 'use_now', 1 for 'wait'
        """
        if random.random() <= self.epsilon:
            return random.randint(0, 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
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
    
    def calculate_powerup_reward(self, board_before, board_after, powerup_type):
        """
        Calculate reward based on board improvement after powerup usage
        Args:
            board_before: [lines, holes, bumpiness, height] before powerup
            board_after: [lines, holes, bumpiness, height] after powerup
            powerup_type: type of powerup used
        """
        lines_before, holes_before, bumpiness_before, height_before = board_before
        lines_after, holes_after, bumpiness_after, height_after = board_after
        
        # Base reward for lines cleared
        lines_cleared = lines_after - lines_before
        line_reward = lines_cleared * 100  # 100 points per line
        
        # Reward for reducing holes
        holes_reduced = holes_before - holes_after
        hole_reward = holes_reduced * 50  # 50 points per hole filled
        
        # Reward for reducing bumpiness
        bumpiness_reduced = bumpiness_before - bumpiness_after
        bumpiness_reward = bumpiness_reduced * 10  # 10 points per bumpiness reduction
        
        # Reward for reducing height
        height_reduced = height_before - height_after
        height_reward = height_reduced * 20  # 20 points per height reduction
        
        # Penalty for wasting powerup (if no improvement)
        waste_penalty = 0
        if holes_reduced <= 0 and bumpiness_reduced <= 0 and height_reduced <= 0 and lines_cleared <= 0:
            waste_penalty = -100
        
        # PowerUp specific bonuses
        powerup_bonus = 0
        if powerup_type == 'bottom_line_clear' and lines_cleared > 0:
            powerup_bonus = 50
        elif powerup_type == 'gravity' and holes_reduced > 0:
            powerup_bonus = 75
        elif powerup_type == 'bomb' and (holes_reduced > 0 or bumpiness_reduced > 0):
            powerup_bonus = 60
        
        total_reward = line_reward + hole_reward + bumpiness_reward + height_reward + powerup_bonus + waste_penalty
        
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
        print(f"PowerUp DQN model saved to {filepath}")
    
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
            print(f"PowerUp DQN model loaded from {filepath}")
            return True
        return False
    
    def log_episode_metrics(self, episode, powerup_rewards, powerup_usage_stats):
        """Log episode metrics to TensorBoard"""
        self.episodes = episode
        
        # Log powerup performance
        if powerup_rewards:
            avg_reward = np.mean(powerup_rewards)
            self.writer.add_scalar('PowerUp/Average_Reward', avg_reward, episode)
            self.writer.add_scalar('PowerUp/Total_Reward', sum(powerup_rewards), episode)
        
        # Log powerup usage statistics
        for powerup_type, stats in powerup_usage_stats.items():
            used_count = stats.get('used', 0)
            total_count = stats.get('total', 0)
            usage_rate = used_count / total_count if total_count > 0 else 0
            
            self.writer.add_scalar(f'PowerUp_Usage/{powerup_type}_rate', usage_rate, episode)
            self.writer.add_scalar(f'PowerUp_Usage/{powerup_type}_used', used_count, episode)
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()