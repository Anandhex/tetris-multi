# dqn_agent.py
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

class TetrisDQN(nn.Module):
    def __init__(self, input_size, hidden_size=512, output_size=40):
        super(TetrisDQN, self).__init__()
        
        # Convolutional layers for board state
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((5, 5))
        
        # Fixed conv output size after adaptive pooling
        conv_output_size = 64 * 5 * 5  # Always 1600 after adaptive pooling
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size + 8, hidden_size)  # +8 for piece info and metrics
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)
        
        self.dropout = nn.Dropout(0.3)
        # Use LayerNorm instead of BatchNorm to avoid batch size issues
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, board, piece_info, metrics):
        # Process board through conv layers
        x = F.relu(self.conv1(board))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # Always outputs batch_size x 64 x 5 x 5
        
        # Flatten conv output
        x = x.view(x.size(0), -1)  # Should always be batch_size x 1600
        
        # Concatenate with piece info and metrics
        x = torch.cat([x, piece_info, metrics], dim=1)  # batch_size x 1608
        
        # Fully connected layers with layer norm
        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

class DQNAgent:
    def __init__(self, state_size, action_size=40, lr=0.001, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 tensorboard_log_dir=None):
        self.device = device
        self.action_size = action_size
        self.memory = deque(maxlen=50000)  # Increased memory size
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995  # Slower decay
        self.learning_rate = lr
        self.batch_size = 64  # Increased batch size
        self.target_update_freq = 1000
        self.steps = 0
        self.training_steps = 0
        
        # TensorBoard writer
        if tensorboard_log_dir is None:
            tensorboard_log_dir = f"runs/tetris_dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(tensorboard_log_dir)
        self.tensorboard_log_dir = tensorboard_log_dir
        
        # Neural networks
        self.q_network = TetrisDQN(state_size, output_size=action_size).to(device)
        self.target_network = TetrisDQN(state_size, output_size=action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.9)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.q_values_history = []
        
        print(f"DQN Agent initialized on {device}")
        print(f"TensorBoard logs: {tensorboard_log_dir}")
        
        # Log model architecture
        self.writer.add_text('Model/Architecture', str(self.q_network))
        self.writer.add_text('Model/Parameters', f"Total parameters: {sum(p.numel() for p in self.q_network.parameters())}")
    
    def preprocess_state(self, game_state):
        """Convert game state to neural network input"""
        board = np.array(game_state.get('board', []))
        
        # Handle variable board sizes from curriculum learning
        if len(board) == 0:
            # Default empty board (20x10)
            board = np.zeros(200)
            board_height = 20
            board_width = 10
        else:
            # Calculate actual board dimensions
            board_width = 10  # Always 10
            board_height = len(board) // board_width
            
            if board_height == 0:
                board_height = 20
                board = np.zeros(200)
        
        # Ensure board is properly sized
        expected_size = board_height * board_width
        if len(board) != expected_size:
            # Pad or truncate to expected size
            if len(board) < expected_size:
                board = np.pad(board, (0, expected_size - len(board)), mode='constant')
            else:
                board = board[:expected_size]
        
        # Reshape board to proper dimensions
        board = board.reshape(1, 1, board_height, board_width)
        
        # Convert to float32 for better performance
        board = board.astype(np.float32)
        
        # Piece information (ensure consistent size)
        piece_info = game_state.get('currentPiece', [0, 0, 0, 0])
        next_piece = game_state.get('nextPiece', [0])
        
        # Pad to exactly 4 features for piece info
        piece_features = (piece_info + [0, 0, 0, 0])[:4]
        piece_features = np.array(piece_features, dtype=np.float32).reshape(1, -1)
        
        # Game metrics (normalized) - ensure exactly 4 features
        metrics = np.array([
            min(game_state.get('score', 0) / 10000.0, 1.0),  # Normalize score
            min(game_state.get('holesCount', 0) / 20.0, 1.0),  # Normalize holes
            min(game_state.get('stackHeight', 0) / 20.0, 1.0),  # Normalize height
            1.0 if game_state.get('perfectClear', False) else 0.0
        ], dtype=np.float32).reshape(1, -1)
        
        return {
            'board': torch.FloatTensor(board).to(self.device),
            'piece_info': torch.FloatTensor(piece_features).to(self.device),
            'metrics': torch.FloatTensor(metrics).to(self.device)
        }
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() <= self.epsilon:
            action = random.randrange(self.action_size)
            self.writer.add_scalar('Agent/Random_Action_Taken', 1, self.steps)
            return action
        
        processed_state = self.preprocess_state(state)
        
        with torch.no_grad():
            q_values = self.q_network(
                processed_state['board'],
                processed_state['piece_info'],
                processed_state['metrics']
            )
            
            # Log Q-values statistics
            if training and self.steps % 100 == 0:
                self.writer.add_scalar('Agent/Q_Values_Mean', q_values.mean().item(), self.steps)
                self.writer.add_scalar('Agent/Q_Values_Max', q_values.max().item(), self.steps)
                self.writer.add_scalar('Agent/Q_Values_Min', q_values.min().item(), self.steps)
                self.writer.add_scalar('Agent/Q_Values_Std', q_values.std().item(), self.steps)
            
            action = q_values.argmax().item()
            self.writer.add_scalar('Agent/Random_Action_Taken', 0, self.steps)
            return action
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
        
        # Log memory statistics
        if len(self.memory) % 1000 == 0:
            self.writer.add_scalar('Agent/Memory_Size', len(self.memory), self.steps)
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Ensure we have enough samples for batch norm (minimum 2)
        actual_batch_size = min(self.batch_size, len(self.memory))
        if actual_batch_size < 2:
            return None
        
        batch = random.sample(self.memory, actual_batch_size)
        states = [e[0] for e in batch]
        actions = [e[1] for e in batch]
        rewards = [e[2] for e in batch]
        next_states = [e[3] for e in batch]
        dones = [e[4] for e in batch]
        
        # Process states
        board_batch = []
        piece_batch = []
        metrics_batch = []
        
        next_board_batch = []
        next_piece_batch = []
        next_metrics_batch = []
        
        for state, next_state in zip(states, next_states):
            processed_state = self.preprocess_state(state)
            board_batch.append(processed_state['board'])
            piece_batch.append(processed_state['piece_info'])
            metrics_batch.append(processed_state['metrics'])
            
            if next_state is not None:
                processed_next_state = self.preprocess_state(next_state)
                next_board_batch.append(processed_next_state['board'])
                next_piece_batch.append(processed_next_state['piece_info'])
                next_metrics_batch.append(processed_next_state['metrics'])
            else:
                # Use zeros for terminal states
                next_board_batch.append(torch.zeros_like(processed_state['board']))
                next_piece_batch.append(torch.zeros_like(processed_state['piece_info']))
                next_metrics_batch.append(torch.zeros_like(processed_state['metrics']))
        
        # Combine batches
        board_batch = torch.cat(board_batch)
        piece_batch = torch.cat(piece_batch)
        metrics_batch = torch.cat(metrics_batch)
        
        next_board_batch = torch.cat(next_board_batch)
        next_piece_batch = torch.cat(next_piece_batch)
        next_metrics_batch = torch.cat(next_metrics_batch)
        
        # Current Q values
        current_q_values = self.q_network(board_batch, piece_batch, metrics_batch)
        current_q_values = current_q_values.gather(1, torch.LongTensor(actions).unsqueeze(1).to(self.device))
        
        # Next Q values from target network (Double DQN)
        with torch.no_grad():
            next_q_values_online = self.q_network(next_board_batch, next_piece_batch, next_metrics_batch)
            next_actions = next_q_values_online.argmax(1)
            
            next_q_values_target = self.target_network(next_board_batch, next_piece_batch, next_metrics_batch)
            max_next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        
        # Calculate target Q values
        target_q_values = []
        for i in range(actual_batch_size):
            if dones[i]:
                target_q_values.append(rewards[i])
            else:
                target_q_values.append(rewards[i] + 0.99 * max_next_q_values[i])
        
        target_q_values = torch.FloatTensor(target_q_values).unsqueeze(1).to(self.device)
        
        # Compute loss (Huber loss for stability)
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Log training metrics
        self.training_steps += 1
        self.writer.add_scalar('Training/Loss', loss.item(), self.training_steps)
        self.writer.add_scalar('Training/Epsilon', self.epsilon, self.training_steps)
        self.writer.add_scalar('Training/Learning_Rate', self.scheduler.get_last_lr()[0], self.training_steps)
        self.writer.add_scalar('Training/Q_Values_Current_Mean', current_q_values.mean().item(), self.training_steps)
        self.writer.add_scalar('Training/Q_Values_Target_Mean', target_q_values.mean().item(), self.training_steps)
        
        # Log gradient norms
        total_norm = 0
        for p in self.q_network.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.writer.add_scalar('Training/Gradient_Norm', total_norm, self.training_steps)
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.writer.add_scalar('Training/Target_Network_Update', 1, self.steps)
            print(f"Target network updated at step {self.steps}")
        
        return loss.item()
    
    def log_episode_metrics(self, episode, episode_reward, episode_length, episode_score, 
                          episode_lines, game_metrics):
        """Log episode metrics to TensorBoard"""
        self.writer.add_scalar('Episode/Reward', episode_reward, episode)
        self.writer.add_scalar('Episode/Length', episode_length, episode)
        self.writer.add_scalar('Episode/Score', episode_score, episode)
        self.writer.add_scalar('Episode/Lines_Cleared', episode_lines, episode)
        
        # Game-specific metrics
        self.writer.add_scalar('Game/Holes_Count', game_metrics.get('holes', 0), episode)
        self.writer.add_scalar('Game/Stack_Height', game_metrics.get('stack_height', 0), episode)
        self.writer.add_scalar('Game/Perfect_Clear', 1 if game_metrics.get('perfect_clear', False) else 0, episode)
        
        # Running averages
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        if len(self.episode_rewards) >= 100:
            avg_reward_100 = np.mean(self.episode_rewards[-100:])
            avg_length_100 = np.mean(self.episode_lengths[-100:])
            self.writer.add_scalar('Episode/Avg_Reward_100', avg_reward_100, episode)
            self.writer.add_scalar('Episode/Avg_Length_100', avg_length_100, episode)
        
        if len(self.episode_rewards) >= 10:
            avg_reward_10 = np.mean(self.episode_rewards[-10:])
            avg_length_10 = np.mean(self.episode_lengths[-10:])
            self.writer.add_scalar('Episode/Avg_Reward_10', avg_reward_10, episode)
            self.writer.add_scalar('Episode/Avg_Length_10', avg_length_10, episode)
    
    def log_curriculum_change(self, episode, stage_info):
        """Log curriculum changes"""
        self.writer.add_scalar('Curriculum/Board_Height', stage_info['height'], episode)
        self.writer.add_scalar('Curriculum/Board_Preset', stage_info['preset'], episode)
        self.writer.add_scalar('Curriculum/Tetromino_Types', stage_info['pieces'], episode)
    
    def save(self, filepath):
        """Save the model and training state"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'training_steps': self.training_steps,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
        }
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
        
        # Save model graph to TensorBoard
        try:
            dummy_board = torch.randn(1, 1, 20, 10).to(self.device)
            dummy_piece = torch.randn(1, 4).to(self.device)
            dummy_metrics = torch.randn(1, 4).to(self.device)
            self.writer.add_graph(self.q_network, (dummy_board, dummy_piece, dummy_metrics))
        except Exception as e:
            print(f"Could not save model graph: {e}")
    
    def load(self, filepath):
        """Load the model and training state"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint['steps']
            self.training_steps = checkpoint.get('training_steps', 0)
            self.episode_rewards = checkpoint.get('episode_rewards', [])
            self.episode_lengths = checkpoint.get('episode_lengths', [])
            
            print(f"Model loaded from {filepath}")
            print(f"Resumed at step {self.steps}, epsilon {self.epsilon:.3f}")
            return True
        return False
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()