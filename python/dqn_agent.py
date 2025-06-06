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
        
     # Updated CNN layers for 2-channel input
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)  # Changed: 1 -> 2 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Keep your adaptive pooling - it's perfect for this use case
        self.pool = nn.AdaptiveAvgPool2d((5, 5))

        # Update the flattened size calculation
        # After conv3 + pool: 64 channels * 5 * 5 = 1600
        conv_output_size = 64 * 5 * 5  # 1600

        # Additional input sizes from preprocessing
        piece_info_size = 19  # From _encode_piece_info
        metrics_size = 8      # From _encode_metrics

        # Total input to first fully connected layer
        total_input_size = conv_output_size + piece_info_size + metrics_size  # 1627

        # Fully connected layers
        self.fc1 = nn.Linear(total_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_size)  # output_size = action_size

    
        # Forward pass example   
    def forward(self, state_dict):
        # Extract inputs
        board = state_dict['board']        # Shape: (batch, 2, 20, 10)
        piece_info = state_dict['piece_info']  # Shape: (batch, 19)
        metrics = state_dict['metrics']    # Shape: (batch, 8)
            
        # CNN processing
        x = F.relu(self.conv1(board))      # (batch, 32, 20, 10)
        x = F.relu(self.conv2(x))          # (batch, 64, 20, 10)
        x = F.relu(self.conv3(x))          # (batch, 64, 20, 10)
        x = self.pool(x)                   # (batch, 64, 5, 5)
        x = x.view(x.size(0), -1)          # (batch, 1600)
            
        # Concatenate all features
        combined = torch.cat([x, piece_info, metrics], dim=1)  # (batch, 1627)
            
        # Fully connected layers
        x = F.relu(self.fc1(combined))     # (batch, 512)
        x = F.relu(self.fc2(x))            # (batch, 256)
        x = self.fc3(x)                    # (batch, action_size)
            
        return x

class DQNAgent:
    def __init__(self, state_size,curriculum_stages, action_size=40, lr=0.001, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 tensorboard_log_dir=None,current_curriculum_stage=0):
        self.device = device
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = lr
        self.batch_size = 32  # Reduced batch size for stability
        self.target_update_freq = 1000
        self.steps = 0
        self.training_steps = 0
        
        # Standard board size for normalization
        self.standard_height = 20
        self.standard_width = 10
        
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
        self.current_curriculum_stage =current_curriculum_stage
        self.curriculum_stages = curriculum_stages
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.q_values_history = []
        
        print(f"DQN Agent initialized on {device}")
        print(f"TensorBoard logs: {tensorboard_log_dir}")
        
        # Log model architecture
        self.writer.add_text('Model/Architecture', str(self.q_network))
        self.writer.add_text('Model/Parameters', f"Total parameters: {sum(p.numel() for p in self.q_network.parameters())}")
    
    def normalize_board_size(self, board, current_height, current_width):
        """Normalize board to standard size for consistent processing"""
        if current_height == self.standard_height and current_width == self.standard_width:
            return board
        
        # Reshape to 2D for processing
        board_2d = board.reshape(current_height, current_width)
        
        # Create standard size board (20x10)
        normalized_board = np.zeros((self.standard_height, self.standard_width), dtype=np.float32)
        
        # Copy existing board data
        copy_height = min(current_height, self.standard_height)
        copy_width = min(current_width, self.standard_width)
        
        # Place the smaller board at the bottom (gravity effect)
        start_row = self.standard_height - copy_height
        normalized_board[start_row:start_row + copy_height, :copy_width] = board_2d[:copy_height, :copy_width]
        
        return normalized_board.flatten()

    def update_curriculum_stage(self, stage):
        self.current_curriculum_stage = stage    


    def preprocess_state(self, game_state):
        """Convert game state to neural network input - fixed size with curriculum awareness"""
        if not isinstance(game_state, dict) or 'board' not in game_state:
            return self._create_empty_state()
        
        # Get current curriculum stage
        current_stage = self.curriculum_stages[self.current_curriculum_stage]
        curr_height = current_stage['height']
        
        # Always use standard 20x10 board for network input
        board_data = game_state.get('board', [])
        try:
            board = np.array(board_data, dtype=np.float32)
            if board.size != curr_height * 10:
                board = np.zeros(curr_height * 10)
            
            # Reshape to current curriculum size then pad/crop to standard size
            curr_board = board.reshape(curr_height, 10)
            
            # Create standard 20x10 board
            standard_board = np.zeros((self.standard_height, self.standard_width), dtype=np.float32)
            
            # Place curriculum board at the BOTTOM (like real Tetris)
            start_row = self.standard_height - curr_height
            standard_board[start_row:, :] = curr_board
            
            # Add curriculum mask as second channel
            mask = np.zeros((self.standard_height, self.standard_width), dtype=np.float32)
            mask[start_row:, :] = 1.0  # Mark active area
            
            # Stack board and mask as channels
            board_with_mask = np.stack([standard_board, mask], axis=0)  # Shape: (2, 20, 10)
            board_with_mask = board_with_mask.reshape(1, 2, self.standard_height, self.standard_width)
            
        except Exception as e:
            print(f"Board processing error: {e}")
            board_with_mask = np.zeros((1, 2, self.standard_height, self.standard_width), dtype=np.float32)
            # Set bottom rows as active for current curriculum
            start_row = self.standard_height - current_stage['height']
            board_with_mask[0, 1, start_row:, :] = 1.0
        
        # Enhanced piece information
        piece_info = self._encode_piece_info(game_state, current_stage)
        
        # Curriculum-adaptive game metrics
        metrics = self._encode_metrics(game_state, current_stage)
        
        return {
            'board': torch.FloatTensor(board_with_mask).to(self.device),
            'piece_info': torch.FloatTensor(piece_info).to(self.device),
            'metrics': torch.FloatTensor(metrics).to(self.device)
        }

    def _encode_piece_info(self, game_state, current_stage):
        """Encode current piece information"""
        # Piece type encoding (one-hot for 7 piece types)
        piece_type = game_state.get('currentPieceType', 0)  # 0-6 for I,O,T,S,Z,J,L
        piece_onehot = np.zeros(7, dtype=np.float32)
        if 0 <= piece_type < 7:
            piece_onehot[piece_type] = 1.0
        
        # Piece position and rotation (normalized to standard board)
        piece_x = game_state.get('currentPieceX', 0) / 10.0  # Normalize to [0,1]
        piece_y = game_state.get('currentPieceY', 0) / 20.0  # Always normalize to full board
        piece_rotation = game_state.get('currentPieceRotation', 0) / 3.0  # 0-3 rotations
        
        # Next piece info (if available)
        next_piece = game_state.get('nextPieceType', 0)
        next_onehot = np.zeros(7, dtype=np.float32)
        if 0 <= next_piece < 7:
            next_onehot[next_piece] = 1.0
        
        # Curriculum stage info
        stage_features = [
            current_stage['height'] / 20.0,  # Relative height
            float(self.current_curriculum_stage) / len(self.curriculum_stages),  # Progress
        ]
        
        # Combine all piece info
        piece_features = np.concatenate([
            piece_onehot,      # 7 features
            [piece_x, piece_y, piece_rotation],  # 3 features
            next_onehot,       # 7 features
            stage_features     # 2 features
        ]).reshape(1, -1)  # Total: 19 features
        
        return piece_features

    def _encode_metrics(self, game_state, current_stage):
        """Encode game metrics - curriculum adaptive"""
        try:
            score = float(game_state.get('score', 0))
            lines_cleared = float(game_state.get('linesCleared', 0))
            level = float(game_state.get('level', 1))
            holes = float(game_state.get('holesCount', 0))
            stack_height = float(game_state.get('stackHeight', 0))
            perfect_clear = float(game_state.get('perfectClear', False))
            
            # Adaptive normalization based on curriculum
            curr_height = current_stage['height']
            max_expected_score = curr_height * 1000  # Rough estimate
            
            metrics = np.array([
                min(score / max_expected_score, 1.0),  # Normalized score
                min(lines_cleared / 100.0, 1.0),      # Lines cleared
                min(level / 10.0, 1.0),               # Level
                min(holes / (curr_height * 2), 1.0),  # Holes relative to board size
                stack_height / curr_height,           # Stack height ratio (curriculum adaptive)
                perfect_clear,                        # Perfect clear flag
                curr_height / 20.0,                   # Current difficulty indicator
                float(self.current_curriculum_stage) / len(self.curriculum_stages)  # Overall progress
            ], dtype=np.float32).reshape(1, -1)
            
        except (ValueError, TypeError):
            metrics = np.zeros((1, 8), dtype=np.float32)
        
        return metrics

    def _create_empty_state(self):
        """Create empty state for error cases"""
        current_stage = self.curriculum_stages[self.current_curriculum_stage]
        
        # Create empty board with mask
        empty_board = np.zeros((1, 2, self.standard_height, self.standard_width), dtype=np.float32)
        # Set active area mask
        start_row = self.standard_height - current_stage['height']
        empty_board[0, 1, start_row:, :] = 1.0
        
        return {
            'board': torch.FloatTensor(empty_board).to(self.device),
            'piece_info': torch.zeros((1, 19), dtype=torch.float32).to(self.device),
            'metrics': torch.zeros((1, 8), dtype=torch.float32).to(self.device)
        }  
    def act(self, state, training=True):
            """Choose action using epsilon-greedy policy"""
            if training and random.random() <= self.epsilon:
                action = random.randrange(self.action_size)
                self.writer.add_scalar('Agent/Random_Action_Taken', 1, self.steps)
                return action
            
            processed_state = self.preprocess_state(state)
            
            with torch.no_grad():
                q_values = self.q_network(processed_state)
                
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
        
        if len(self.memory) % 1000 == 0:
            self.writer.add_scalar('Agent/Memory_Size', len(self.memory), self.steps)
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return None
        
        actual_batch_size = min(self.batch_size, len(self.memory))
        if actual_batch_size < 2:
            return None
        
        batch = random.sample(self.memory, actual_batch_size)
        
        # Process each experience individually to handle different board sizes
        board_tensors = []
        piece_tensors = []
        metrics_tensors = []
        next_board_tensors = []
        next_piece_tensors = []
        next_metrics_tensors = []
        
        actions = []
        rewards = []
        dones = []
        
        for state, action, reward, next_state, done in batch:
            # Process current state
            processed_state = self.preprocess_state(state)
            board_tensors.append(processed_state['board'])
            piece_tensors.append(processed_state['piece_info'])
            metrics_tensors.append(processed_state['metrics'])
            
            # Process next state
            if next_state is not None:
                processed_next_state = self.preprocess_state(next_state)
                next_board_tensors.append(processed_next_state['board'])
                next_piece_tensors.append(processed_next_state['piece_info'])
                next_metrics_tensors.append(processed_next_state['metrics'])
            else:
                # Terminal state - use zeros with standard dimensions
                empty_state = self._create_empty_state()
                next_board_tensors.append(empty_state['board'])
                next_piece_tensors.append(empty_state['piece_info'])
                next_metrics_tensors.append(empty_state['metrics'])
            
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        
        # Now all tensors should have the same dimensions and can be concatenated
        try:
            board_batch = torch.cat(board_tensors, dim=0)
            piece_batch = torch.cat(piece_tensors, dim=0)
            metrics_batch = torch.cat(metrics_tensors, dim=0)
            
            next_board_batch = torch.cat(next_board_tensors, dim=0)
            next_piece_batch = torch.cat(next_piece_tensors, dim=0)
            next_metrics_batch = torch.cat(next_metrics_tensors, dim=0)
        except RuntimeError as e:
            print(f"Error concatenating tensors: {e}")
            return None
        
        # Current Q values
        state_batch_dict = {'board': board_batch, 'piece_info': piece_batch, 'metrics': metrics_batch}
        next_state_batch_dict = {'board': next_board_batch, 'piece_info': next_piece_batch, 'metrics': next_metrics_batch}
        
        # Current Q values
        current_q_values = self.q_network(state_batch_dict)
        current_q_values = current_q_values.gather(1, torch.LongTensor(actions).unsqueeze(1).to(self.device))
        
        # Next Q values from target network (Double DQN)
        with torch.no_grad():
            next_q_values_online = self.q_network(next_state_batch_dict)
            next_actions = next_q_values_online.argmax(1)
            
            next_q_values_target = self.target_network(next_state_batch_dict)
            max_next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        
        # Calculate target Q values
        target_q_values = []
        for i in range(actual_batch_size):
            if dones[i]:
                target_q_values.append(rewards[i])
            else:
                target_q_values.append(rewards[i] + 0.99 * max_next_q_values[i])
        
        target_q_values = torch.FloatTensor(target_q_values).unsqueeze(1).to(self.device)
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
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
        self.writer.add_scalar('Game/Holes_Count', game_metrics.get('holes_count', 0), episode)
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