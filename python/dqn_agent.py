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
        
        # Use the new QNetwork architecture
        standard_height = 20  # Define this constant
        
        # Board CNN encoder
        self.board_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * standard_height * 10, 128), nn.ReLU()
        )
        
        # MLPs for other inputs
        self.piece_mlp = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
        self.metrics_mlp = nn.Sequential(nn.Linear(5, 32), nn.ReLU()) 
        self.heights_mlp = nn.Sequential(nn.Linear(10, 32), nn.ReLU())
        self.next_piece_mlp = nn.Sequential(nn.Linear(7, 32), nn.ReLU())
        self.curriculum_mlp = nn.Sequential(nn.Linear(9, 32), nn.ReLU())
        
        # Final fusion & head
        total_size = 128 + 32*5  # 288
        self.fc_out = nn.Sequential(
            nn.Linear(total_size, 128), nn.ReLU(),
            nn.Linear(128, output_size)
        )
    
    def forward(self, board, piece_info, metrics, heights, next_piece, curriculum):
        # Process each input through its dedicated pathway
        board_features = self.board_cnn(board)
        piece_features = self.piece_mlp(piece_info)
        metrics_features = self.metrics_mlp(metrics)
        heights_features = self.heights_mlp(heights)
        next_piece_features = self.next_piece_mlp(next_piece)
        curriculum_features = self.curriculum_mlp(curriculum)
        
        # Fuse all features
        combined = torch.cat([
            board_features, piece_features, metrics_features,
            heights_features, next_piece_features, curriculum_features
        ], dim=1)
        
        return self.fc_out(combined)

class DQNAgent:
    def __init__(self, state_size, action_size=40, lr=0.001, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 tensorboard_log_dir=None):
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
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.q_values_history = []
        
        print(f"DQN Agent initialized on {device}")
        print(f"TensorBoard logs: {tensorboard_log_dir}")

        # Add curriculum-specific exploration parameters
        self.base_epsilon = 1.0  # Store original epsilon for resets
        self.curriculum_epsilon_boost = 0.3  # How much to boost epsilon on curriculum change
        self.curriculum_decay_episodes = 200  # Episodes over which to decay the boost
        self.curriculum_boost_active = False
        self.curriculum_boost_episode_start = 0
        
        # Log model architecture
        self.writer.add_text('Model/Architecture', str(self.q_network))
        self.writer.add_text('Model/Parameters', f"Total parameters: {sum(p.numel() for p in self.q_network.parameters())}")

    def reset_exploration_for_curriculum(self, boost_factor=0.3, decay_episodes=200):
        """Reset/boost exploration when curriculum changes"""
        print(f"🔄 Boosting exploration: epsilon {self.epsilon:.3f} → {min(1.0, self.epsilon + boost_factor):.3f}")
        
        # Boost epsilon for new curriculum stage
        self.epsilon = min(1.0, self.epsilon + boost_factor)
        self.curriculum_epsilon_boost = boost_factor
        self.curriculum_decay_episodes = decay_episodes
        self.curriculum_boost_active = True
        self.curriculum_boost_episode_start = 0  # Will be set by trainer
        
        # Log the exploration reset
        self.writer.add_scalar('Curriculum/Epsilon_Boost', boost_factor, self.steps)
        self.writer.add_scalar('Agent/Epsilon_After_Curriculum_Change', self.epsilon, self.steps)
    
    def update_curriculum_epsilon(self, episodes_since_change):
        """Gradually decay curriculum epsilon boost"""
        if not self.curriculum_boost_active:
            return
            
        if episodes_since_change >= self.curriculum_decay_episodes:
            self.curriculum_boost_active = False
            return
        
        # Linearly decay the boost over time
        decay_progress = episodes_since_change / self.curriculum_decay_episodes
        current_boost = self.curriculum_epsilon_boost * (1 - decay_progress)
        
        # Calculate what epsilon should be without the boost
        base_epsilon = max(self.epsilon_min, self.base_epsilon * (self.epsilon_decay ** self.steps))
        
        # Apply the decaying boost
        self.epsilon = min(1.0, base_epsilon + current_boost)
        
        # Log curriculum epsilon decay
        if episodes_since_change % 10 == 0:
            self.writer.add_scalar('Curriculum/Epsilon_Boost_Remaining', current_boost, self.steps)
            self.writer.add_scalar('Curriculum/Episodes_Since_Change', episodes_since_change, self.steps)

    
    def normalize_board_size(self, board, current_height, current_width):
        """Normalize board to standard size for consistent processing"""
        # Ensure parameters are integers
        current_height = int(current_height)
        current_width = int(current_width)
        
        if current_height == self.standard_height and current_width == self.standard_width:
            return board
        
        # Ensure board is the right size for reshaping
        expected_size = current_height * current_width
        if len(board) != expected_size:
            print(f"Warning: Board size mismatch. Expected {expected_size}, got {len(board)}")
            # Pad or truncate as needed
            if len(board) < expected_size:
                board = np.pad(board, (0, expected_size - len(board)), 'constant')
            else:
                board = board[:expected_size]
        
        # Reshape to 2D for processing
        board_2d = board.reshape(current_height, current_width)
        
        # Create standard size board (20x10)
        normalized_board = np.zeros((self.standard_height, self.standard_width), dtype=np.float32)
        
        # Copy existing board data - place smaller boards at the bottom (gravity effect)
        copy_height = min(current_height, self.standard_height)
        copy_width = min(current_width, self.standard_width)
        
        start_row = self.standard_height - copy_height
        normalized_board[start_row:start_row + copy_height, :copy_width] = board_2d[:copy_height, :copy_width]
        
        return normalized_board.flatten()
    
    def preprocess_state(self, game_state):
        """Convert game state to neural network input with extended features"""

        if not isinstance(game_state, dict) or 'board' not in game_state:
            print("Warning: preprocess_state received invalid game_state.")
            return {
                'board': torch.zeros((1, 1, self.standard_height, self.standard_width), dtype=torch.float32).to(self.device),
                'piece_info': torch.zeros((1, 4), dtype=torch.float32).to(self.device),
                'metrics': torch.zeros((1, 5), dtype=torch.float32).to(self.device),
                'heights': torch.zeros((1, 10), dtype=torch.float32).to(self.device),
                'next_piece': torch.zeros((1, 7), dtype=torch.float32).to(self.device),
                'curriculum': torch.zeros((1, 9), dtype=torch.float32).to(self.device)
            }

        board_data = game_state.get('board', [])
        if not board_data or len(board_data) == 0:
            board_array = np.zeros(self.standard_height * self.standard_width, dtype=np.float32)
        else:
            board_array = np.array(board_data, dtype=np.float32)
            
            # CRITICAL FIX: Always normalize board to standard size
            current_height = int(game_state.get('curriculumBoardHeight', self.standard_height))
            current_width = self.standard_width  # Always 10 for Tetris
            
            # Normalize the board size
            board_array = self.normalize_board_size(board_array, current_height, current_width)

        # Always reshape to standard dimensions (20x10)
        board_tensor = torch.tensor(
            board_array.reshape(1, 1, self.standard_height, self.standard_width), 
            dtype=torch.float32
        ).to(self.device)

        # Current piece
        piece = game_state.get('currentPiece', [0, 0, 0, 0])
        piece_info_tensor = torch.tensor([piece], dtype=torch.float32).to(self.device)

        # Metrics
        holes = float(game_state.get('holesCount', 0))
        stack_height = float(game_state.get('stackHeight', 0))
        perfect_clear = float(game_state.get('perfectClear', False))
        bumpiness = float(game_state.get('bumpiness', 0))
        covered = float(game_state.get('covered', 0))
        
        metrics_tensor = torch.tensor([[holes, stack_height, perfect_clear, bumpiness, covered]], 
                                    dtype=torch.float32).to(self.device)

        # Heights - normalize to standard width (10)
        heights = game_state.get('heights', [0]*10)
        if len(heights) != 10:
            # Pad or truncate to 10 columns
            normalized_heights = [0] * 10
            for i in range(min(len(heights), 10)):
                normalized_heights[i] = heights[i]
            heights = normalized_heights
        
        heights = np.array(heights, dtype=np.float32)
        heights_tensor = torch.tensor(heights, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Next piece (one-hot)
        next_piece_index = game_state.get('nextPiece', [0])[0]
        next_piece_one_hot = np.zeros(7, dtype=np.float32)
        if 0 <= next_piece_index < 7:
            next_piece_one_hot[next_piece_index] = 1.0
        next_piece_tensor = torch.tensor(next_piece_one_hot, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Curriculum
        allowed_count = int(game_state.get('allowedTetrominoTypes', 7))
        allowed_mask = np.zeros(7, dtype=np.float32)
        allowed_mask[:allowed_count] = 1.0
        board_h = float(game_state.get('curriculumBoardHeight', 20))
        preset = float(game_state.get('curriculumBoardPreset', 0))
        curriculum_vector = np.concatenate([[board_h / 20.0], [preset / 5.0], allowed_mask])
        curriculum_tensor = torch.tensor(curriculum_vector, dtype=torch.float32).unsqueeze(0).to(self.device)

        return {
            'board': board_tensor,
            'piece_info': piece_info_tensor,
            'metrics': metrics_tensor,
            'heights': heights_tensor,
            'next_piece': next_piece_tensor,
            'curriculum': curriculum_tensor
        }
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy over valid actions only"""
        valid_actions = state['validActions']
        if not valid_actions:
            print("❌ No valid actions available — returning fallback or skipping step.")
            return 0

        if training and random.random() <= self.epsilon:
            action = random.choice(valid_actions)
            self.writer.add_scalar('Agent/Random_Action_Taken', 1, self.steps)
            return action

        processed_state = self.preprocess_state(state)
        
        with torch.no_grad():
            # Update to pass all features to the network
            q_values = self.q_network(
                processed_state['board'],
                processed_state['piece_info'],
                processed_state['metrics'],
                processed_state['heights'],      # Add this
                processed_state['next_piece'],   # Add this
                processed_state['curriculum']    # Add this
            )

            # Rest remains the same...
            masked_q_values = torch.full_like(q_values, float('-inf'))
            masked_q_values[0, valid_actions] = q_values[0, valid_actions]
            best_action = masked_q_values.argmax().item()
            self.writer.add_scalar('Agent/Random_Action_Taken', 0, self.steps)
            
            if training and self.steps % 100 == 0:
                self.writer.add_scalar('Agent/Q_Values_Mean', q_values[0, valid_actions].mean().item(), self.steps)
                self.writer.add_scalar('Agent/Q_Values_Max', q_values[0, valid_actions].max().item(), self.steps)
                self.writer.add_scalar('Agent/Q_Values_Min', q_values[0, valid_actions].min().item(), self.steps)
                self.writer.add_scalar('Agent/Q_Values_Std', q_values[0, valid_actions].std().item(), self.steps)

            return best_action
    
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
        
        # Process each experience - ADD the new tensor lists
        board_tensors = []
        piece_tensors = []
        metrics_tensors = []
        heights_tensors = []        # Add this
        next_piece_tensors = []     # Add this
        curriculum_tensors = []     # Add this
        
        next_board_tensors = []
        next_piece_tensors_state = []
        next_metrics_tensors = []
        next_heights_tensors = []        # Add this
        next_next_piece_tensors = []     # Add this
        next_curriculum_tensors = []     # Add this
        
        actions = []
        rewards = []
        dones = []
        
        for state, action, reward, next_state, done in batch:
            # Process current state
            processed_state = self.preprocess_state(state)
            board_tensors.append(processed_state['board'])
            piece_tensors.append(processed_state['piece_info'])
            metrics_tensors.append(processed_state['metrics'])
            heights_tensors.append(processed_state['heights'])          # Add this
            next_piece_tensors.append(processed_state['next_piece'])    # Add this
            curriculum_tensors.append(processed_state['curriculum'])    # Add this
            
            # Process next state
            if next_state is not None:
                processed_next_state = self.preprocess_state(next_state)
                next_board_tensors.append(processed_next_state['board'])
                next_piece_tensors_state.append(processed_next_state['piece_info'])
                next_metrics_tensors.append(processed_next_state['metrics'])
                next_heights_tensors.append(processed_next_state['heights'])          # Add this
                next_next_piece_tensors.append(processed_next_state['next_piece'])    # Add this
                next_curriculum_tensors.append(processed_next_state['curriculum'])    # Add this
            else:
                # Terminal state - use zeros with standard dimensions
                next_board_tensors.append(torch.zeros((1, 1, self.standard_height, self.standard_width), dtype=torch.float32).to(self.device))
                next_piece_tensors_state.append(torch.zeros((1, 4), dtype=torch.float32).to(self.device))
                next_metrics_tensors.append(torch.zeros((1, 5), dtype=torch.float32).to(self.device))  # Fixed: was 4, should be 5
                next_heights_tensors.append(torch.zeros((1, 10), dtype=torch.float32).to(self.device))     # Add this
                next_next_piece_tensors.append(torch.zeros((1, 7), dtype=torch.float32).to(self.device))   # Add this
                next_curriculum_tensors.append(torch.zeros((1, 9), dtype=torch.float32).to(self.device))   # Add this
            
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        
        # Concatenate all tensors - ADD the new ones
        try:
            board_batch = torch.cat(board_tensors, dim=0)
            piece_batch = torch.cat(piece_tensors, dim=0)
            metrics_batch = torch.cat(metrics_tensors, dim=0)
            heights_batch = torch.cat(heights_tensors, dim=0)          # Add this
            next_piece_batch = torch.cat(next_piece_tensors, dim=0)    # Add this
            curriculum_batch = torch.cat(curriculum_tensors, dim=0)    # Add this
            
            next_board_batch = torch.cat(next_board_tensors, dim=0)
            next_piece_state_batch = torch.cat(next_piece_tensors_state, dim=0)
            next_metrics_batch = torch.cat(next_metrics_tensors, dim=0)
            next_heights_batch = torch.cat(next_heights_tensors, dim=0)          # Add this
            next_next_piece_batch = torch.cat(next_next_piece_tensors, dim=0)    # Add this
            next_curriculum_batch = torch.cat(next_curriculum_tensors, dim=0)    # Add this
        except RuntimeError as e:
            print(f"Error concatenating tensors: {e}")
            return None
        
        # Current Q values - UPDATE to pass all features
        current_q_values = self.q_network(
            board_batch, piece_batch, metrics_batch, 
            heights_batch, next_piece_batch, curriculum_batch
        )
        current_q_values = current_q_values.gather(1, torch.LongTensor(actions).unsqueeze(1).to(self.device))
        
        # Next Q values - UPDATE to pass all features
        with torch.no_grad():
            next_q_values_online = self.q_network(
                next_board_batch, next_piece_state_batch, next_metrics_batch,
                next_heights_batch, next_next_piece_batch, next_curriculum_batch
            )
            next_actions = next_q_values_online.argmax(1)
            
            next_q_values_target = self.target_network(
                next_board_batch, next_piece_state_batch, next_metrics_batch,
                next_heights_batch, next_next_piece_batch, next_curriculum_batch
            )
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