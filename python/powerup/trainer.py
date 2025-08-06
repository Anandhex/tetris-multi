# trainer.py - Enhanced version with fixes
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
from typing import List, Tuple

class PowerupTrainer:
    """Enhanced training class for powerup DQN with overfitting prevention"""
    
    def __init__(self, dataset_path: str, save_dir: str = "models"):
        from python.powerup.environments import TrainingEnvironment
        from powerup_dqn_agent import PowerupDQNAgent
        
        self.environment = TrainingEnvironment(dataset_path)
        
        # FIXED agent configuration for stability
        self.agent = PowerupDQNAgent(
            epsilon=1.0,           # Start with full exploration
            epsilon_min=0.05,      # Lower minimum for more exploration
            epsilon_decay=0.9998,  # Even slower decay
            learning_rate=0.0001,  # Much lower learning rate for stability
            memory_size=15000,     # Reasonable memory size
            batch_size=32,         # Smaller batch for more frequent updates
            gamma=0.99,            # Higher discount factor
            tau=0.001              # Very slow target network updates
        )
        
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Enhanced metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.action_usage = {'none': 0, 'bottom_clear': 0, 'gravity': 0, 'bomb': 0}
        self.validation_rewards = []
        
        # Adjusted early stopping for more patience
        self.best_validation_reward = -float('inf')
        self.patience = 800  # Increased patience
        self.patience_counter = 0
        
        print("Enhanced trainer initialized:")
        self.agent.print_model_info()
    
    def balanced_reward_function(self, old_board: np.ndarray, new_board: np.ndarray, 
                                action: dict, powerups_available: dict) -> float:
        """Enhanced reward function that STRONGLY discourages 'none' overuse"""
        
        # Base improvement reward (SCALED DOWN to prevent huge values)
        old_quality = self._evaluate_board_quality(old_board)
        new_quality = self._evaluate_board_quality(new_board)
        improvement = new_quality - old_quality
        
        base_reward = improvement * 0.1  # Reduced from 10 to 0.1
        
        # Action-specific bonuses (MUCH SMALLER SCALE)
        if action['type'] == 'bottom_clear':
            blocks_cleared = np.sum(old_board[-1, :])
            base_reward += blocks_cleared * 0.8 + 3  # INCREASED bonus
            
        elif action['type'] == 'gravity':
            old_holes = self._count_holes(old_board)
            new_holes = self._count_holes(new_board)
            holes_filled = old_holes - new_holes
            base_reward += holes_filled * 1.5 + 3  # INCREASED bonus
            
        elif action['type'] == 'bomb':
            blocks_destroyed = np.sum(old_board) - np.sum(new_board)
            if blocks_destroyed > 0:
                base_reward += blocks_destroyed * 0.5 + 3  # INCREASED bonus
            else:
                base_reward -= 2  # Penalty for useless bomb
                
        elif action['type'] == 'none':
            # MUCH STRONGER penalties to heavily discourage 'none'
            total_actions = sum(self.action_usage.values())
            
            if total_actions > 50:  # After some actions
                none_usage = self.action_usage.get('none', 0)
                none_percentage = none_usage / total_actions
                
                # Progressive penalties get MUCH stronger
                if none_percentage > 0.5:  # If 'none' used more than 50%
                    base_reward -= 8  # VERY strong penalty
                elif none_percentage > 0.4:  # If 'none' used more than 40%
                    base_reward -= 5  # Strong penalty
                elif none_percentage > 0.3:  # If 'none' used more than 30%
                    base_reward -= 3  # Moderate penalty
                elif none_percentage > 0.25:  # If 'none' used more than 25%
                    base_reward -= 1  # Small penalty
            
            # STRONGER penalties if good powerups were available but not used
            available_count = sum(powerups_available.values())
            if available_count > 0:
                base_reward -= 2 * available_count  # Penalty per unused powerup
                
                # Specific penalties for not using obviously good powerups
                if powerups_available.get('bottom_clear', False):
                    bottom_blocks = np.sum(old_board[-1, :])
                    if bottom_blocks >= 6:  # If bottom row is 60%+ filled
                        base_reward -= 4  # Strong penalty for not clearing
                    elif bottom_blocks >= 4:  # If bottom row is 40%+ filled
                        base_reward -= 2  # Moderate penalty
                
                if powerups_available.get('gravity', False):
                    holes = self._count_holes(old_board)
                    if holes >= 5:  # Many holes
                        base_reward -= 4  # Strong penalty for not using gravity
                    elif holes >= 3:  # Some holes
                        base_reward -= 2  # Moderate penalty
                
                if powerups_available.get('bomb', False):
                    if self._has_good_bomb_targets(old_board):
                        base_reward -= 3  # Penalty for not using bomb when good targets exist
        
        # Enhanced action diversity bonus - REWARD using underused actions more
        total_actions = sum(self.action_usage.values())
        if total_actions > 100:  # After some training
            action_frequency = self.action_usage.get(action['type'], 0) / total_actions
            
            # Strong bonuses for balanced usage (target: 25% each)
            if action['type'] != 'none':
                if action_frequency < 0.20:  # Underused action
                    base_reward += 3  # Strong bonus for underused powerups
                elif action_frequency < 0.25:  # Slightly underused
                    base_reward += 1  # Small bonus
            
            # Penalty for overusing any action (especially 'none')
            if action_frequency > 0.4:  # Overused action
                if action['type'] == 'none':
                    base_reward -= 5  # Extra penalty for overusing 'none'
                else:
                    base_reward -= 2  # Penalty for overusing any action
        
        # Clip reward to reasonable range [-15, 15]
        base_reward = np.clip(base_reward, -15, 15)
        
        return float(base_reward)
    
    def train(self, episodes: int = 5000, target_update_freq: int = 200,
              save_freq: int = 500, render_freq: int = 100):
        """Enhanced training loop with validation and early stopping"""
        
        print(f"Starting enhanced training for {episodes} episodes...")
        
        for episode in range(episodes):
            # Reset environment
            self.environment.reset()
            
            episode_reward = 0
            episode_length = 0
            episode_loss = 0
            
            # Shorter episodes to prevent reward explosion
            for step in range(8):  # Reduced from 15 to 8
                # Get current state
                current_features = self.environment.get_features()
                
                # Choose action
                action = self.agent.choose_action(self.environment)
                action_id = self.agent.action_space.encode_action(action['type'])
                
                # Store board state before action
                old_board = self.environment.get_board_state().copy()
                powerups_before = self.environment.get_powerup_availability().copy()
                
                # Apply action
                new_board, original_reward = self.environment.apply_powerup(action)
                
                # Use enhanced reward function
                enhanced_reward = self.balanced_reward_function(
                    old_board, new_board, action, powerups_before
                )
                
                # Get next state
                next_features = self.environment.get_features()
                
                # Check if any powerups are left
                powerups_left = any(self.environment.get_powerup_availability().values())
                done = not powerups_left
                
                # Store experience with enhanced reward
                self.agent.remember(current_features, action_id, enhanced_reward, next_features, done)
                
                episode_reward += enhanced_reward
                episode_length += 1
                self.action_usage[action['type']] += 1
                
                if done:
                    break
            
            # Train the agent more frequently but with smaller updates
            if len(self.agent.memory) > self.agent.batch_size:
                loss = self.agent.train()
                episode_loss = loss if loss > 0 else 0
            
            # Less frequent hard updates
            if episode % target_update_freq == 0:
                self.agent.hard_update()
            
            # Record metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.losses.append(episode_loss)
            
            # More frequent validation and logging
            if episode % render_freq == 0 and episode > 0:
                val_reward = self.validate_model()
                self.validation_rewards.append(val_reward)
                
                # Early stopping check
                if val_reward > self.best_validation_reward:
                    self.best_validation_reward = val_reward
                    self.patience_counter = 0
                    # Save best model
                    best_model_path = os.path.join(self.save_dir, "best_model.pth")
                    self.agent.save_model(best_model_path)
                    print(f"New best validation reward: {val_reward:.2f}")
                else:
                    self.patience_counter += render_freq
                
                # Check for early stopping
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at episode {episode} - no improvement for {self.patience} episodes")
                    break
                
                # Enhanced logging
                avg_reward = np.mean(self.episode_rewards[-render_freq:])
                avg_loss = np.mean(self.losses[-render_freq:]) if self.losses[-render_freq:] else 0
                
                # Calculate action distribution
                total_actions = sum(self.action_usage.values())
                action_dist = {k: (v/total_actions)*100 for k, v in self.action_usage.items()} if total_actions > 0 else {}
                
                print(f"Episode {episode}:")
                print(f"  Training Avg Reward: {avg_reward:.2f}")
                print(f"  Validation Reward: {val_reward:.2f}")
                print(f"  Avg Loss: {avg_loss:.4f}")
                print(f"  Action Distribution: {action_dist}")
                print(f"  Epsilon: {self.agent.epsilon:.3f}")
                print(f"  Memory: {len(self.agent.memory)}")
                
                # ENHANCED WARNING system for action imbalance
                if 'none' in action_dist:
                    none_pct = action_dist['none']
                    if none_pct > 45:
                        print(f"  🔴 CRITICAL: 'none' severely overused ({none_pct:.1f}%)! Increasing penalties...")
                        # Temporarily reduce epsilon to encourage more exploration of powerups
                        self.agent.epsilon = max(self.agent.epsilon * 0.95, 0.1)
                    elif none_pct > 35:
                        print(f"  🟡 WARNING: 'none' overused ({none_pct:.1f}%)! Should be ~25%")
                    elif 20 <= none_pct <= 30:
                        print(f"  ✅ GOOD: Balanced 'none' usage ({none_pct:.1f}%)")
                
                # Check if any powerup is underused
                for action, pct in action_dist.items():
                    if action != 'none' and pct < 15:
                        print(f"  📈 BOOST: '{action}' underused ({pct:.1f}%) - increasing bonuses")
                
                # Check CUDA memory if available
                try:
                    from powerup_dqn_agent import check_cuda_memory
                    check_cuda_memory()
                except:
                    pass
            
            # Save model periodically
            if episode % save_freq == 0 and episode > 0:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_path = os.path.join(self.save_dir, f"powerup_model_ep{episode}_{timestamp}.pth")
                self.agent.save_model(model_path)
        
        # Final save
        final_model_path = os.path.join(self.save_dir, "powerup_model_final.pth")
        self.agent.save_model(final_model_path)
        
        self.plot_enhanced_metrics()
        
        return final_model_path
    
    def validate_model(self, num_tests: int = 50):
        """Validate model on unseen data"""
        old_epsilon = self.agent.epsilon
        self.agent.epsilon = 0  # No exploration during validation
        
        total_reward = 0
        for _ in range(num_tests):
            self.environment.reset()
            action = self.agent.choose_action(self.environment)
            _, reward = self.environment.apply_powerup(action)
            total_reward += reward
        
        self.agent.epsilon = old_epsilon  # Restore epsilon
        return total_reward / num_tests
    
    def plot_enhanced_metrics(self):
        """Plot comprehensive training metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards with validation - FIX: Match array lengths
        ax1.plot(self.episode_rewards, label='Training', alpha=0.7)
        if self.validation_rewards:
            # Create matching episode numbers for validation rewards
            val_episodes = []
            for i, _ in enumerate(self.validation_rewards):
                episode_num = (i + 1) * 100  # Validation every 100 episodes
                if episode_num <= len(self.episode_rewards):
                    val_episodes.append(episode_num)
            
            # Only plot if we have matching lengths
            if len(val_episodes) == len(self.validation_rewards):
                ax1.plot(val_episodes, self.validation_rewards, 'r-', label='Validation', linewidth=2)
            else:
                # Fallback: just plot validation rewards against their indices
                ax1.plot(range(0, len(self.validation_rewards) * 100, 100), 
                        self.validation_rewards, 'r-', label='Validation', linewidth=2)
        
        ax1.set_title('Training vs Validation Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True)
        
        # Action distribution
        total_actions = sum(self.action_usage.values())
        if total_actions > 0:
            action_percentages = {k: (v/total_actions)*100 for k, v in self.action_usage.items()}
            bars = ax2.bar(action_percentages.keys(), action_percentages.values())
            
            # Color bars: red for 'none' if overused, green for others
            for i, (action, percentage) in enumerate(action_percentages.items()):
                if action == 'none' and percentage > 35:
                    bars[i].set_color('red')
                else:
                    bars[i].set_color('green')
            
            ax2.set_title('Action Distribution (%)')
            ax2.set_ylabel('Percentage')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add percentage labels on bars
            for i, (action, percentage) in enumerate(action_percentages.items()):
                ax2.text(i, percentage + 1, f'{percentage:.1f}%', ha='center')
        
        # Moving average rewards
        window = 100
        if len(self.episode_rewards) >= window:
            moving_avg = [np.mean(self.episode_rewards[i:i+window]) 
                         for i in range(len(self.episode_rewards)-window)]
            ax3.plot(moving_avg)
            ax3.set_title(f'Moving Average Reward (window={window})')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Average Reward')
            ax3.grid(True)
        
        # Training loss
        if self.losses:
            non_zero_losses = [loss for loss in self.losses if loss > 0]
            if non_zero_losses:
                ax4.plot(non_zero_losses)
                ax4.set_title('Training Loss')
                ax4.set_xlabel('Training Step')
                ax4.set_ylabel('MSE Loss')
                ax4.set_yscale('log')
                ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'enhanced_training_metrics.png'), dpi=300)
        plt.show()
    
    # Helper methods for reward calculation
    def _evaluate_board_quality(self, board: np.ndarray) -> float:
        """Enhanced board quality evaluation with smaller scale"""
        holes = self._count_holes(board)
        bumpiness = self._calculate_bumpiness(board)
        max_height = self._get_max_height(board)
        lines_ready = self._lines_ready_to_clear(board)
        
        # MUCH smaller scale to prevent huge rewards
        quality = (
            -holes * 0.3 +          # Reduced from -3.0
            -bumpiness * 0.1 +      # Reduced from -1.0
            -max_height * 0.05 +    # Reduced from -0.5
            lines_ready * 1.0       # Reduced from 10.0
        )
        return quality
    
    def _count_holes(self, board: np.ndarray) -> int:
        """Count holes in board"""
        holes = 0
        rows, cols = board.shape
        for col in range(cols):
            block_found = False
            for row in range(rows):
                if board[row, col] == 1:
                    block_found = True
                elif block_found and board[row, col] == 0:
                    holes += 1
        return holes
    
    def _calculate_bumpiness(self, board: np.ndarray) -> int:
        """Calculate surface bumpiness"""
        heights = []
        rows, cols = board.shape
        for col in range(cols):
            height = 0
            for row in range(rows):
                if board[row, col] == 1:
                    height = rows - row
                    break
            heights.append(height)
        
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness
    
    def _get_max_height(self, board: np.ndarray) -> int:
        """Get maximum column height"""
        max_height = 0
        rows, cols = board.shape
        for col in range(cols):
            for row in range(rows):
                if board[row, col] == 1:
                    max_height = max(max_height, rows - row)
                    break
        return max_height
    
    def _lines_ready_to_clear(self, board: np.ndarray) -> int:
        """Count complete lines"""
        rows, cols = board.shape
        complete_lines = 0
        for row in range(rows):
            if np.sum(board[row, :]) == cols:
                complete_lines += 1
        return complete_lines
    
    def _has_good_bomb_targets(self, board: np.ndarray) -> bool:
        """Check if there are good bomb targets (only surface blocks)"""
        surface_blocks = self._get_surface_blocks(board)
        
        for block_row, block_col in surface_blocks:
            effectiveness = self._calculate_bomb_effectiveness_on_surface(board, block_row, block_col)
            if effectiveness >= 4:  # At least 4 blocks destroyed to be worth it
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
        """
        Calculate bomb effectiveness when bomb hits a surface block
        
        Explosion pattern: 3x3 area centered on hit block
        Relative offsets from hit point: (-1,-1) to (1,1)
        """
        rows, cols = board.shape
        
        # Define 3x3 explosion pattern around hit point
        explosion_offsets = [
            (-1, -1), (-1, 0), (-1, 1),  # Row above
            (0, -1),  (0, 0),  (0, 1),   # Same row as hit
            (1, -1),  (1, 0),  (1, 1)    # Row below
        ]
        
        # Count blocks that would be destroyed in explosion
        blocks_destroyed = 0
        destroyed_positions = []
        
        for row_offset, col_offset in explosion_offsets:
            explosion_row = hit_row + row_offset
            explosion_col = hit_col + col_offset
            
            # Check if position is valid (within board bounds)
            if 0 <= explosion_row < rows and 0 <= explosion_col < cols:
                if board[explosion_row, explosion_col] == 1:
                    blocks_destroyed += 1
                    destroyed_positions.append((explosion_row, explosion_col))
        
        if blocks_destroyed == 0:
            return 0
        
        # Simulate board after explosion to calculate improvements
        temp_board = board.copy()
        for destroy_row, destroy_col in destroyed_positions:
            temp_board[destroy_row, destroy_col] = 0
        
        # Calculate hole reduction (important benefit)
        old_holes = self._count_holes(board)
        new_holes = self._count_holes(temp_board)
        holes_reduced = old_holes - new_holes
        
        # Calculate bumpiness improvement
        old_bumpiness = self._calculate_bumpiness(board)
        new_bumpiness = self._calculate_bumpiness(temp_board)
        bumpiness_reduced = old_bumpiness - new_bumpiness
        
        # Weighted effectiveness score
        effectiveness = (
            blocks_destroyed * 1.0 +      # Base value for blocks destroyed
            holes_reduced * 2.0 +         # High bonus for hole reduction
            bumpiness_reduced * 0.5       # Bonus for surface smoothing
        )
        
        return effectiveness