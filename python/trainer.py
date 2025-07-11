# trainer.py - Changes for PyTorch compatibility
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np

class PowerupTrainer:
    """Main training class for powerup DQN"""
    
    def __init__(self, dataset_path: str, save_dir: str = "models"):
        from environments import TrainingEnvironment
        from powerup_dqn_agent import PowerupDQNAgent
        
        self.environment = TrainingEnvironment(dataset_path)
        self.agent = PowerupDQNAgent()
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []  # Now stores PyTorch loss values
        
        # Print model info
        self.agent.print_model_info()
    
    def train(self, episodes: int = 1000, target_update_freq: int = 100,
              save_freq: int = 100, render_freq: int = 50):
        """Main training loop"""
        
        print(f"Starting training for {episodes} episodes...")
        
        for episode in range(episodes):
            # Reset environment
            self.environment.reset()
            
            episode_reward = 0
            episode_length = 0
            episode_loss = 0
            
            # Run episode (single step for now, can extend to multi-step)
            for step in range(10):  # Max 10 powerup decisions per episode
                # Get current state
                current_features = self.environment.get_features()
                
                # Choose action
                action = self.agent.choose_action(self.environment)
                action_id = self.agent.action_space.encode_action(action['type'])
                
                # Apply action
                new_board, reward = self.environment.apply_powerup(action)
                
                # Get next state
                next_features = self.environment.get_features()
                
                # Check if any powerups are left
                powerups_left = any(self.environment.get_powerup_availability().values())
                done = not powerups_left
                
                # Store experience
                self.agent.remember(current_features, action_id, reward, next_features, done)
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # Train the agent (returns loss value)
            loss = self.agent.train()
            episode_loss = loss if loss > 0 else 0
            
            # Hard update target network (less frequent than soft updates)
            if episode % target_update_freq == 0:
                self.agent.hard_update()
                print(f"Episode {episode}: Hard update of target network")
            
            # Record metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.losses.append(episode_loss)
            
            # Logging
            if episode % render_freq == 0:
                avg_reward = np.mean(self.episode_rewards[-render_freq:])
                avg_loss = np.mean(self.losses[-render_freq:]) if self.losses[-render_freq:] else 0
                print(f"Episode {episode}:")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Avg Loss: {avg_loss:.4f}")
                print(f"  Epsilon: {self.agent.epsilon:.3f}")
                print(f"  Memory: {len(self.agent.memory)}")
                
                # Check CUDA memory if available
                from powerup_dqn_agent import check_cuda_memory
                check_cuda_memory()
            
            # Save model
            if episode % save_freq == 0 and episode > 0:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_path = os.path.join(self.save_dir, f"powerup_model_ep{episode}_{timestamp}.pth")
                self.agent.save_model(model_path)
        
        # Final save
        final_model_path = os.path.join(self.save_dir, "powerup_model_final.pth")
        self.agent.save_model(final_model_path)
        
        self.plot_training_metrics()
        
        return final_model_path
    
    def plot_training_metrics(self):
        """Plot training progress"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        
        # Episode rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        
        # Moving average of rewards
        window_size = 50
        if len(self.episode_rewards) >= window_size:
            moving_avg = []
            for i in range(window_size, len(self.episode_rewards)):
                moving_avg.append(np.mean(self.episode_rewards[i-window_size:i]))
            ax2.plot(range(window_size, len(self.episode_rewards)), moving_avg)
            ax2.set_title(f'Moving Average Reward (window={window_size})')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Average Reward')
            ax2.grid(True)
        
        # Training loss
        if self.losses:
            # Filter out zero losses (when no training occurred)
            non_zero_losses = [loss for loss in self.losses if loss > 0]
            if non_zero_losses:
                ax3.plot(non_zero_losses)
                ax3.set_title('Training Loss')
                ax3.set_xlabel('Training Step')
                ax3.set_ylabel('MSE Loss')
                ax3.set_yscale('log')  # Log scale for loss
                ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_metrics.png'), dpi=300)
        plt.show()