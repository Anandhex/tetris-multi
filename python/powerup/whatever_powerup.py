from simplified_dqn_network import EnhancedDQNAgent
from powerup_dqn_network import PowerUpDQNAgent
from datetime import datetime
from tetris_client import UnityTetrisClient
from statistics import mean
from tqdm import tqdm
import logging
import numpy as np
import random
import json
import os
import time

class IntegratedTetrisTrainer:
    """Integrated trainer for both block placement and powerup decision making"""
    
    def __init__(self,
                 block_model_path: str = 'python/model_20250706-105237.h5',
                 powerup_model_path: str = None,
                 load_powerup_model: bool = False,
                 tensorboard_log_dir: str = None,
                 episodes: int = 3000):
        
        # Unity client for Tetris
        self.client = UnityTetrisClient()
        
        # Training hyperparameters
        self.episodes = episodes
        self.max_steps = None
        self.powerup_interval = 5  # Powerup appears every 5 blocks
        self.max_powerup_hold = 10  # Maximum blocks to hold powerup
        
        # Board parameters
        self.BOARD_HEIGHT = 20
        self.BOARD_WIDTH = 10
        
        # Logging and checkpoint
        self.start_at = f"integrated_model_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.best_score = -float('inf')
        self.total_steps = 0
        
        # Build tensorboard paths
        if tensorboard_log_dir is None:
            tensorboard_log_dir = f"logs/{self.start_at}"
        
        block_log_dir = f"{tensorboard_log_dir}/block_placement"
        powerup_log_dir = f"{tensorboard_log_dir}/powerup_decisions"
        
        # Initialize block placement agent (pre-trained)
        self.block_agent = EnhancedDQNAgent(
            n_neurons=[32, 32, 32],
            activations=['relu', 'relu', 'relu', 'linear'],
            epsilon=0,  # No exploration for trained block agent
            mem_size=1000,
            discount=0.95,
            tensorboard_log_dir=block_log_dir
        )
        
        # Load pre-trained block placement model
        if os.path.exists(block_model_path):
            self.block_agent.model.load_weights(block_model_path)
            print(f"Loaded block placement model from {block_model_path}")
        else:
            raise FileNotFoundError(f"Block placement model not found: {block_model_path}")
        
        # Initialize powerup DQN agent
        self.powerup_agent = PowerUpDQNAgent(
            state_size=8,  # [lines, holes, bumpiness, height, blocks_since_powerup, powerup_type_one_hot(3)]
            action_size=2,  # [use_now, wait]
            tensorboard_log_dir=powerup_log_dir
        )
        
        # Load powerup model if requested
        if load_powerup_model and powerup_model_path and os.path.exists(powerup_model_path):
            self.powerup_agent.load_model(powerup_model_path)
            print(f"Loaded powerup model from {powerup_model_path}")
        
        # PowerUp tracking
        self.powerup_types = ['bottom_line_clear', 'gravity', 'bomb']
        self.current_powerup = None
        self.blocks_since_powerup = 0
        self.powerup_rewards = []
        self.powerup_usage_stats = {pt: {'used': 0, 'total': 0} for pt in self.powerup_types}
        
        print(f"Integrated Tetris Trainer initialized")
        print(f"Block placement model: {block_model_path}")
        print(f"PowerUp model: {'New' if not load_powerup_model else powerup_model_path}")
    
    def ensure_connection(self, retries: int = 5, delay: float = 2.0):
        """Ensure connection to Unity client"""
        for _ in range(retries):
            state = self.client.wait_for_game_ready(timeout=5.0)
            if state is not None:
                return True
            time.sleep(delay)
        raise ConnectionError("Unable to connect to Unity Tetris client.")
    
    def get_board_features(self, board_state=None):
        """
        Extract board features from current game state
        Returns: [lines_cleared, holes, bumpiness, height]
        """
        if board_state is None:
            # Get current board state from Unity client
            board_state = self.client.get_board_state()
        
        # You'll need to implement this based on your Unity client's API
        # For now, using placeholder values
        lines = board_state.get('lines_cleared', 0)
        holes = board_state.get('holes', 0)
        bumpiness = board_state.get('bumpiness', 0)
        height = board_state.get('height', 0)
        
        return [lines, holes, bumpiness, height]
    
    def assign_random_powerup(self):
        """Assign a random powerup to the player"""
        return random.choice(self.powerup_types)
    
    def use_powerup(self, powerup_type, board_features_before):
        """
        Use the specified powerup and return the reward
        """
        # Send powerup command to Unity client
        result = self.client.use_powerup(powerup_type)
        
        if result.get('success', False):
            # Get board state after powerup
            board_features_after = self.get_board_features()
            
            # Calculate reward
            reward = self.powerup_agent.calculate_powerup_reward(
                board_features_before, board_features_after, powerup_type
            )
            
            # Update usage statistics
            self.powerup_usage_stats[powerup_type]['used'] += 1
            
            return reward, board_features_after
        else:
            # Powerup failed, small penalty
            return -10, board_features_before
    
    def train(self):
        """Main training loop"""
        scores = []
        
        # Connect to Unity Tetris client
        self.client.connect()
        
        print("Starting integrated training...")
        
        for episode in tqdm(range(self.episodes), desc="Training Episodes"):
            # Reset game
            self.client.env_reset()
            done = False
            steps = 0
            blocks_placed = 0
            episode_reward = 0.0
            episode_score = 0
            
            # PowerUp state
            self.current_powerup = None
            self.blocks_since_powerup = 0
            episode_powerup_rewards = []
            
            # Main game loop
            while not done and (self.max_steps is None or steps < self.max_steps):
                # Get possible block placements
                next_states = self.client.get_possible_states()
                
                if not next_states:
                    # No valid moves, game over
                    break
                
                # Use block placement agent to select best move
                action_map = {}
                for key, feats in next_states.items():
                    col_str, rot_str = key.split(":")
                    col_i, rot_i = int(col_str), int(rot_str)
                    action_map[tuple(feats)] = (col_i, rot_i)
                
                feature_list = list(action_map.keys())
                best_state = self.block_agent.best_state(feature_list, episode)
                col, rot = action_map[tuple(best_state)]
                
                # Get board features before move
                board_features_before = self.get_board_features()
                
                # Check if we should assign a powerup
                if blocks_placed > 0 and blocks_placed % self.powerup_interval == 0 and self.current_powerup is None:
                    self.current_powerup = self.assign_random_powerup()
                    self.blocks_since_powerup = 0
                    self.powerup_usage_stats[self.current_powerup]['total'] += 1
                    print(f"PowerUp assigned: {self.current_powerup}")
                
                # PowerUp decision making
                powerup_reward = 0
                if self.current_powerup is not None:
                    # Create state for powerup agent
                    powerup_state = self.powerup_agent.get_state(
                        board_features_before, 
                        self.blocks_since_powerup, 
                        self.current_powerup
                    )
                    
                    # Decide whether to use powerup
                    powerup_action = self.powerup_agent.act(powerup_state)
                    
                    # Force use if held too long
                    if self.blocks_since_powerup >= self.max_powerup_hold:
                        powerup_action = 0  # Use now
                    
                    if powerup_action == 0:  # Use powerup
                        powerup_reward, board_features_after_powerup = self.use_powerup(
                            self.current_powerup, board_features_before
                        )
                        
                        # Store experience for powerup agent
                        next_powerup_state = self.powerup_agent.get_state(
                            board_features_after_powerup, 0, 'none'
                        )
                        
                        self.powerup_agent.remember(
                            powerup_state, powerup_action, powerup_reward, 
                            next_powerup_state, True  # Episode ends for this powerup
                        )
                        
                        episode_powerup_rewards.append(powerup_reward)
                        board_features_before = board_features_after_powerup
                        
                        print(f"PowerUp used: {self.current_powerup}, Reward: {powerup_reward:.2f}")
                        self.current_powerup = None
                        self.blocks_since_powerup = 0
                    else:
                        # Store experience for waiting
                        self.powerup_agent.remember(
                            powerup_state, powerup_action, -1,  # Small penalty for waiting
                            powerup_state, False  # Continue episode
                        )
                
                # Execute block placement
                curr_meta = self.client.send_action_and_wait({"col": col, "rot": rot}, timeout=30.0)
                if curr_meta is None:
                    print(f"Episode {episode}: timeout, skipping step")
                    break
                
                done = curr_meta.get('gameOver', False)
                reward = curr_meta.get('reward', 0)
                
                # Update counters
                blocks_placed += 1
                if self.current_powerup is not None:
                    self.blocks_since_powerup += 1
                
                # Accumulate metrics
                episode_reward += reward + powerup_reward
                episode_score = curr_meta.get('score', 0)
                steps += 1
                self.total_steps += 1
            
            # Skip very short episodes
            if steps <= 6:
                continue
            
            # Train powerup agent
            if len(self.powerup_agent.memory) > self.powerup_agent.batch_size:
                for _ in range(3):  # Train multiple times per episode
                    self.powerup_agent.replay()
            
            # Record episode metrics
            scores.append(episode_reward)
            self.powerup_rewards.extend(episode_powerup_rewards)
            
            # Log metrics
            if episode % 50 == 0:
                # Log general metrics
                self.powerup_agent.writer.add_scalar('Episode/Total_Reward', episode_reward, episode)
                self.powerup_agent.writer.add_scalar('Episode/Game_Score', episode_score, episode)
                self.powerup_agent.writer.add_scalar('Episode/Steps', steps, episode)
                self.powerup_agent.writer.add_scalar('Episode/Blocks_Placed', blocks_placed, episode)
                
                # Log powerup metrics
                self.powerup_agent.log_episode_metrics(episode, episode_powerup_rewards, self.powerup_usage_stats)
                
                # Log running averages
                if len(scores) >= 50:
                    avg_score = mean(scores[-50:])
                    self.powerup_agent.writer.add_scalar('Stats/Avg_Score_50', avg_score, episode)
                
                if len(self.powerup_rewards) >= 50:
                    avg_powerup_reward = mean(self.powerup_rewards[-50:])
                    self.powerup_agent.writer.add_scalar('Stats/Avg_PowerUp_Reward_50', avg_powerup_reward, episode)
                
                print(f"Episode {episode}: Score={episode_score}, Reward={episode_reward:.2f}, "
                      f"PowerUp Rewards={sum(episode_powerup_rewards):.2f}")
            
            # Save best model
            if episode_reward > self.best_score:
                self.best_score = episode_reward
                self.powerup_agent.save_model(f"{self.start_at}_best_powerup.pth")
                print(f"New best model saved! Score: {episode_reward:.2f}")
            
            # Periodic checkpoint
            if episode % 100 == 0:
                self.powerup_agent.save_model(f"{self.start_at}_checkpoint.pth")
                
                # Save training stats
                with open(f'{self.start_at}_stats.json', 'w') as f:
                    json.dump({
                        'episode': episode,
                        'total_steps': self.total_steps,
                        'best_score': self.best_score,
                        'powerup_usage_stats': self.powerup_usage_stats,
                        'avg_score_last_100': mean(scores[-100:]) if len(scores) >= 100 else None
                    }, f, indent=2)
        
        # Final save
        self.powerup_agent.save_model(f"{self.start_at}_final.pth")
        
        # Cleanup
        self.powerup_agent.close()
        self.block_agent.close()
        self.client.disconnect()
        
        print("Training complete!")
        print(f"Best score: {self.best_score:.2f}")
        print(f"PowerUp usage stats: {self.powerup_usage_stats}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Initialize trainer
    trainer = IntegratedTetrisTrainer(
        block_model_path='model_20250706-105237.h5',
        load_powerup_model=False,  # Set to True to load existing powerup model
        episodes=3000
    )
    
    # Start training
    trainer.train()