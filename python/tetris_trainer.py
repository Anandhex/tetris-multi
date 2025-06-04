# tetris_trainer.py
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import logging
import numpy as np
import os
from tetris_client import UnityTetrisClient
from dqn_agent import DQNAgent

class TetrisTrainer:
    def __init__(self, agent_type='dqn', load_model=False, model_path='tetris_model.pth', 
                 tensorboard_log_dir=None):
        self.client = UnityTetrisClient()
        self.agent_type = agent_type
        self.model_path = model_path
        
        # Create tensorboard log directory
        if tensorboard_log_dir is None:
            tensorboard_log_dir = f"runs/tetris_{agent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize agent
        if agent_type == 'dqn':
            self.agent = DQNAgent(state_size=208, tensorboard_log_dir=tensorboard_log_dir)
            if load_model:
                self.agent.load(model_path)
        
        # Training metrics
        self.episode_scores = []
        self.episode_lines = []
        self.episode_lengths = []
        self.episode_rewards = []
        
        # Curriculum parameters
        self.current_curriculum_stage = 0
        self.curriculum_stages = [
            {'episodes': 100, 'height': 8, 'preset': 1, 'pieces': 1, 'name': 'Very Easy'},
            {'episodes': 200, 'height': 10, 'preset': 2, 'pieces': 2, 'name': 'Easy'},
            {'episodes': 300, 'height': 15, 'preset': 3, 'pieces': 5, 'name': 'Medium'},
            {'episodes': 500, 'height': 20, 'preset': 4, 'pieces': 7, 'name': 'Hard'},
            {'episodes': float('inf'), 'height': 20, 'preset': 0, 'pieces': 7, 'name': 'Full Game'},
        ]
        
        # Setup logging
        self.setup_logging()
        
        # Best model tracking
        self.best_score = 0
        self.best_avg_score = 0
        
    def setup_logging(self):
        """Setup logging for training progress"""
        log_filename = f"tetris_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Also log to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(console_handler)
        
        logging.info("Training session started")
    
    def get_curriculum_stage(self, episode):
        """Get curriculum parameters for current episode"""
        total_episodes = 0
        for i, stage in enumerate(self.curriculum_stages):
            total_episodes += stage['episodes']
            if episode < total_episodes:
                return i, stage
        
        # Return final stage if beyond curriculum
        return len(self.curriculum_stages) - 1, self.curriculum_stages[-1]
    
    def apply_curriculum(self, episode):
        """Apply curriculum learning parameters"""
        stage_idx, stage = self.get_curriculum_stage(episode)
        if stage_idx != self.current_curriculum_stage:
            print(f"\n{'='*60}")
            print(f"CURRICULUM CHANGE - Episode {episode}")
            print(f"Previous Stage: {self.curriculum_stages[self.current_curriculum_stage]['name'] if self.current_curriculum_stage < len(self.curriculum_stages) else 'Unknown'}")
            print(f"New Stage: {stage['name']}")
            print(f"Height: {stage['height']}, Preset: {stage['preset']}, Pieces: {stage['pieces']}")
            print(f"{'='*60}\n")
            
            # Send curriculum change with stage name
            success = self.client.send_curriculum_change(
                board_height=stage['height'],
                board_preset=stage['preset'],
                tetromino_types=stage['pieces'],
                stage_name=stage['name']
            )
            
            if success:
                # Wait for confirmation
                confirmation = self.client.get_curriculum_confirmation(timeout=5.0)
                if confirmation:
                    actual_curriculum = self.client.get_curriculum_info(confirmation)
                    print(f"✓ Curriculum confirmed: {actual_curriculum}")
                    
                    logging.info(f"Curriculum updated for episode {episode}: "
                            f"Stage={stage['name']}, Height={stage['height']}, "
                            f"Preset={stage['preset']}, Pieces={stage['pieces']}")
                    
                    # Log to TensorBoard
                    self.agent.log_curriculum_change(episode, stage)
                    self.agent.writer.add_text('Curriculum/Stage_Change', 
                                            f"Episode {episode}: Changed to {stage['name']}", episode)
                    
                    self.current_curriculum_stage = stage_idx
                else:
                    print(f"⚠ Warning: Curriculum change not confirmed by Unity")
            else:
                print(f"✗ Error: Failed to send curriculum change to Unity")
    
    def calculate_reward(self, prev_state, current_state, action):
        """Calculate custom reward function"""
        reward = current_state.get('reward', 0)
        
        # Additional reward shaping
        if prev_state is not None:
            # Reward for clearing lines (exponential)
            lines_cleared = current_state.get('linesCleared', 0) - prev_state.get('linesCleared', 0)
            if lines_cleared > 0:
                reward += lines_cleared ** 2 * 10
            
            # Penalty for creating holes
            holes_created = current_state.get('holesCount', 0) - prev_state.get('holesCount', 0)
            reward -= holes_created * 5
            
            # Reward for keeping stack low
            stack_height = current_state.get('stackHeight', 0)
            if stack_height < 10:
                reward += (10 - stack_height) * 0.5
            elif stack_height > 15:
                reward -= (stack_height - 15) * 2
            
            # Bonus for perfect clear
            if current_state.get('perfectClear', False):
                reward += 100
        
        return reward
    
    def train(self, episodes=2000, save_interval=100, eval_interval=200):
        """Main training loop"""
        if not self.client.connect():
            print("Failed to connect to Unity. Make sure Unity is running!")
            return
        
        print(f"Starting training for {episodes} episodes...")
        logging.info(f"Training started: {episodes} episodes, agent: {self.agent_type}")
    
        # Log hyperparameters to TensorBoard
        hparams = {
            'lr': self.agent.learning_rate,
            'batch_size': self.agent.batch_size,
            'epsilon_decay': self.agent.epsilon_decay,
            'target_update_freq': self.agent.target_update_freq,
            'memory_size': self.agent.memory.maxlen,
        }
        self.agent.writer.add_hparams(hparams, {'hparam/placeholder': 0})
        
        try:
            for episode in range(episodes):
                # Apply curriculum
                self.apply_curriculum(episode)
                
                # Reset environment
                self.client.send_reset()
                
                # Wait for game to be ready
                state = self.client.wait_for_game_ready(timeout=10.0)
                if state is None:
                    print(f"Episode {episode}: Failed to get initial state")
                    continue
                
                # Verify curriculum is applied
                curriculum_info = self.client.get_curriculum_info(state)
                stage_idx, expected_stage = self.get_curriculum_stage(episode)
                
                # Log curriculum verification
                if episode % 50 == 0:
                    print(f"Episode {episode} curriculum verification:")
                    print(f"  Expected: Height={expected_stage['height']}, Preset={expected_stage['preset']}, Pieces={expected_stage['pieces']}")
                    print(f"  Actual:   Height={curriculum_info['board_height']}, Preset={curriculum_info['board_preset']}, Pieces={curriculum_info['allowed_tetromino_types']}")
                
                # Continue with existing training loop...
                episode_reward = 0
                episode_score = 0
                episode_lines = 0
                steps = 0
                prev_state = None
                
                while True:
                    # Choose action
                    action = self.agent.act(state, training=True)
                    
                    # Send action and wait for result
                    next_state = self.client.send_action_and_wait(action, timeout=10.0)
                    
                    if next_state is None:
                        print(f"Episode {episode}: Timeout waiting for next state")
                        break
                    
                    # Check if game is over
                    done = self.client.is_game_over(next_state)
                    
                    # Calculate custom reward
                    reward = self.calculate_reward(prev_state, next_state, action)
                    
                    episode_reward += reward
                    episode_score = next_state.get('score', 0)
                    episode_lines = next_state.get('linesCleared', 0)
                    
                    # Store experience
                    self.agent.remember(state, action, reward, next_state if not done else None, done)
                    
                    # Train agent
                    if len(self.agent.memory) > self.agent.batch_size:
                        try:
                            loss = self.agent.replay()
                        except RuntimeError as e:
                            print(f"Training error at episode {episode}, step {steps}: {e}")
                            # Continue without this training step
                            pass
                    
                    steps += 1
                    
                    if done:
                        # Log game over details
                        board_metrics = self.client.get_board_metrics(next_state)
                        if episode % 10 == 0:
                            print(f"Episode {episode} ended: Score={episode_score}, "
                                f"Lines={episode_lines}, Steps={steps}, "
                                f"Final height={board_metrics['stack_height']}, "
                                f"Holes={board_metrics['holes_count']}")
                        break
                    
                    # Check if ready for next action
                    action_info = self.client.get_action_space_info(next_state)
                    if not action_info['waiting_for_action']:
                        # Wait for next action opportunity
                        next_state = self.client.wait_for_game_ready(timeout=5.0)
                        if next_state is None:
                            break
                    
                    prev_state = state
                    state = next_state
                
                # Record episode metrics
                self.episode_scores.append(episode_score)
                self.episode_lines.append(episode_lines)
                self.episode_lengths.append(steps)
                self.episode_rewards.append(episode_reward)
                
                # Game metrics for TensorBoard
                game_metrics = self.client.get_board_metrics(next_state) if next_state else {}
                
                # Log to TensorBoard
                self.agent.log_episode_metrics(episode, episode_reward, steps, episode_score, 
                                            episode_lines, game_metrics)
                
                # ... rest of your existing training loop code ...
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            logging.info("Training interrupted by user")
        
        except Exception as e:
            print(f"Training error: {e}")
            logging.error(f"Training error: {e}")
        
        finally:
            # Save final model and metrics
            self.save_model()
            self.save_metrics()
            self.agent.close()  # Close TensorBoard writer
            self.client.disconnect()
            logging.info("Training session ended")
            print("Training completed!") 
    def save_model(self):
        """Save the trained model"""
        self.agent.save(self.model_path)
    
    def save_metrics(self):
        """Save training metrics"""
        metrics_df = pd.DataFrame({
            'episode': range(len(self.episode_scores)),
            'score': self.episode_scores,
            'lines_cleared': self.episode_lines,
            'episode_length': self.episode_lengths,
            'episode_reward': self.episode_rewards,
        })
        
        metrics_df.to_csv('training_metrics.csv', index=False)
        print("Training metrics saved to training_metrics.csv")
    
    def evaluate(self, episodes=10, episode_offset=0):
        """Evaluate the trained agent"""
        print(f"Evaluating agent for {episodes} episodes...")
        
        eval_scores = []
        eval_lines = []
        eval_rewards = []
        
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0  # No exploration during evaluation
        
        for eval_ep in range(episodes):
            self.client.send_reset()
            
            # Wait for initial state
            state = None
            for _ in range(50):
                state = self.client.get_game_state(timeout=0.2)
                if state and state.get('waitingForAction', False):
                    break
            
            if state is None:
                continue
            
            episode_score = 0
            episode_lines = 0
            episode_reward = 0
            prev_state = None
            
            while True:
                # Choose best action (no exploration)
                action = self.agent.act(state, training=False)
                
                if not self.client.send_action(action):
                    break
                
                # Wait for result
                next_state = None
                for _ in range(100):
                    next_state = self.client.get_game_state(timeout=0.1)
                    if next_state:
                        break
                
                if next_state is None:
                    break
                
                # Calculate reward
                reward = self.calculate_reward(prev_state, next_state, action)
                episode_reward += reward
                
                done = next_state.get('gameOver', False)
                episode_score = next_state.get('score', 0)
                episode_lines = next_state.get('linesCleared', 0)
                
                if done:
                    break
                
                if not next_state.get('waitingForAction', False):
                    continue
                
                prev_state = state
                state = next_state
            
            eval_scores.append(episode_score)
            eval_lines.append(episode_lines)
            eval_rewards.append(episode_reward)
            
            print(f"Eval Episode {eval_ep}: Score={episode_score}, Lines={episode_lines}, Reward={episode_reward:.2f}")
        
        # Restore original epsilon
        self.agent.epsilon = original_epsilon
        
        # Log evaluation results to TensorBoard
        if eval_scores:
            avg_score = np.mean(eval_scores)
            avg_lines = np.mean(eval_lines)
            avg_reward = np.mean(eval_rewards)
            max_score = max(eval_scores)
            
            eval_episode = episode_offset // 200  # Evaluation episode counter
            self.agent.writer.add_scalar('Evaluation/Avg_Score', avg_score, eval_episode)
            self.agent.writer.add_scalar('Evaluation/Avg_Lines', avg_lines, eval_episode)
            self.agent.writer.add_scalar('Evaluation/Avg_Reward', avg_reward, eval_episode)
            self.agent.writer.add_scalar('Evaluation/Max_Score', max_score, eval_episode)
            
            print(f"\nEvaluation Results:")
            print(f"Average Score: {avg_score:.2f}")
            print(f"Average Lines: {avg_lines:.2f}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Best Score: {max_score}")
            print(f"Score Range: {min(eval_scores)} - {max_score}")
        
        return eval_scores, eval_lines, eval_rewards