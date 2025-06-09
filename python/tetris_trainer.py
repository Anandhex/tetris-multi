# improved_tetris_trainer.py
import sys
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import logging
import numpy as np
import os
from tetris_client import UnityTetrisClient
from dqn_agent import DQNAgent
import torch
from collections import Counter
from collections import deque

class TetrisTrainer:
    def __init__(self, agent_type='dqn', load_model=False, model_path='tetris_model.pth', 
                 tensorboard_log_dir=None, score_window_size=100):
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
        
        self.action_counter = Counter()
        
        # Training metrics
        self.episode_scores = []
        self.episode_lines = []
        self.episode_lengths = []
        self.episode_rewards = []
        self.max_score = 2000  

        # Performance-based curriculum tracking
        self.last_curriculum_change_episode = 0
        self.score_window = deque(maxlen=score_window_size)
        self.recent_scores = deque(maxlen=20)  # Track last 20 episodes for curriculum decisions
        self.consecutive_good_episodes = 0
        self.curriculum_advancement_threshold = 1000  # Score threshold for advancement
        self.consecutive_episodes_required = 10  # Consecutive episodes above threshold
        
        # Enhanced curriculum parameters
        self.current_curriculum_stage = 0
        self.curriculum_stages = [
            {
                'episodes': float('inf'),  # Performance-based progression
                'height': 20, 
                'preset': 1, 
                'pieces': 1, 
                'name': 'Very Easy',
                'advancement_threshold': 900,
                'consecutive_required': 8
            },
            {
                'episodes': float('inf'),
                'height': 20, 
                'preset': 2, 
                'pieces': 2, 
                'name': 'Easy',
                'advancement_threshold': 1100,
                'consecutive_required': 10
            },
            {
                'episodes': float('inf'),
                'height': 20, 
                'preset': 3, 
                'pieces': 3, 
                'name': 'Medium',
                'advancement_threshold': 1200,
                'consecutive_required': 12
            },
            {
                'episodes': float('inf'),
                'height': 20, 
                'preset': 0, 
                'pieces': 5, 
                'name': 'Hard',
                'advancement_threshold': 2000,
                'consecutive_required': 15
            },
            {
                'episodes': float('inf'),
                'height': 20, 
                'preset': 0, 
                'pieces': 7, 
                'name': 'Full Game',
                'advancement_threshold': float('inf'),  # No advancement from final stage
                'consecutive_required': float('inf')
            },
        ]
        
        # Reward structure parameters
        self.reward_config = {
            'survival_bonus': 0.1,
            'lines_multiplier': 75,  # Increased for better line clearing incentive
            'height_reward_threshold': 8,
            'height_reward_multiplier': 0.7,
            'height_penalty_threshold': 15,
            'height_penalty_multiplier': 4,
            'holes_creation_penalty': 15,
            'existing_holes_penalty': 0.2,
            'bumpiness_penalty': 0.4,
            'covered_holes_penalty': 2.0,
            'perfect_clear_bonus': 300,
            'death_penalty': 250,
            'score_multiplier': 0.02,
            # New rewards
            'efficiency_bonus_multiplier': 5,  # Reward for lines/pieces ratio
            'combo_bonus_base': 10,  # Bonus for consecutive line clears
            'placement_speed_bonus': 2,  # Bonus for quick decisions
        }
        
        # Setup logging
        self.setup_logging()
        
        # Best model tracking
        self.best_score = 0
        self.best_avg_score = 0
        
        # Performance tracking for curriculum
        self.last_piece_count = 0
        self.combo_count = 0
        
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
    
    def check_curriculum_advancement(self, episode, episode_score):
        """Check if agent should advance to next curriculum stage based on performance"""
        if self.current_curriculum_stage >= len(self.curriculum_stages) - 1:
            return False  # Already at final stage
        
        current_stage = self.curriculum_stages[self.current_curriculum_stage]
        threshold = current_stage['advancement_threshold']
        required_consecutive = current_stage['consecutive_required']
        
        # Add score to recent tracking
        self.recent_scores.append(episode_score)
        
        # Check if current episode meets threshold
        if episode_score >= threshold:
            self.consecutive_good_episodes += 1
        else:
            self.consecutive_good_episodes = 0
        
        # Check if we have enough consecutive good episodes
        should_advance = self.consecutive_good_episodes >= required_consecutive
        
        if should_advance:
            print(f"\n🎯 CURRICULUM ADVANCEMENT TRIGGERED!")
            print(f"   Episodes above {threshold}: {self.consecutive_good_episodes}/{required_consecutive}")
            print(f"   Recent scores: {list(self.recent_scores)[-5:]}")  # Show last 5
            return True
        
        # Log progress toward advancement
        if episode % 25 == 0 and self.consecutive_good_episodes > 0:
            print(f"📈 Curriculum progress: {self.consecutive_good_episodes}/{required_consecutive} "
                  f"episodes above {threshold} (current: {episode_score})")
        
        return False
    
    def apply_curriculum(self, episode, force_change=False):
        """Apply curriculum learning parameters with performance-based progression"""
        if not force_change and self.current_curriculum_stage < len(self.curriculum_stages) - 1:
            # Check if we should advance based on performance
            if len(self.recent_scores) > 0:
                latest_score = self.recent_scores[-1]
                if self.check_curriculum_advancement(episode, latest_score):
                    force_change = True
                    self.current_curriculum_stage += 1
                    self.consecutive_good_episodes = 0  # Reset counter
        
        if not force_change:
            return
        
        stage = self.curriculum_stages[self.current_curriculum_stage]
        
        print(f"\n{'='*70}")
        print(f"🚀 CURRICULUM CHANGE - Episode {episode}")
        if self.current_curriculum_stage > 0:
            prev_stage = self.curriculum_stages[self.current_curriculum_stage - 1]
            print(f"Previous Stage: {prev_stage['name']}")
        print(f"New Stage: {stage['name']}")
        print(f"Height: {stage['height']}, Preset: {stage['preset']}, Pieces: {stage['pieces']}")
        print(f"Next advancement at: {stage['advancement_threshold']} score for {stage['consecutive_required']} episodes")
        print(f"{'='*70}\n")
        
        # Reset exploration for new curriculum stage
      
        self.last_curriculum_change_episode = episode
        
        # Send curriculum change to Unity
        success = self.client.send_curriculum_change(
            board_height=stage['height'],
            board_preset=stage['preset'],
            tetromino_types=stage['pieces'],
            stage_name=stage['name']
        )
        
        if success:
            confirmation = self.client.get_curriculum_confirmation(timeout=5.0)
            if confirmation:
                actual_curriculum = self.client.get_curriculum_info(confirmation)
                print(f"✓ Curriculum confirmed: {actual_curriculum}")
                
                logging.info(f"Curriculum updated for episode {episode}: "
                        f"Stage={stage['name']}, Height={stage['height']}, "
                        f"Preset={stage['preset']}, Pieces={stage['pieces']}")
                
                self.agent.log_curriculum_change(episode, stage)
                self.agent.writer.add_text('Curriculum/Stage_Change', 
                                        f"Episode {episode}: Advanced to {stage['name']}", episode)
            else:
                print(f"⚠ Warning: Curriculum change not confirmed by Unity")
        else:
            print(f"✗ Error: Failed to send curriculum change to Unity")
    
    def calculate_reward(self, prev_state, current_state, action, step):
        """Enhanced reward function with better shaping and efficiency bonuses"""
        
        # Base reward from Unity
        base_reward = current_state.get('reward', 0.0)
        total_reward = base_reward
        
        # Update action counter
        self.action_counter[action] += 1
        
        # Calculate deltas
        lines_cleared = 0
        holes_created = 0
        score_gained = 0
        
        if prev_state is not None:
            lines_cleared = (current_state.get('linesCleared', 0) or 0) - (prev_state.get('linesCleared', 0) or 0)
            holes_created = (current_state.get('holesCount', 0) or 0) - (prev_state.get('holesCount', 0) or 0)
            score_gained = (current_state.get('score', 0) or 0) - (prev_state.get('score', 0) or 0)
        
        # Read current state metrics
        stack_height = current_state.get('stackHeight', 0) or 0
        holes_count = current_state.get('holesCount', 0) or 0
        bumpiness = current_state.get('bumpiness', 0) or 0
        covered_holes = current_state.get('covered', 0) or 0
        perfect_clear = current_state.get('perfectClear', False) or False
        pieces_placed = current_state.get('piecesPlaced', 0) or 0
        
        # Enhanced reward components
        cfg = self.reward_config
        
        # 1. Survival bonus
        survival_bonus = cfg['survival_bonus']
        
        # 2. Lines cleared reward (exponential scaling)
        lines_reward = 0.0
        if lines_cleared > 0:
            lines_reward = (lines_cleared ** 2) * cfg['lines_multiplier']
            
            # Combo bonus for consecutive line clears
            if hasattr(self, 'last_lines_cleared') and self.last_lines_cleared > 0:
                self.combo_count += 1
                combo_bonus = self.combo_count * cfg['combo_bonus_base']
                lines_reward += combo_bonus
            else:
                self.combo_count = 0
        else:
            self.combo_count = 0
        
        self.last_lines_cleared = lines_cleared
        
        # 3. Efficiency bonus (lines per piece ratio)
        efficiency_bonus = 0.0
        if pieces_placed > 0:
            total_lines = current_state.get('linesCleared', 0)
            efficiency_ratio = total_lines / pieces_placed
            if efficiency_ratio > 0.15:  # Above 15% efficiency
                efficiency_bonus = efficiency_ratio * cfg['efficiency_bonus_multiplier']
        
        # 4. Height management (refined)
        height_reward = 0.0
        if stack_height <= cfg['height_reward_threshold']:
            height_reward = (cfg['height_reward_threshold'] - stack_height) * cfg['height_reward_multiplier']
        elif stack_height >= cfg['height_penalty_threshold']:
            # Exponential penalty for dangerous heights
            danger_level = stack_height - cfg['height_penalty_threshold']
            height_reward = -(danger_level ** 1.5) * cfg['height_penalty_multiplier']
        
        # 5. Perfect clear bonus
        perfect_clear_bonus = cfg['perfect_clear_bonus'] if perfect_clear else 0
        
        # 6. Hole penalties (refined)
        holes_penalty = holes_created * cfg['holes_creation_penalty']
        existing_holes_penalty = holes_count * cfg['existing_holes_penalty']
        
        # 7. Board quality penalties
        bumpiness_penalty = bumpiness * cfg['bumpiness_penalty']
        covered_penalty = covered_holes * cfg['covered_holes_penalty']
        
        # 8. Score-based reward
        score_reward = score_gained * cfg['score_multiplier']
        
        # 9. Placement speed bonus (reward quick decisions)
        speed_bonus = cfg['placement_speed_bonus'] if step % 10 < 3 else 0
        
        # Combine all rewards
        shaped_reward = (
            survival_bonus +
            lines_reward +
            efficiency_bonus +
            score_reward +
            height_reward +
            perfect_clear_bonus +
            speed_bonus -
            holes_penalty -
            existing_holes_penalty -
            bumpiness_penalty -
            covered_penalty
        )
        
        total_reward += shaped_reward
        
        # Enhanced TensorBoard logging
        w = self.agent.writer
        
        # Log reward components
        w.add_scalar('Reward/Total', total_reward, step)
        w.add_scalar('Reward/Base', base_reward, step)
        w.add_scalar('Reward/Shaped', shaped_reward, step)
        w.add_scalar('Reward/Lines', lines_reward, step)
        w.add_scalar('Reward/Efficiency', efficiency_bonus, step)
        w.add_scalar('Reward/Height', height_reward, step)
        w.add_scalar('Reward/PerfectClear', perfect_clear_bonus, step)
        w.add_scalar('Reward/Speed', speed_bonus, step)
        w.add_scalar('Reward/Combo', self.combo_count, step)
        
        # Log penalties
        w.add_scalar('Penalty/Holes', holes_penalty, step)
        w.add_scalar('Penalty/ExistingHoles', existing_holes_penalty, step)
        w.add_scalar('Penalty/Bumpiness', bumpiness_penalty, step)
        w.add_scalar('Penalty/Covered', covered_penalty, step)
        
        # Log state metrics
        w.add_scalar('State/StackHeight', stack_height, step)
        w.add_scalar('State/HolesCount', holes_count, step)
        w.add_scalar('State/Bumpiness', bumpiness, step)
        w.add_scalar('State/CoveredHoles', covered_holes, step)
        w.add_scalar('State/LinesCleared', current_state.get('linesCleared', 0), step)
        w.add_scalar('State/Score', current_state.get('score', 0), step)
        w.add_scalar('State/PiecesPlaced', pieces_placed, step)
        
        # Performance metrics
        if pieces_placed > 0:
            w.add_scalar('Performance/EfficiencyRatio', 
                        current_state.get('linesCleared', 0) / pieces_placed, step)
        
        # Curriculum tracking
        w.add_scalar('Curriculum/CurrentStage', self.current_curriculum_stage, step)
        w.add_scalar('Curriculum/ConsecutiveGoodEpisodes', self.consecutive_good_episodes, step)
        
        # Action distribution histogram every 100 steps
        if step % 100 == 0:
            counts = [self.action_counter[i] for i in range(40)]
            w.add_histogram('Policy/ActionDistribution',
                          torch.tensor(counts, dtype=torch.float32), step)
        
        w.flush()
        return total_reward
    
    def train(self, episodes=sys.maxsize, save_interval=100, eval_interval=200):
        """Main training loop with enhanced curriculum management"""
        if not self.client.connect():
            print("Failed to connect to Unity. Make sure Unity is running!")
            return
        
        print(f"Starting training with performance-based curriculum...")
        print(f"Initial stage: {self.curriculum_stages[0]['name']}")
        logging.info(f"Training started: {episodes} episodes, agent: {self.agent_type}")
        # Log hyperparameters to TensorBoard
        hparams = {
            'lr': self.agent.learning_rate,
            'batch_size': self.agent.batch_size,
            'target_update_freq': self.agent.target_update_freq,
            'memory_size': self.agent.memory.maxlen,
        }
        self.agent.writer.add_hparams(hparams, {'hparam/placeholder': 0})
        print(f"Memory size: {len(self.agent.memory)}, Batch size: {self.agent.batch_size}")
        
        try:
            for episode in range(episodes):

                # Apply curriculum (will check for advancement automatically)
                if episode == 0:
                    self.apply_curriculum(episode, force_change=True)  # Initialize first stage
                else:
                    self.apply_curriculum(episode)
                
                # Reset environment and wait for game ready
                state = self.client.wait_for_game_ready(timeout=10.0)
                if state is None:
                    print(f"Episode {episode}: Failed to get initial state")
                    continue
                
                
                # Enhanced episode logging
                if episode % 1 == 0:
                    current_stage = self.curriculum_stages[self.current_curriculum_stage]
                    print(f"\nEpisode {episode} Status:")
                    print(f"  Stage: {current_stage['name']}")
                    print(f"  Progress: {self.consecutive_good_episodes}/{current_stage['consecutive_required']} "
                          f"above {current_stage['advancement_threshold']}")
                    if len(self.recent_scores) > 0:
                        print(f"  Recent avg score: {np.mean(list(self.recent_scores)[-5:]):.1f}")
                
                # Training loop continues as before...
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
                    reward = self.calculate_reward(prev_state, next_state, action, steps)
                    if done:
                        reward -= self.reward_config['death_penalty']
                    
                    episode_reward += reward
                    episode_score = next_state.get('score', 0)
                    episode_lines = next_state.get('linesCleared', 0)
                    
                    # Store experience
                    self.agent.remember(state, action, reward, next_state if not done else None, done)
                    
                    # Train agent
                    if len(self.agent.memory) > self.agent.batch_size:
                        try:
                            loss = self.agent.replay()
                        except Exception as e:
                            print(f"DEBUG: Error type: {type(e)}")
                            import traceback
                            traceback.print_exc()
                    
                    steps += 1
                    if done:
                        # Enhanced game over logging
                        board_metrics = self.client.get_board_metrics(next_state)
                        pieces_placed = next_state.get('piecesPlaced', 0)
                        efficiency = episode_lines / pieces_placed if pieces_placed > 0 else 0
                        
                        if episode % 1 == 0:
                            print(f"Episode {episode}: Score={episode_score}, "
                                f"Reward={episode_reward:.1f}, Lines={episode_lines}, "
                                f"Steps={steps}, Efficiency={efficiency:.3f}, "
                                f"Height={board_metrics.get('stack_height', 0)}")
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
                
                # Save model if new best score
                if episode_score > self.best_score:
                    self.best_score = episode_score
                    self.save_model(f"best_model_score_{episode_score}.pth")
                    print(f"🏆 New best score: {episode_score}!")
                
                # Regular saves and evaluation
                if episode % save_interval == 0 and episode > 0:
                    self.save_model()
                    self.save_metrics()
                
                if episode % eval_interval == 0 and episode > 0:
                    self.evaluate(episodes=5, episode_offset=episode)
        
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
            self.agent.close()
            self.client.disconnect()
            logging.info("Training session ended")
            print("Training completed!")
    
    def save_model(self, filename=None):
        """Save the trained model"""
        if filename is None:
            filename = self.model_path
        self.agent.save(filename)
    
    def save_metrics(self):
        """Save training metrics with enhanced data"""
        metrics_df = pd.DataFrame({
            'episode': range(len(self.episode_scores)),
            'score': self.episode_scores,
            'lines_cleared': self.episode_lines,
            'episode_length': self.episode_lengths,
            'episode_reward': self.episode_rewards,
        })
        
        # Add rolling averages
        metrics_df['score_ma_20'] = metrics_df['score'].rolling(window=20, min_periods=1).mean()
        metrics_df['reward_ma_20'] = metrics_df['episode_reward'].rolling(window=20, min_periods=1).mean()
        
        metrics_df.to_csv('training_metrics.csv', index=False)
        print("Enhanced training metrics saved to training_metrics.csv")
    
    def evaluate(self, episodes=10, episode_offset=0):
        """Evaluate the trained agent with current curriculum stage"""
        print(f"Evaluating agent for {episodes} episodes...")
        
        eval_scores = []
        eval_lines = []
        eval_rewards = []
        
        
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
                reward = self.calculate_reward(prev_state, next_state, action, steps)
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
        
        
        # Enhanced evaluation logging
        if eval_scores:
            avg_score = np.mean(eval_scores)
            avg_lines = np.mean(eval_lines)
            avg_reward = np.mean(eval_rewards)
            max_score = max(eval_scores)
            std_score = np.std(eval_scores)
            
            eval_episode = episode_offset // 200
            self.agent.writer.add_scalar('Evaluation/Avg_Score', avg_score, eval_episode)
            self.agent.writer.add_scalar('Evaluation/Avg_Lines', avg_lines, eval_episode)
            self.agent.writer.add_scalar('Evaluation/Avg_Reward', avg_reward, eval_episode)
            self.agent.writer.add_scalar('Evaluation/Max_Score', max_score, eval_episode)
            self.agent.writer.add_scalar('Evaluation/Score_Std', std_score, eval_episode)
            
            current_stage = self.curriculum_stages[self.current_curriculum_stage]
            print(f"\n📊 Evaluation Results (Stage: {current_stage['name']}):")
            print(f"Average Score: {avg_score:.2f} ± {std_score:.2f}")
            print(f"Average Lines: {avg_lines:.2f}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Best Score: {max_score}")
            print(f"Score Range: {min(eval_scores)} - {max_score}")
            
            # Check if evaluation performance suggests readiness for advancement
            if avg_score > current_stage['advancement_threshold']:
                print(f"🎯 Evaluation suggests readiness for curriculum advancement!")
        
        return eval_scores, eval_lines, eval_rewards