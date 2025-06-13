# improved_tetris_trainer.py
import sys
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import logging
import numpy as np
import os
from tetris_client import UnityTetrisClient
from simplified_dqn_network import EnhancedDQNAgent
import torch
from collections import Counter
from collections import deque

class TetrisTrainer:
    def __init__(self, agent_type='dqn', load_model=False, model_path='tetris_model.pth', 
                 tensorboard_log_dir=None, score_window_size=100):
        self.client = UnityTetrisClient()
        self.agent_type = agent_type
        self.model_path = model_path
        self.total_steps = 0 
        
        # Create tensorboard log directory
        if tensorboard_log_dir is None:
            tensorboard_log_dir = f"runs/tetris_{agent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize agent
        if agent_type == 'dqn':
            self.agent = EnhancedDQNAgent(tensorboard_log_dir=tensorboard_log_dir)
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
            # {
            #     'episodes': float('inf'),  # Performance-based progression
            #     'height': 20, 
            #     'preset': 1, 
            #     'pieces': 1, 
            #     'name': 'Very Easy',
            #     'advancement_threshold': 900,
            #     'consecutive_required': 8
            # },
            # {
            #     'episodes': float('inf'),
            #     'height': 20, 
            #     'preset': 2, 
            #     'pieces': 2, 
            #     'name': 'Easy',
            #     'advancement_threshold': 1100,
            #     'consecutive_required': 10
            # },
            # {
            #     'episodes': float('inf'),
            #     'height': 20, 
            #     'preset': 3, 
            #     'pieces': 3, 
            #     'name': 'Medium',
            #     'advancement_threshold': 1200,
            #     'consecutive_required': 12
            # },
            # {
            #     'episodes': float('inf'),
            #     'height': 20, 
            #     'preset': 0, 
            #     'pieces': 5, 
            #     'name': 'Hard',
            #     'advancement_threshold': 2000,
            #     'consecutive_required': 15
            # },
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
            'survival_bonus': 0.1,                      # flat survival incentive
            'lines_multiplier': 1.0,                    # Tetris score multiplier
            'combo_bonus_base': 5,                      # combo bonus per streak
            'efficiency_threshold': 0.15,               # lines/piece threshold
            'efficiency_bonus_multiplier': 3.0,         # bonus for efficient placement
            'height_reward_threshold': 10,              # reward for staying below this height
            'height_reward_multiplier': 0.5,            # reward per row under threshold
            'height_penalty_threshold': 16,             # stack height danger zone
            'height_penalty_multiplier': 3.0,           # exponential penalty above danger
            'holes_creation_penalty': 10,               # penalty for creating holes
            'existing_holes_penalty': 0.5,              # penalty for existing holes
            'bumpiness_penalty': 0.3,                   # penalty for surface unevenness
            'well_penalty': 0.2,                        # quadratic penalty for wells
            'skyscraper_height_threshold': 4,           # when a column is “too tall”
            'skyscraper_penalty': 0.1,                  # quadratic penalty for spikes
            'perfect_clear_bonus': 200,                 # bonus for perfect clears
            'death_penalty': 100    
        }
        
        # Setup logging
        self.setup_logging()
        
        # Best model tracking
        self.best_score = 0
        self.best_avg_score = 0
        
        # Performance tracking for curriculum
        self.last_piece_count = 0
        self.combo_count = 0
        self.BOARD_HEIGHT = 20
        self.BOARD_WIDTH = 10

    def _reshape_board(self, flat_board):
        return np.array(flat_board, dtype=np.uint8).reshape((self.BOARD_HEIGHT, self.BOARD_WIDTH))

    def _column_heights(self, board):
        heights = []
        for x in range(self.BOARD_WIDTH):
            col = board[:, x]
            filled = np.where(col == 1)[0]
            heights.append(self.BOARD_HEIGHT - filled.min() if filled.size else 0)
        return np.array(heights)

    def _count_holes(self, board, heights):
        holes = 0
        for x, h in enumerate(heights):
            if h > 0:
                col = board[-h:, x]
                holes += (col == 0).sum()
        return int(holes)

    def _bumpiness(self, heights):
        return int(np.abs(np.diff(heights)).sum())

    def _wells_depth(self, heights):
        total = 0
        for i, h in enumerate(heights):
            left = heights[i - 1] if i > 0 else self.BOARD_HEIGHT
            right = heights[i + 1] if i < self.BOARD_WIDTH - 1 else self.BOARD_HEIGHT
            d = min(left, right) - h
            if d > 0:
                total += d
        return int(total)

    def extract_features(self, flat_board):
        try:
            if (flat_board is None
                    or not isinstance(flat_board, list)
                    or len(flat_board) != 200
                    or any(v is None for v in flat_board)):
                raise ValueError("Malformed board")

            board = self._reshape_board(flat_board)
            heights = self._column_heights(board)

            max_height = int(np.max(heights)) if heights.size > 0 else 0

            return {
                'column_heights': heights,
                'holes': self._count_holes(board, heights),
                'bumpiness': self._bumpiness(heights),
                'wells': self._wells_depth(heights),
                'stack_height': max_height,
            }

        except Exception as e:
            print(f"[ERROR] Feature extraction failed: {e}")
            return {
                'column_heights': np.zeros(10, dtype=int),
                'holes': 0,
                'bumpiness': 0,
                'wells': 0,
                'stack_height': 0,
            }


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
        """Simple curriculum advancement based on score threshold"""
        if self.current_curriculum_stage >= len(self.curriculum_stages) - 1:
            return False  # Already at final stage
        
        # Use the advancement threshold from current curriculum stage
        current_stage = self.curriculum_stages[self.current_curriculum_stage]
        threshold = current_stage['advancement_threshold']
        
        # If we consistently hit the score threshold, advance
        if episode_score >= threshold:
            self.consecutive_good_episodes += 1
        else:
            self.consecutive_good_episodes = 0
        
        # Advance after 3 consecutive episodes hitting the threshold
        if self.consecutive_good_episodes >= current_stage['consecutive_required']:
            print(f"\n🎯 ADVANCING CURRICULUM! Episode {episode}, Score: {episode_score} >= {threshold}")
            return True
        
        return False
    
    def apply_curriculum_with_reconnect(self, episode, episode_score, force_change=False):
        """Apply curriculum change with connection recovery"""
        # Skip curriculum check on first episode (no previous score)
        if episode == 0:
            if force_change:
                # Initialize first stage
                pass
            else:
                return True
        else:
            # Check advancement using previous episode's score
            if not force_change:
                if self.check_curriculum_advancement(episode, episode_score):
                    self.current_curriculum_stage += 1
                    self.consecutive_good_episodes = 0
                    force_change = True
        
        if not force_change:
            return True  # No change needed, connection is fine
        
        stage = self.curriculum_stages[self.current_curriculum_stage]
        print(f"🚀 New Stage: {stage['name']} - Height: {stage['height']}, Pieces: {stage['pieces']}")
        
        # Try curriculum change with connection recovery
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Send curriculum change
                success = self.client.send_curriculum_change(
                    board_height=stage['height'],
                    board_preset=stage['preset'],
                    tetromino_types=stage['pieces'],
                    stage_name=stage['name']
                )
                
                if success:
                    # Wait longer for Unity to process and stabilize
                    import time
                    time.sleep(3.0)  # Increased wait time
                    
                    # Send reset to ensure clean state
                    self.client.send_reset()
                    time.sleep(1.0)
                    
                    # Test connection by waiting for game ready
                    test_state = self.client.wait_for_game_ready(timeout=10.0)
                    if test_state is not None:
                        print(f"✅ Curriculum change successful on attempt {attempt + 1}")
                        return True
                
                print(f"⚠️ Curriculum change failed, attempt {attempt + 1}/{max_retries}")
                
            except Exception as e:
                print(f"❌ Connection error during curriculum change: {e}")
            
            # Reconnect if not the last attempt
            if attempt < max_retries - 1:
                print("🔄 Reconnecting to Unity...")
                self.client.disconnect()
                import time
                time.sleep(3.0)  # Give Unity more time to reset
                
                # Try to reconnect with retries
                reconnect_success = False
                for reconnect_attempt in range(3):
                    if self.client.connect():
                        reconnect_success = True
                        break
                    time.sleep(2.0)
                
                if not reconnect_success:
                    print(f"❌ Reconnection failed on attempt {attempt + 1}")
                    continue
                print("✅ Reconnected successfully")
        
        print("❌ Failed to apply curriculum change after all retries")
        return False

    def ensure_connection_ready(self):
        """Ensure connection is ready before starting episode"""
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                # Try to get initial state
                state = self.client.wait_for_game_ready(timeout=5.0)
                if state is not None:
                    return state
                
                print(f"⚠️ No initial state, attempt {attempt + 1}/{max_attempts}")
                
                # Try reset
                self.client.send_reset()
                import time
                time.sleep(2.0)
                
            except Exception as e:
                print(f"❌ Connection issue: {e}")
                
                # Reconnect if not last attempt
                if attempt < max_attempts - 1:
                    self.client.disconnect()
                    import time
                    time.sleep(2.0)
                    if not self.client.connect():
                        continue
        
        return None
    
    def calculate_reward(self, prev_state, current_state, action, step):
        # 1) Validate board
        board = current_state.get('board')
        if not isinstance(board, list) or len(board) != self.BOARD_HEIGHT * self.BOARD_WIDTH:
            self.agent.writer.add_scalar('reward/invalid_board', 1, step)
            return -0.5

        f_curr = self.extract_features(board)
        print(f_curr)
        lines_prev = prev_state.get('linesCleared', 0) if prev_state else 0
        lines = max(0, current_state.get('linesCleared', 0) - lines_prev)

        # Heuristic component
        heuristic = (
            -0.51 * f_curr['stack_height']
            + 5.0 * (lines ** 2)
            - 0.36 * f_curr['holes']
            - 0.18 * f_curr['bumpiness']
        )

        # Score‐based component with a single death penalty
        score_comp = lines ** 2
        if current_state.get('gameOver', False):
            score_comp -= 20

        # Shaping terms (tuned softly)
        shaped = (
            0.2
            + lines * self.reward_config['lines_multiplier']
            - f_curr['holes'] * 0.25
            - f_curr['bumpiness'] * 0.1
            - f_curr['wells'] * 0.1
        )
        h = f_curr['stack_height']
        if h < self.reward_config['height_reward_threshold']:
            shaped += (self.reward_config['height_reward_threshold'] - h) * self.reward_config['height_reward_multiplier']
        if h > self.reward_config['height_penalty_threshold']:
            shaped -= ((h - self.reward_config['height_penalty_threshold']) ** 2) * self.reward_config['height_penalty_multiplier']

        # Blend heuristic → score
        alpha = min(1.0, step / 10_000)
        raw_reward = (1 - alpha) * heuristic + alpha * score_comp + shaped

        # --- FORCE normalization & clipping ---
        divided = raw_reward / 50.0
        clipped = np.clip(divided, -1.0, 1.0)
        reward = float(clipped)

        # Log each step so you can see the numbers
        w = self.agent.writer
        w.add_scalar('debug/raw_reward', raw_reward, step)
        w.add_scalar('debug/divided_reward', divided, step)
        w.add_scalar('debug/clipped_reward', reward, step)

        # And the normal logs
        w.add_scalar('reward/total',       reward,        step)
        w.add_scalar('reward/heuristic',   heuristic,     step)
        w.add_scalar('reward/score',       score_comp,    step)
        w.add_scalar('reward/shaped',      shaped,        step)
        w.add_scalar('reward/blend_alpha', alpha,         step)

        return reward

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
        prev_episode_score = 0
        
        try:
            for episode in range(episodes):

                if episode == 0:
                    # Initialize first curriculum stage
                    connection_ok = self.apply_curriculum_with_reconnect(episode, 0, force_change=True)
                else:
                    # Use previous episode's score for curriculum decision
                    connection_ok = self.apply_curriculum_with_reconnect(episode, prev_episode_score)
                
                if not connection_ok:
                    print("⚠️ Curriculum change failed, but continuing...")
                
                # Enhanced episode logging
                if episode % 1 == 0:
                    current_stage = self.curriculum_stages[self.current_curriculum_stage]
                    print(f"\nEpisode {episode} Status:")
                    print(f"  Stage: {current_stage['name']}")
                    print(f"  Progress: {self.consecutive_good_episodes}/{current_stage['consecutive_required']} "
                          f"above {current_stage['advancement_threshold']}")
                    if len(self.recent_scores) > 0:
                        print(f"  Recent avg score: {np.mean(list(self.recent_scores)[-5:]):.1f}")
                
                # GET INITIAL STATE - This was missing in the original code!
                state = self.ensure_connection_ready()
                if state is None:
                    print(f"Episode {episode}: Failed to get initial game state, skipping episode")
                    continue
                
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
                    self.total_steps += 1
                    reward = self.calculate_reward(prev_state, next_state, action, self.total_steps)
                    
                    
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
                
                # if episode % eval_interval == 0 and episode > 0:
                #     self.evaluate(episodes=5, episode_offset=episode)

                prev_episode_score = episode_score
                    
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            logging.info("Training interrupted by user")
        
        except Exception as e:
            print(f"Training error: {e}")
            logging.error(f"Training error: {e}")
            import traceback
            traceback.print_exc()
        
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
                try:
                    reward = self.calculate_reward(prev_state, next_state, action, self.total_steps)
                    episode_reward += reward
                except Exception as e:
                    print("Debug:"+e);    
                
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