from simplified_dqn_network import EnhancedDQNAgent
from powerup_dqn_network import FinalPowerUpDQNAgent  # Fixed import
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

class FinalCompleteTrainer:
    """Final complete trainer with all components properly integrated"""
    
    def __init__(self,
                 block_model_path: str = 'model_20250706-105237.h5',
                 powerup_model_path: str = None,
                 load_powerup_model: bool = False,
                 tensorboard_log_dir: str = None,
                 episodes: int = 3000):
        
        # Unity client
        self.client = UnityTetrisClient()
        
        # Training hyperparameters
        self.episodes = episodes
        self.max_steps = None
        self.powerup_interval = 5
        self.max_powerup_hold = 10
        
        # Logging
        self.start_at = f"final_complete_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.best_score = -float('inf')
        self.total_steps = 0
        
        # Tensorboard paths
        if tensorboard_log_dir is None:
            tensorboard_log_dir = f"logs/{self.start_at}"
        
        block_log_dir = f"{tensorboard_log_dir}/block_placement"
        powerup_log_dir = f"{tensorboard_log_dir}/final_powerup"
        
        # Initialize block placement agent (pre-trained)
        self.block_agent = EnhancedDQNAgent(
            n_neurons=[32, 32, 32],
            activations=['relu', 'relu', 'relu', 'linear'],
            epsilon=0,  # No exploration for trained model
            mem_size=1000,
            discount=0.95,
            tensorboard_log_dir=block_log_dir
        )
        
        # Load pre-trained block placement model
        if os.path.exists(block_model_path):
            self.block_agent.model.load_weights(block_model_path)
            print(f"✓ Loaded block placement model from {block_model_path}")
        else:
            raise FileNotFoundError(f"Block placement model not found: {block_model_path}")
        
        # Initialize final PowerUp DQN - Fixed class name
        self.powerup_agent = FinalPowerUpDQNAgent(
            state_size=7,
            tensorboard_log_dir=powerup_log_dir
        )
        
        # Load powerup model if requested
        if load_powerup_model and powerup_model_path and os.path.exists(powerup_model_path):
            self.powerup_agent.load_model(powerup_model_path)
            print(f"✓ Loaded powerup model from {powerup_model_path}")
        
        # PowerUp tracking
        self.powerup_types = ['bottom_line_clear', 'gravity', 'bomb']
        self.current_powerup = None
        self.blocks_since_powerup = 0
        self.powerup_rewards = []
        self.powerup_usage_stats = {
            pt: {
                'used': 0, 
                'total': 0, 
                'avg_reward': 0.0, 
                'decisions': [],
                'total_reward': 0.0
            } 
            for pt in self.powerup_types
        }
        
        print(f"✓ Final Complete Trainer initialized")
        print(f"✓ Device: {self.powerup_agent.device}")
        print(f"✓ Tensorboard: {tensorboard_log_dir}")
    
    def calculate_bumpiness_from_board(self, board_2d):
        """Calculate bumpiness from 2D board representation"""
        if board_2d is None or board_2d.size == 0:
            return 0
        
        heights = []
        for col in range(board_2d.shape[1]):
            height = 0
            for row in range(board_2d.shape[0]):
                if board_2d[row, col] != 0:
                    height = board_2d.shape[0] - row
                    break
            heights.append(height)
        
        # Calculate bumpiness as sum of height differences
        bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))
        return bumpiness
    
    def get_board_features_and_2d(self, game_state=None):
        """
        Get both board features and 2D representation
        
        Returns:
            tuple: (board_features, board_2d)
        """
        if game_state is None:
            game_state = self.client.get_game_state(timeout=1.0)
        
        if not game_state:
            return [0, 0, 0, 0], np.zeros((20, 10))
        
        # Get board metrics
        board_metrics = self.client.get_board_metrics(game_state)
        board_2d = self.client.get_board_state(game_state)
        
        lines = board_metrics.get('lines_cleared', 0)
        holes = board_metrics.get('holes_count', 0)
        height = board_metrics.get('stack_height', 0)
        
        # Calculate bumpiness from 2D board
        bumpiness = self.calculate_bumpiness_from_board(board_2d)
        
        return [lines, holes, bumpiness, height], board_2d
    
    def simulate_powerup_effect(self, decision_data, board_features_before):
        """Simulate powerup effect when Unity execution fails"""
        powerup_type = decision_data['powerup_type']
        action = decision_data['action']
        
        if action == 'wait':
            return board_features_before, -1, {'success': True, 'action': 'wait'}
        
        # Simulate improvement based on powerup type and impact
        impact = decision_data.get('impact', 0)
        
        if powerup_type == 'bomb':
            holes_reduced = max(0, min(int(impact / 15), board_features_before[1]))
            bumpiness_reduced = max(0, min(int(impact / 20), board_features_before[2] // 2))
            height_reduced = max(0, min(2, int(impact / 30)))
            lines_cleared = 0
            
        elif powerup_type == 'gravity':
            holes_reduced = max(0, min(int(impact / 15), board_features_before[1]))
            bumpiness_reduced = random.randint(0, 1)
            height_reduced = max(0, min(2, holes_reduced // 3))
            lines_cleared = 0
            
        elif powerup_type == 'bottom_line_clear':
            lines_cleared = max(1, min(2, int(impact / 40)))
            holes_reduced = random.randint(0, 2)
            bumpiness_reduced = random.randint(0, 1)
            height_reduced = lines_cleared
        
        else:
            return board_features_before, 0, {'success': False}
        
        # Calculate new board features
        board_features_after = [
            board_features_before[0] + lines_cleared,
            max(0, board_features_before[1] - holes_reduced),
            max(0, board_features_before[2] - bumpiness_reduced),
            max(0, board_features_before[3] - height_reduced)
        ]
        
        # Create mock execution result
        mock_result = {
            'success': True,
            'action': action,
            'powerup_type': powerup_type,
            'impact_metrics': {
                'lines_cleared': lines_cleared,
                'holes_filled': holes_reduced,
                'bumpiness_reduced': bumpiness_reduced,
                'height_reduced': height_reduced,
                'actual_impact': impact
            },
            'simulated': True
        }
        
        return board_features_after, impact, mock_result
    
    def train(self):
        """Main training loop with complete integration"""
        scores = []
        
        # Connect to Unity
        print("🔗 Connecting to Unity...")
        self.client.connect()
        
        print(f"🚀 Starting final PowerUp training for {self.episodes} episodes...")
        
        for episode in tqdm(range(self.episodes), desc="Training Episodes"):
            try:
                print(f"\n🎮 Starting Episode {episode}")
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
                episode_decisions = []
                
                while not done and (self.max_steps is None or steps < self.max_steps):
                    # ═══════════════════════════════════════
                    # PHASE 1: BLOCK PLACEMENT (Pre-trained)
                    # ═══════════════════════════════════════
                    
                    # Get possible block placements
                    next_states = self.client.get_possible_states()
                    if not next_states:
                        print("🐛 DEBUG: No possible states available")
                        break
                    
                    # Use pre-trained block placement agent
                    action_map = {}
                    for key, feats in next_states.items():
                        col_str, rot_str = key.split(":")
                        col_i, rot_i = int(col_str), int(rot_str)
                        action_map[tuple(feats)] = (col_i, rot_i)
                    
                    feature_list = list(action_map.keys())
                    best_state = self.block_agent.best_state(feature_list, episode)
                    col, rot = action_map[tuple(best_state)]
                    
                    # ═══════════════════════════════════════
                    # PHASE 2: POWERUP ASSIGNMENT
                    # ═══════════════════════════════════════
                    
                    # Get current board state
                    board_features, board_2d = self.get_board_features_and_2d()
                    if board_features is None:
                        break
                    
                    # Check powerup assignment
                    if blocks_placed > 0 and blocks_placed % self.powerup_interval == 0 and self.current_powerup is None:
                        self.current_powerup = random.choice(self.powerup_types)
                        self.blocks_since_powerup = 0
                        self.powerup_usage_stats[self.current_powerup]['total'] += 1
                        print(f"🎁 PowerUp assigned: {self.current_powerup}")
                    
                    # ═══════════════════════════════════════
                    # PHASE 3: POWERUP DECISION (Complete)
                    # ═══════════════════════════════════════
                    
                    powerup_reward = 0
                    
                    if self.current_powerup is not None:
                        # Make complete powerup decision
                        decision_result = self.powerup_agent.make_powerup_decision(
                            self.current_powerup, board_2d, board_features, 
                            self.blocks_since_powerup, steps
                        )
                        
                        if decision_result:
                            decision_data = decision_result['decision_data']
                            
                            # Force use if held too long
                            if self.blocks_since_powerup >= self.max_powerup_hold and decision_data['action'] == 'wait':
                                # Find best use option from all evaluations
                                use_options = [eval for eval in decision_result['all_evaluations'] 
                                            if eval['decision_data']['action'] != 'wait']
                                if use_options:
                                    best_use = max(use_options, key=lambda x: x['q_value'])
                                    decision_result['decision_data'] = best_use['decision_data']
                                    decision_result['state_features'] = best_use['state_features']
                                    decision_result['q_value'] = best_use['q_value']
                                    decision_data = decision_result['decision_data']
                            
                            if decision_data['action'] != 'wait':
                                # ═══════════════════════════════════════
                                # PHASE 4: POWERUP EXECUTION & REWARD
                                # ═══════════════════════════════════════
                                
                                # Execute powerup
                                try:
                                    execution_result = self.client.execute_powerup_decision(decision_result)
                                    
                                    if execution_result.get('success', False):
                                        # Extract board features after powerup
                                        impact_metrics = execution_result.get('impact_metrics', {})
                                        
                                        lines_after = board_features[0] + impact_metrics.get('lines_cleared', 0)
                                        holes_after = max(0, board_features[1] - impact_metrics.get('holes_filled', 0))
                                        bumpiness_after = max(0, board_features[2] - impact_metrics.get('bumpiness_reduced', 0))
                                        height_after = max(0, board_features[3] - impact_metrics.get('height_reduced', 0))
                                        
                                        board_features_after = [lines_after, holes_after, bumpiness_after, height_after]
                                        actual_impact = impact_metrics.get('actual_impact', decision_data.get('impact', 0))
                                    else:
                                        # Execution failed, use simulation
                                        board_features_after, actual_impact, execution_result = self.simulate_powerup_effect(
                                            decision_data, board_features
                                        )
                                        
                                except Exception as e:
                                    print(f"⚠️ PowerUp execution error: {e}")
                                    # Use simulation as fallback
                                    board_features_after, actual_impact, execution_result = self.simulate_powerup_effect(
                                        decision_data, board_features
                                    )
                                
                                # Calculate reward
                                powerup_reward = self.powerup_agent.calculate_placement_reward(
                                    board_features, board_features_after, self.current_powerup, decision_data
                                )
                                
                                # Store experience for training
                                next_state_features = self.powerup_agent.get_placement_state(
                                    board_features_after, 0, 0, 'none'
                                )
                                
                                self.powerup_agent.remember(
                                    decision_result['state_features'], powerup_reward, next_state_features, True
                                )
                                
                                # Update statistics
                                self.powerup_usage_stats[self.current_powerup]['used'] += 1
                                self.powerup_usage_stats[self.current_powerup]['total_reward'] += powerup_reward
                                
                                decision_record = {
                                    'powerup_type': self.current_powerup,
                                    'action': decision_data['action'],
                                    'q_value': decision_result['q_value'],
                                    'reward': powerup_reward,
                                    'decision_type': decision_result['decision_type'],
                                    'predicted_impact': decision_data.get('impact', 0),
                                    'actual_impact': actual_impact
                                }
                                
                                if self.current_powerup == 'bomb':
                                    decision_record.update({
                                        'column': decision_data.get('column', -1),
                                        'landing_row': decision_data.get('landing_row', -1)
                                    })
                                
                                self.powerup_usage_stats[self.current_powerup]['decisions'].append(decision_record)
                                episode_powerup_rewards.append(powerup_reward)
                                episode_decisions.append(decision_record)
                                
                                # Logging
                                if self.current_powerup == 'bomb':
                                    print(f"💣 Bomb used: column {decision_data.get('column', 'N/A')}, "
                                        f"Q={decision_result['q_value']:.2f}, reward={powerup_reward:.1f}")
                                else:
                                    print(f"⚡ {self.current_powerup} used: "
                                        f"Q={decision_result['q_value']:.2f}, reward={powerup_reward:.1f}")
                                
                                self.current_powerup = None
                                self.blocks_since_powerup = 0
                                
                            else:
                                # Hold/wait decision
                                self.blocks_since_powerup += 1
                                powerup_reward = -1
                                
                                # Store experience for waiting
                                self.powerup_agent.remember(
                                    decision_result['state_features'], powerup_reward, 
                                    decision_result['state_features'], False
                                )
                    
                    # ═══════════════════════════════════════
                    # PHASE 5: BLOCK EXECUTION
                    # ═══════════════════════════════════════
                    
                    # Execute block placement
                    print(f"🐛 DEBUG: Executing block placement - col: {col}, rot: {rot}")
                    curr_meta = self.client.send_action_and_wait({"col": col, "rot": rot}, timeout=30.0)
                    if curr_meta is None:
                        print("🐛 DEBUG: No response from block placement")
                        break
                    
                    done = curr_meta.get('gameOver', False)
                    reward = curr_meta.get('reward', 0)
                    episode_score = curr_meta.get('score', 0)
                    
                    if done:
                        print(f"🎯 Episode {episode} ended - Final Score: {episode_score}, Steps: {steps}")
                        break
                    
                    # Update counters
                    blocks_placed += 1
                    if self.current_powerup is not None:
                        self.blocks_since_powerup += 1
                    
                    episode_reward += reward + powerup_reward
                    episode_score = curr_meta.get('score', 0)
                    steps += 1
                    self.total_steps += 1
                
                # Skip very short episodes
                if steps <= 6:
                    continue
                
                # ═══════════════════════════════════════
                # PHASE 6: TRAINING UPDATE
                # ═══════════════════════════════════════
                
                # Train powerup agent
                if len(self.powerup_agent.memory) > self.powerup_agent.batch_size:
                    for _ in range(3):  # Multiple training steps per episode
                        self.powerup_agent.replay()
                
                # Record metrics
                scores.append(episode_reward)
                self.powerup_rewards.extend(episode_powerup_rewards)
                
                # Logging
                if episode % 50 == 0:
                    self.log_episode_metrics(episode, episode_reward, episode_score, 
                                        episode_powerup_rewards, steps, blocks_placed)
                
                # Save best model
                if episode_reward > self.best_score:
                    self.best_score = episode_reward
                    self.powerup_agent.save_model(f"{self.start_at}_best.pth")
                    print(f"🏆 New best model saved! Score: {episode_reward:.2f}")
                
                # Periodic checkpoint
                if episode % 100 == 0:
                    self.powerup_agent.save_model(f"{self.start_at}_checkpoint.pth")
                    self.save_training_stats(episode)
            except Exception as e:
                print(f"🐛 ERROR in Episode {episode}: {e}")
                print(f"🐛 DEBUG: Attempting to reconnect...")
                try:
                    self.client.disconnect()
                    time.sleep(1)
                    self.client.connect()
                except:
                    print(f"🐛 ERROR: Failed to reconnect. Stopping training.")
                    break
                continue
        
        # Final save and cleanup
        self.powerup_agent.save_model(f"{self.start_at}_final.pth")
        self.powerup_agent.close()
        self.block_agent.close()
        self.client.disconnect()
        
        print("✅ Final PowerUp training complete!")
        self.print_final_stats()
    
    def log_episode_metrics(self, episode, episode_reward, episode_score, 
                          powerup_rewards, steps, blocks_placed):
        """Log comprehensive metrics"""
        writer = self.powerup_agent.writer
        
        # Basic metrics
        writer.add_scalar('Episode/Total_Reward', episode_reward, episode)
        writer.add_scalar('Episode/Game_Score', episode_score, episode)
        writer.add_scalar('Episode/Steps', steps, episode)
        writer.add_scalar('Episode/Blocks_Placed', blocks_placed, episode)
        
        # PowerUp metrics
        if powerup_rewards:
            writer.add_scalar('PowerUp/Episode_Reward', sum(powerup_rewards), episode)
            writer.add_scalar('PowerUp/Avg_Reward', mean(powerup_rewards), episode)
            writer.add_scalar('PowerUp/Count', len(powerup_rewards), episode)
        
        # Usage statistics per powerup type
        for powerup_type, stats in self.powerup_usage_stats.items():
            if stats['total'] > 0:
                usage_rate = stats['used'] / stats['total']
                avg_reward = stats['total_reward'] / max(1, stats['used'])
                
                writer.add_scalar(f'PowerUp_Usage/{powerup_type}_rate', usage_rate, episode)
                writer.add_scalar(f'PowerUp_Performance/{powerup_type}_avg_reward', avg_reward, episode)
                
                # Decision type analysis
                if stats['decisions']:
                    recent_decisions = stats['decisions'][-10:]
                    exploration_rate = sum(1 for d in recent_decisions if d['decision_type'] == 'exploration') / len(recent_decisions)
                    avg_q_value = mean([d['q_value'] for d in recent_decisions])
                    
                    writer.add_scalar(f'PowerUp_Learning/{powerup_type}_exploration_rate', exploration_rate, episode)
                    writer.add_scalar(f'PowerUp_Learning/{powerup_type}_avg_q_value', avg_q_value, episode)
                    
                    # Bomb-specific metrics
                    if powerup_type == 'bomb':
                        bomb_decisions = [d for d in recent_decisions if 'column' in d and d['column'] >= 0]
                        if bomb_decisions:
                            avg_column = mean([d['column'] for d in bomb_decisions])
                            writer.add_scalar(f'Bomb_Placement/avg_column', avg_column, episode)
        
        # Running averages
        if len(scores := [episode_reward]) >= 50:
            writer.add_scalar('Stats/Avg_Score_50', mean(scores[-50:]), episode)
        
        print(f"📊 Episode {episode}: Score={episode_score}, Reward={episode_reward:.2f}, "
              f"PowerUps={len(powerup_rewards)}, Steps={steps}")
    
    def save_training_stats(self, episode):
        """Save detailed training statistics"""
        stats = {
            'episode': episode,
            'total_steps': self.total_steps,
            'best_score': self.best_score,
            'powerup_usage_stats': {}
        }
        
        for powerup_type, data in self.powerup_usage_stats.items():
            decisions = data['decisions']
            stats['powerup_usage_stats'][powerup_type] = {
                'total_assigned': data['total'],
                'total_used': data['used'],
                'usage_rate': data['used'] / max(1, data['total']),
                'avg_reward': data['total_reward'] / max(1, data['used']),
                'total_reward': data['total_reward']
            }
            
            if decisions:
                # Decision analysis
                recent_decisions = decisions[-50:] if len(decisions) > 50 else decisions
                stats['powerup_usage_stats'][powerup_type].update({
                    'avg_q_value': mean([d['q_value'] for d in recent_decisions]),
                    'exploration_rate': sum(1 for d in recent_decisions if d['decision_type'] == 'exploration') / len(recent_decisions),
                    'avg_predicted_impact': mean([d['predicted_impact'] for d in recent_decisions]),
                    'avg_actual_impact': mean([d['actual_impact'] for d in recent_decisions])
                })
                
                # Bomb-specific analysis
                if powerup_type == 'bomb':
                    bomb_decisions = [d for d in recent_decisions if 'column' in d and d['column'] >= 0]
                    if bomb_decisions:
                        column_distribution = {}
                        for d in bomb_decisions:
                            col = d['column']
                            column_distribution[col] = column_distribution.get(col, 0) + 1
                        
                        stats['powerup_usage_stats'][powerup_type]['column_distribution'] = column_distribution
                        stats['powerup_usage_stats'][powerup_type]['most_used_column'] = max(column_distribution.items(), key=lambda x: x[1])[0] if column_distribution else -1
        
        with open(f'{self.start_at}_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
    
    def print_final_stats(self):
        """Print final training statistics"""
        print("\n" + "="*70)
        print("🏁 FINAL COMPLETE POWERUP TRAINING RESULTS")
        print("="*70)
        print(f"🏆 Best Score: {self.best_score:.2f}")
        print(f"📈 Total Steps: {self.total_steps}")
        print(f"🎮 Device Used: {self.powerup_agent.device}")
        
        print("\n📊 PowerUp Usage Statistics:")
        for powerup_type, stats in self.powerup_usage_stats.items():
            if stats['total'] > 0:
                usage_rate = stats['used'] / stats['total'] * 100
                avg_reward = stats['total_reward'] / max(1, stats['used'])
                
                print(f"  {powerup_type.upper()}:")
                print(f"    Usage Rate: {usage_rate:.1f}% ({stats['used']}/{stats['total']})")
                print(f"    Avg Reward: {avg_reward:.1f}")
                print(f"    Total Reward: {stats['total_reward']:.1f}")
                
                # Recent decision analysis
                if stats['decisions']:
                    recent = stats['decisions'][-20:] if len(stats['decisions']) > 20 else stats['decisions']
                    avg_q = mean([d['q_value'] for d in recent])
                    exploration_rate = sum(1 for d in recent if d['decision_type'] == 'exploration') / len(recent) * 100
                    
                    print(f"    Recent Avg Q-value: {avg_q:.2f}")
                    print(f"    Recent Exploration: {exploration_rate:.1f}%")
                    
                    # Bomb-specific analysis
                    if powerup_type == 'bomb':
                        bomb_decisions = [d for d in recent if 'column' in d and d['column'] >= 0]
                        if bomb_decisions:
                            columns_used = [d['column'] for d in bomb_decisions]
                            column_counts = {}
                            for col in columns_used:
                                column_counts[col] = column_counts.get(col, 0) + 1
                            
                            if column_counts:
                                most_used = max(column_counts.items(), key=lambda x: x[1])
                                print(f"    Most Used Column: {most_used[0]} ({most_used[1]} times)")
        
        print(f"\n💾 Model saved as: {self.start_at}_final.pth")
        print(f"📊 Training stats: {self.start_at}_stats.json")
        print("="*70)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Starting Final Complete PowerUp Trainer")
    print("="*50)
    
    trainer = FinalCompleteTrainer(
        block_model_path='model_20250706-105237.h5',
        load_powerup_model=False,  # Set to True to load existing powerup model
        # powerup_model_path='path/to/existing/powerup_model.pth',  # Uncomment to load
        episodes=3000
    )
    
    trainer.train()