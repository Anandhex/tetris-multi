import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider
import matplotlib.animation as animation
from collections import deque, defaultdict
import json
import pickle
import time
from datetime import datetime
from tqdm import tqdm
import threading
import os

# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class ExtendedPowerupTrainer:
    """Extended trainer with longer training sessions and persistent visualization"""
    
    def __init__(self, powerup_agent, buffer_size=200000, training_config=None):
        self.powerup_agent = powerup_agent
        self.buffer = None  # Will be set later
        
        # Training configuration
        self.training_config = training_config or {
            'total_episodes': 20000,  # Much longer training
            'batch_episodes': 1000,   # Train in batches
            'visualization_interval': 100,
            'save_interval': 1000,
            'curriculum_learning': True,
            'adaptive_epsilon': True
        }
        
        # Extended training statistics
        self.training_stats = {
            'decisions': [],
            'rewards': [],
            'episode_scores': [],
            'powerup_usage': defaultdict(list),
            'learning_curve': [],
            'epsilon_history': [],
            'q_value_history': [],
            'difficulty_performance': defaultdict(list),
            'training_phases': []
        }
        
        # Visualization setup
        self.visualization_active = False
        self.fig = None
        self.axes = None
        self.animation_thread = None
        self.visualization_lock = threading.Lock()
        
        # Training state
        self.current_episode = 0
        self.current_phase = 'initial'
        self.best_performance = -float('inf')
        self.patience_counter = 0
        
        # Setup matplotlib for non-blocking visualization
        plt.ion()
    
    def load_or_generate_dataset(self, dataset_path=None, num_games=8000, force_regenerate=False):
        """Load existing dataset or generate new one"""
        
        # Set default dataset path if none provided
        if dataset_path is None:
            dataset_path = "large_realistic_tetris_dataset.pkl"
        
        # Initialize buffer
        try:
            from enhanced_game_state_generator import EnhancedGameStateBuffer, RealisticTetrisGameGenerator
            self.buffer = EnhancedGameStateBuffer(max_size=200000)
        except ImportError:
            print("❌ Enhanced game state generator not found.")
            print("Make sure enhanced_game_state_generator.py is in the same directory")
            return
        
        # Check if we should force regeneration
        if force_regenerate:
            print(f"🔄 Force regeneration requested - will create new dataset")
            if os.path.exists(dataset_path):
                print(f"🗑️  Existing dataset will be overwritten: {dataset_path}")
        else:
            # Check if dataset file exists and try to load it
            if os.path.exists(dataset_path):
                print(f"📊 Found existing dataset: {dataset_path}")
                print("🔄 Loading existing dataset...")
                
                if self.buffer.load_from_file(dataset_path):
                    print(f"✅ Successfully loaded existing dataset!")
                    print(f"📈 Dataset contains {len(self.buffer.states)} states from {self.buffer.metadata['total_games']} games")
                    
                    # Show dataset statistics
                    stats = self.buffer.get_statistics()
                    print(f"📊 Dataset Statistics:")
                    print(f"   • Games: {stats['total_games']}")
                    print(f"   • States: {stats['total_states']}")
                    print(f"   • Avg game length: {stats['avg_game_length']:.1f}")
                    print(f"   • Difficulties: {dict(stats['difficulty_distribution'])}")
                    return
                else:
                    print("❌ Failed to load existing dataset, will generate new one")
            else:
                print(f"📦 No existing dataset found at: {dataset_path}")
                print("🔄 Will generate new dataset...")
        
        # Generate new dataset
        print(f"🎮 Generating new dataset with {num_games} games...")
        print(f"💾 Will save to: {dataset_path}")
        
        generator = RealisticTetrisGameGenerator()
        difficulties = ['easy', 'medium', 'hard', 'expert']
        
        for i in tqdm(range(num_games), desc="Generating realistic games"):
            difficulty = random.choice(difficulties)
            
            trajectory = generator.generate_diverse_game_trajectory(
                max_pieces=random.randint(120, 250),
                difficulty=difficulty
            )
            
            self.buffer.add_game_trajectory(trajectory)
            
            if (i + 1) % 1000 == 0:
                stats = self.buffer.get_statistics()
                print(f"\nProgress: {i+1}/{num_games} games, {stats['total_states']} total states")
        
        # Save the dataset
        print(f"💾 Saving dataset to: {dataset_path}")
        self.buffer.save_to_file(dataset_path)
        
        stats = self.buffer.get_statistics()
        print(f"✅ Dataset generation complete!")
        print(f"📊 Total states: {stats['total_states']}")
        print(f"🎮 Total games: {stats['total_games']}")
        print(f"📈 Average game length: {stats['avg_game_length']:.1f}")
        print(f"💾 Saved to: {dataset_path}")
    
    def generate_basic_dataset(self, num_games):
        """Fallback basic dataset generation"""
        print("Using basic dataset generation...")
        # This would use your existing FastGameStateGenerator
        # Implementation depends on your existing code
        pass
    
    def setup_persistent_visualization(self):
        """Setup visualization that persists after training"""
        if self.fig is not None:
            plt.close(self.fig)
        
        # Create figure that won't close automatically
        self.fig, self.axes = plt.subplots(3, 4, figsize=(20, 15))
        self.fig.suptitle('Extended Powerup Training - Live Monitoring', fontsize=16)
        
        # Configure subplots
        titles = [
            'Current Board State', 'Powerup Decision Analysis', 'Training Progress', 'Reward Distribution',
            'Bomb Placement Detail', 'Performance by Difficulty', 'Learning Curve', 'Epsilon Schedule',
            'Q-Value Distribution', 'Decision Confidence', 'Powerup Usage Stats', 'Training Phase Info'
        ]
        
        for i, ax in enumerate(self.axes.flat):
            ax.set_title(titles[i])
            ax.clear()
        
        # Add control buttons
        self.add_persistent_controls()
        
        # Enable interactive mode
        plt.ion()
        plt.show()
        
        self.visualization_active = True
        print("Persistent visualization setup complete")
    
    def add_persistent_controls(self):
        """Add controls that work throughout training"""
        # Create control panel
        control_ax = plt.axes([0.02, 0.02, 0.15, 0.12])
        control_ax.text(0.1, 0.9, 'Training Controls', fontweight='bold', transform=control_ax.transAxes)
        
        # Save model button
        save_ax = plt.axes([0.02, 0.16, 0.07, 0.03])
        self.save_button = Button(save_ax, 'Save Model')
        self.save_button.on_clicked(self.save_model_callback)
        
        # Pause/Resume button
        pause_ax = plt.axes([0.10, 0.16, 0.07, 0.03])
        self.pause_button = Button(pause_ax, 'Pause')
        self.pause_button.on_clicked(self.pause_training_callback)
        
        # Phase info
        phase_ax = plt.axes([0.02, 0.20, 0.15, 0.03])
        phase_ax.text(0.1, 0.5, f'Phase: {self.current_phase}', transform=phase_ax.transAxes)
        
        self.training_paused = False
    
    def save_model_callback(self, event):
        """Save model when button is clicked"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"powerup_model_episode_{self.current_episode}_{timestamp}.pth"
        self.powerup_agent.save_model(model_path)
        print(f"Model saved: {model_path}")
    
    def pause_training_callback(self, event):
        """Pause/resume training"""
        self.training_paused = not self.training_paused
        self.pause_button.label.set_text('Resume' if self.training_paused else 'Pause')
        print(f"Training {'paused' if self.training_paused else 'resumed'}")
    
    def update_training_phase(self, episode):
        """Update training phase based on progress"""
        total_episodes = self.training_config['total_episodes']
        progress = episode / total_episodes
        
        if progress < 0.3:
            self.current_phase = 'exploration'
        elif progress < 0.6:
            self.current_phase = 'learning'
        elif progress < 0.8:
            self.current_phase = 'optimization'
        else:
            self.current_phase = 'fine_tuning'
        
        # Adapt epsilon based on phase
        if self.training_config['adaptive_epsilon']:
            if self.current_phase == 'exploration':
                target_epsilon = 0.8
            elif self.current_phase == 'learning':
                target_epsilon = 0.5
            elif self.current_phase == 'optimization':
                target_epsilon = 0.2
            else:
                target_epsilon = 0.1
            
            # Gradually adjust epsilon
            current_epsilon = self.powerup_agent.epsilon
            self.powerup_agent.epsilon = 0.9 * current_epsilon + 0.1 * target_epsilon
    
    def curriculum_learning_sample(self, episode):
        """Sample scenarios based on curriculum learning"""
        if not self.training_config['curriculum_learning']:
            return self.buffer.sample_powerup_scenario()
        
        total_episodes = self.training_config['total_episodes']
        progress = episode / total_episodes
        
        # Start with easier scenarios, gradually increase difficulty
        if progress < 0.3:
            difficulty = random.choice(['easy', 'medium'])
            complexity = random.choice(['low', 'medium'])
        elif progress < 0.6:
            difficulty = random.choice(['medium', 'hard'])
            complexity = random.choice(['medium', 'high'])
        else:
            difficulty = random.choice(['hard', 'expert'])
            complexity = random.choice(['high', 'extreme'])
        
        return self.buffer.sample_powerup_scenario(difficulty=difficulty, complexity=complexity)
    
    def extended_training_loop(self, dataset_path=None, force_regenerate=False, num_games=8000):
        """Main extended training loop with persistent visualization"""
        print(f"🚀 Starting extended training for {self.training_config['total_episodes']} episodes")
        
        # Setup dataset - it will handle the path automatically
        if not self.buffer:
            self.load_or_generate_dataset(dataset_path, num_games, force_regenerate)
        
        if not self.buffer or len(self.buffer.states) == 0:
            print("❌ No dataset available for training!")
            return
        self.setup_persistent_visualization()
        
        # Training loop
        batch_size = self.training_config['batch_episodes']
        total_episodes = self.training_config['total_episodes']
        
        for batch_start in range(0, total_episodes, batch_size):
            batch_end = min(batch_start + batch_size, total_episodes)
            
            print(f"\nTraining batch {batch_start//batch_size + 1}: Episodes {batch_start}-{batch_end}")
            
            # Train this batch
            batch_rewards = []
            batch_decisions = []
            
            for episode in tqdm(range(batch_start, batch_end), desc=f"Batch {batch_start//batch_size + 1}"):
                # Check if training is paused
                while self.training_paused:
                    time.sleep(0.1)
                
                self.current_episode = episode
                
                # Update training phase
                self.update_training_phase(episode)
                
                # Sample scenario with curriculum learning
                scenario = self.curriculum_learning_sample(episode)
                if not scenario:
                    continue
                
                # Make powerup decision
                decision_result = self.powerup_agent.make_powerup_decision(
                    scenario['available_powerup'],
                    scenario['board_2d'],
                    scenario['board_features'],
                    scenario['blocks_since_powerup'],
                    episode
                )
                
                if not decision_result:
                    continue
                
                # Calculate reward with enhanced simulation
                reward = self.simulate_enhanced_powerup_effect(scenario, decision_result)
                
                # Store experience
                next_state = self.powerup_agent.get_placement_state(
                    scenario['board_features'], 0, 0, 'none'
                )
                
                self.powerup_agent.remember(
                    decision_result['state_features'],
                    reward,
                    next_state,
                    True
                )
                
                # Train the agent
                if len(self.powerup_agent.memory) > self.powerup_agent.batch_size:
                    self.powerup_agent.replay()
                
                # Update statistics
                self.update_training_statistics(episode, scenario, decision_result, reward)
                
                # Update visualization
                if episode % self.training_config['visualization_interval'] == 0:
                    self.update_persistent_visualization(scenario, decision_result, reward)
                
                # Save model periodically
                if episode % self.training_config['save_interval'] == 0 and episode > 0:
                    self.save_checkpoint(episode)
                
                batch_rewards.append(reward)
                batch_decisions.append(decision_result)
            
            # Analyze batch performance
            self.analyze_batch_performance(batch_start, batch_rewards, batch_decisions)
            
            # Check for early stopping
            if self.should_stop_training():
                print("Early stopping triggered")
                break
        
        # Final save and analysis
        self.finalize_training()
        
        # Keep visualization open
        self.keep_visualization_alive()
    
    def simulate_enhanced_powerup_effect(self, scenario, decision_result):
        """Enhanced powerup effect simulation"""
        decision_data = decision_result['decision_data']
        board_before = scenario['board_features']
        powerup_type = scenario['available_powerup']
        board_2d = scenario['board_2d']
        
        if decision_data['action'] == 'wait':
            # Penalty for waiting increases with board complexity
            complexity_penalty = sum(board_before[1:]) * 0.1
            return -2 - complexity_penalty
        
        # Simulate actual powerup effects
        if powerup_type == 'bomb':
            return self.simulate_bomb_effect(board_2d, decision_data, board_before)
        elif powerup_type == 'gravity':
            return self.simulate_gravity_effect(board_2d, board_before)
        elif powerup_type == 'bottom_line_clear':
            return self.simulate_bottom_clear_effect(board_2d, board_before)
        
        return 0
    
    def simulate_bomb_effect(self, board_2d, decision_data, board_before):
        """Simulate bomb explosion effect"""
        if 'column' not in decision_data:
            return 0
        
        col = decision_data['column']
        landing_row = decision_data.get('landing_row', 19)
        
        # Count blocks destroyed in 3x3 area
        blocks_destroyed = 0
        holes_filled = 0
        
        test_board = board_2d.copy()
        
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r, c = landing_row + dr, col + dc
                if 0 <= r < 20 and 0 <= c < 10:
                    if test_board[r, c] != 0:
                        blocks_destroyed += 1
                        test_board[r, c] = 0
        
        # Simulate gravity effect after explosion
        for c in range(10):
            # Drop blocks down
            empty_row = 19
            for r in range(19, -1, -1):
                if test_board[r, c] != 0:
                    if r != empty_row:
                        test_board[empty_row, c] = test_board[r, c]
                        test_board[r, c] = 0
                    empty_row -= 1
        
        # Calculate improvement
        holes_after = self.count_holes(test_board)
        height_after = self.get_max_height(test_board)
        
        holes_filled = max(0, board_before[1] - holes_after)
        height_reduced = max(0, board_before[3] - height_after)
        
        # Reward calculation
        reward = blocks_destroyed * 10 + holes_filled * 20 + height_reduced * 15
        
        # Efficiency bonus
        if blocks_destroyed >= 5:
            reward += 50
        
        return reward
    
    def simulate_gravity_effect(self, board_2d, board_before):
        """Simulate gravity powerup effect"""
        test_board = board_2d.copy()
        
        # Apply gravity - drop all blocks down
        for c in range(10):
            # Collect all blocks in this column
            blocks = []
            for r in range(20):
                if test_board[r, c] != 0:
                    blocks.append(test_board[r, c])
                test_board[r, c] = 0
            
            # Place blocks at bottom
            for i, block in enumerate(blocks):
                test_board[19 - i, c] = block
        
        # Calculate improvement
        holes_after = self.count_holes(test_board)
        holes_filled = max(0, board_before[1] - holes_after)
        
        # Check for line clears
        lines_cleared = 0
        for r in range(20):
            if all(test_board[r, c] != 0 for c in range(10)):
                lines_cleared += 1
        
        reward = holes_filled * 25 + lines_cleared * 100
        return reward
    
    def simulate_bottom_clear_effect(self, board_2d, board_before):
        """Simulate bottom line clear effect"""
        test_board = board_2d.copy()
        
        # Clear bottom line
        bottom_blocks = sum(1 for c in range(10) if test_board[19, c] != 0)
        test_board[19, :] = 0
        
        # Drop everything down
        for c in range(10):
            empty_row = 19
            for r in range(18, -1, -1):
                if test_board[r, c] != 0:
                    test_board[empty_row, c] = test_board[r, c]
                    if r != empty_row:
                        test_board[r, c] = 0
                    empty_row -= 1
        
        # Calculate improvement
        height_after = self.get_max_height(test_board)
        height_reduced = max(0, board_before[3] - height_after)
        
        reward = bottom_blocks * 20 + height_reduced * 10
        return reward
    
    def count_holes(self, board):
        """Count holes in board"""
        holes = 0
        for col in range(10):
            found_block = False
            for row in range(20):
                if board[row, col] != 0:
                    found_block = True
                elif found_block and board[row, col] == 0:
                    holes += 1
        return holes
    
    def get_max_height(self, board):
        """Get maximum height of board"""
        for row in range(20):
            if np.any(board[row, :] != 0):
                return 20 - row
        return 0
    
    def update_training_statistics(self, episode, scenario, decision_result, reward):
        """Update comprehensive training statistics"""
        # Basic stats
        self.training_stats['rewards'].append(reward)
        self.training_stats['epsilon_history'].append(self.powerup_agent.epsilon)
        self.training_stats['q_value_history'].append(decision_result['q_value'])
        
        # Powerup usage
        powerup_type = scenario['available_powerup']
        self.training_stats['powerup_usage'][powerup_type].append(reward)
        
        # Difficulty performance
        difficulty = scenario.get('difficulty', 'medium')
        self.training_stats['difficulty_performance'][difficulty].append(reward)
        
        # Decision record
        decision_record = {
            'episode': episode,
            'powerup_type': powerup_type,
            'action': decision_result['decision_data']['action'],
            'q_value': decision_result['q_value'],
            'reward': reward,
            'decision_type': decision_result['decision_type'],
            'board_complexity': sum(scenario['board_features'][1:]),
            'phase': self.current_phase
        }
        self.training_stats['decisions'].append(decision_record)
        
        # Learning curve (rolling average)
        if len(self.training_stats['rewards']) >= 100:
            recent_rewards = self.training_stats['rewards'][-100:]
            rolling_avg = np.mean(recent_rewards)
            self.training_stats['learning_curve'].append(rolling_avg)
    
    def update_persistent_visualization(self, scenario, decision_result, reward):
        """Update all visualization plots"""
        with self.visualization_lock:
            try:
                # Clear all axes
                for ax in self.axes.flat:
                    ax.clear()
                
                # Update each subplot
                self.plot_current_board(scenario, self.axes[0, 0])
                self.plot_decision_analysis(decision_result, self.axes[0, 1])
                self.plot_training_progress(self.axes[0, 2])
                self.plot_reward_distribution(self.axes[0, 3])
                
                self.plot_bomb_placement(scenario, decision_result, self.axes[1, 0])
                self.plot_difficulty_performance(self.axes[1, 1])
                self.plot_learning_curve(self.axes[1, 2])
                self.plot_epsilon_schedule(self.axes[1, 3])
                
                self.plot_q_value_distribution(self.axes[2, 0])
                self.plot_decision_confidence(self.axes[2, 1])
                self.plot_powerup_usage(self.axes[2, 2])
                self.plot_phase_info(self.axes[2, 3])
                
                plt.tight_layout()
                plt.draw()
                plt.pause(0.01)
                
            except Exception as e:
                print(f"Visualization update error: {e}")
    
    def plot_current_board(self, scenario, ax):
        """Plot current board state"""
        board = scenario['board_2d']
        powerup_type = scenario['available_powerup']
        
        # Create colored visualization
        board_colored = np.zeros((20, 10, 3))
        for i in range(20):
            for j in range(10):
                if board[i, j] != 0:
                    intensity = 0.3 + 0.7 * (board[i, j] / 7)
                    board_colored[i, j] = [0, 0, intensity]
                else:
                    board_colored[i, j] = [0.1, 0.1, 0.1]
        
        ax.imshow(board_colored, aspect='equal')
        ax.set_title(f'{powerup_type.title()} Available (Episode {self.current_episode})')
        
        # Add features text
        features = scenario['board_features']
        feature_text = f'H:{features[3]} Ho:{features[1]} B:{features[2]}'
        ax.text(0.02, 0.98, feature_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(19.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
    
    def plot_decision_analysis(self, decision_result, ax):
        """Plot decision analysis"""
        if not decision_result or 'all_evaluations' not in decision_result:
            ax.text(0.5, 0.5, 'No decision data', ha='center', va='center', transform=ax.transAxes)
            return
        
        evaluations = decision_result['all_evaluations']
        chosen_decision = decision_result['decision_data']
        
        options = []
        q_values = []
        colors = []
        
        for eval_data in evaluations:
            decision = eval_data['decision_data']
            q_val = eval_data['q_value']
            
            is_chosen = (decision == chosen_decision)
            
            if decision['action'] == 'wait':
                options.append('Wait')
                colors.append('red' if is_chosen else 'lightcoral')
            elif decision['powerup_type'] == 'bomb':
                col = decision.get('column', '?')
                options.append(f'B{col}')
                colors.append('blue' if is_chosen else 'lightblue')
            else:
                options.append('Use')
                colors.append('green' if is_chosen else 'lightgreen')
            
            q_values.append(q_val)
        
        if options:
            bars = ax.bar(range(len(options)), q_values, color=colors)
            ax.set_xticks(range(len(options)))
            ax.set_xticklabels(options, rotation=45, ha='right')
            ax.set_ylabel('Q-Value')
            ax.set_title('Decision Analysis')
    
    def plot_training_progress(self, ax):
        """Plot training progress"""
        if len(self.training_stats['rewards']) < 10:
            ax.text(0.5, 0.5, 'Collecting data...', ha='center', va='center', transform=ax.transAxes)
            return
        
        recent_rewards = self.training_stats['rewards'][-1000:]
        ax.plot(recent_rewards, alpha=0.7)
        
        # Add rolling average
        if len(recent_rewards) >= 50:
            window = 50
            rolling_avg = []
            for i in range(window, len(recent_rewards)):
                rolling_avg.append(np.mean(recent_rewards[i-window:i]))
            ax.plot(range(window, len(recent_rewards)), rolling_avg, 'r-', linewidth=2)
        
        ax.set_ylabel('Reward')
        ax.set_xlabel('Recent Episodes')
        ax.set_title('Training Progress')
        ax.grid(True, alpha=0.3)
    
    def plot_reward_distribution(self, ax):
        """Plot reward distribution"""
        rewards = self.training_stats['rewards'][-1000:]
        if len(rewards) < 10:
            ax.text(0.5, 0.5, 'Need more data', ha='center', va='center', transform=ax.transAxes)
            return
        
        ax.hist(rewards, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.1f}')
        ax.set_xlabel('Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Reward Distribution')
        ax.legend()
    
    def plot_bomb_placement(self, scenario, decision_result, ax):
        """Plot bomb placement visualization"""
        if (scenario['available_powerup'] != 'bomb' or 
            not decision_result or 
            'all_evaluations' not in decision_result):
            ax.text(0.5, 0.5, 'No bomb data', ha='center', va='center', transform=ax.transAxes)
            return
        
        board = scenario['board_2d']
        ax.imshow(board, cmap='Blues', aspect='equal', alpha=0.3)
        
        # Show bomb options
        bomb_options = [eval_data for eval_data in decision_result['all_evaluations'] 
                       if eval_data['decision_data'].get('action') != 'wait']
        
        if bomb_options:
            q_values = [opt['q_value'] for opt in bomb_options]
            q_min, q_max = min(q_values), max(q_values)
            q_range = q_max - q_min if q_max != q_min else 1
            
            for eval_data in bomb_options:
                decision = eval_data['decision_data']
                if 'column' in decision:
                    col = decision['column']
                    landing_row = decision.get('landing_row', 19)
                    q_val = eval_data['q_value']
                    
                    normalized_q = (q_val - q_min) / q_range
                    color = plt.cm.RdYlBu_r(normalized_q)
                    
                    is_chosen = (decision == decision_result['decision_data'])
                    alpha = 0.8 if is_chosen else 0.4
                    
                    explosion = patches.Rectangle((col-1.5, landing_row-1.5), 3, 3,
                                                linewidth=2, edgecolor='red',
                                                facecolor=color, alpha=alpha)
                    ax.add_patch(explosion)
        
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(19.5, -0.5)
        ax.set_title('Bomb Placement Analysis')
    
    def plot_difficulty_performance(self, ax):
        """Plot performance by difficulty"""
        difficulty_perf = self.training_stats['difficulty_performance']
        
        if not difficulty_perf:
            ax.text(0.5, 0.5, 'No difficulty data', ha='center', va='center', transform=ax.transAxes)
            return
        
        difficulties = []
        avg_rewards = []
        
        for difficulty, rewards in difficulty_perf.items():
            if len(rewards) >= 10:
                difficulties.append(difficulty)
                avg_rewards.append(np.mean(rewards[-100:]))
        
        if difficulties:
            bars = ax.bar(difficulties, avg_rewards, color=['green', 'yellow', 'orange', 'red'][:len(difficulties)])
            ax.set_ylabel('Average Reward')
            ax.set_title('Performance by Difficulty')
            
            for bar, reward in zip(bars, avg_rewards):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{reward:.1f}', ha='center', va='bottom')
    
    def plot_learning_curve(self, ax):
        """Plot learning curve"""
        learning_curve = self.training_stats['learning_curve']
        
        if len(learning_curve) < 5:
            ax.text(0.5, 0.5, 'Building curve...', ha='center', va='center', transform=ax.transAxes)
            return
        
        ax.plot(learning_curve, 'g-', linewidth=2)
        ax.set_ylabel('Average Reward (100 episodes)')
        ax.set_xlabel('Training Progress')
        ax.set_title('Learning Curve')
        ax.grid(True, alpha=0.3)
    
    def plot_epsilon_schedule(self, ax):
        """Plot epsilon decay schedule"""
        epsilon_history = self.training_stats['epsilon_history']
        
        if len(epsilon_history) < 10:
            ax.text(0.5, 0.5, 'Tracking epsilon...', ha='center', va='center', transform=ax.transAxes)
            return
        
        ax.plot(epsilon_history, 'b-', linewidth=2)
        ax.set_ylabel('Epsilon')
        ax.set_xlabel('Episodes')
        ax.set_title('Epsilon Schedule')
        ax.grid(True, alpha=0.3)
    
    def plot_q_value_distribution(self, ax):
        """Plot Q-value distribution"""
        q_values = self.training_stats['q_value_history'][-1000:]
        
        if len(q_values) < 10:
            ax.text(0.5, 0.5, 'Collecting Q-values...', ha='center', va='center', transform=ax.transAxes)
            return
        
        ax.hist(q_values, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(q_values), color='red', linestyle='--', label=f'Mean: {np.mean(q_values):.2f}')
        ax.set_xlabel('Q-Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Q-Value Distribution')
        ax.legend()
    
    def plot_decision_confidence(self, ax):
        """Plot decision confidence over time"""
        decisions = self.training_stats['decisions'][-500:]
        
        if len(decisions) < 10:
            ax.text(0.5, 0.5, 'Building confidence...', ha='center', va='center', transform=ax.transAxes)
            return
        
        confidences = [abs(d['q_value']) for d in decisions]
        ax.plot(confidences, alpha=0.7)
        
        # Add trend line
        if len(confidences) >= 20:
            z = np.polyfit(range(len(confidences)), confidences, 1)
            p = np.poly1d(z)
            ax.plot(range(len(confidences)), p(range(len(confidences))), "r--", alpha=0.8)
        
        ax.set_ylabel('Confidence (|Q-value|)')
        ax.set_xlabel('Recent Decisions')
        ax.set_title('Decision Confidence Trend')
        ax.grid(True, alpha=0.3)
    
    def plot_powerup_usage(self, ax):
        """Plot powerup usage statistics"""
        powerup_usage = self.training_stats['powerup_usage']
        
        if not powerup_usage:
            ax.text(0.5, 0.5, 'No usage data', ha='center', va='center', transform=ax.transAxes)
            return
        
        powerups = []
        avg_rewards = []
        usage_counts = []
        
        for powerup, rewards in powerup_usage.items():
            if len(rewards) > 0:
                powerups.append(powerup[:6])  # Shorten names
                avg_rewards.append(np.mean(rewards))
                usage_counts.append(len(rewards))
        
        if powerups:
            bars = ax.bar(powerups, avg_rewards, color=['red', 'blue', 'green'][:len(powerups)])
            ax.set_ylabel('Average Reward')
            ax.set_title('Powerup Usage Performance')
            
            for bar, reward, count in zip(bars, avg_rewards, usage_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{reward:.1f}\n({count})', ha='center', va='bottom')
    
    def plot_phase_info(self, ax):
        """Plot current training phase information"""
        ax.axis('off')
        
        # Create info text
        info_text = []
        info_text.append(f"Episode: {self.current_episode}")
        info_text.append(f"Phase: {self.current_phase}")
        info_text.append(f"Epsilon: {self.powerup_agent.epsilon:.3f}")
        
        if self.training_stats['rewards']:
            recent_avg = np.mean(self.training_stats['rewards'][-100:])
            info_text.append(f"Recent Avg: {recent_avg:.1f}")
        
        info_text.append(f"Memory: {len(self.powerup_agent.memory)}")
        info_text.append(f"Best: {self.best_performance:.1f}")
        
        # Display text
        full_text = '\n'.join(info_text)
        ax.text(0.05, 0.95, full_text, transform=ax.transAxes,
                verticalalignment='top', fontfamily='monospace', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    def analyze_batch_performance(self, batch_start, batch_rewards, batch_decisions):
        """Analyze performance of completed batch"""
        if not batch_rewards:
            return
        
        avg_reward = np.mean(batch_rewards)
        
        # Update best performance
        if avg_reward > self.best_performance:
            self.best_performance = avg_reward
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Print batch summary
        print(f"Batch {batch_start//self.training_config['batch_episodes'] + 1} complete:")
        print(f"  Average reward: {avg_reward:.2f}")
        print(f"  Best performance: {self.best_performance:.2f}")
        print(f"  Epsilon: {self.powerup_agent.epsilon:.3f}")
        print(f"  Phase: {self.current_phase}")
    
    def should_stop_training(self):
        """Check if training should stop early"""
        # Stop if no improvement for many batches
        patience_limit = 10
        return self.patience_counter >= patience_limit
    
    def save_checkpoint(self, episode):
        """Save training checkpoint"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = f"checkpoints/powerup_model_ep{episode}_{timestamp}.pth"
        self.powerup_agent.save_model(model_path)
        
        # Save training stats
        stats_path = f"checkpoints/training_stats_ep{episode}_{timestamp}.json"
        with open(stats_path, 'w') as f:
            # Convert numpy types for JSON serialization
            stats_copy = {}
            for key, value in self.training_stats.items():
                if isinstance(value, list):
                    stats_copy[key] = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in value]
                else:
                    stats_copy[key] = value
            json.dump(stats_copy, f, indent=2)
        
        print(f"Checkpoint saved: {model_path}")
    
    def finalize_training(self):
        """Finalize training and save final results"""
        print("\nFinalizing training...")
        
        # Save final model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_model_path = f"final_powerup_model_{timestamp}.pth"
        self.powerup_agent.save_model(final_model_path)
        
        # Save comprehensive stats
        final_stats_path = f"final_training_stats_{timestamp}.json"
        with open(final_stats_path, 'w') as f:
            stats_copy = {}
            for key, value in self.training_stats.items():
                if isinstance(value, list):
                    stats_copy[key] = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in value]
                else:
                    stats_copy[key] = value
            json.dump(stats_copy, f, indent=2)
        
        # Print final summary
        self.print_final_summary()
        
        print(f"Final model saved: {final_model_path}")
        print(f"Final stats saved: {final_stats_path}")
    
    def print_final_summary(self):
        """Print comprehensive training summary"""
        print("\n" + "="*80)
        print("EXTENDED POWERUP TRAINING SUMMARY")
        print("="*80)
        
        if self.training_stats['rewards']:
            total_episodes = len(self.training_stats['rewards'])
            avg_reward = np.mean(self.training_stats['rewards'])
            final_epsilon = self.powerup_agent.epsilon
            
            print(f"Total episodes: {total_episodes}")
            print(f"Average reward: {avg_reward:.2f}")
            print(f"Best performance: {self.best_performance:.2f}")
            print(f"Final epsilon: {final_epsilon:.3f}")
            
            # Performance by powerup
            print("\nPowerup Performance:")
            for powerup_type, rewards in self.training_stats['powerup_usage'].items():
                if rewards:
                    print(f"  {powerup_type}: {len(rewards)} uses, avg reward: {np.mean(rewards):.2f}")
            
            # Performance by difficulty
            print("\nDifficulty Performance:")
            for difficulty, rewards in self.training_stats['difficulty_performance'].items():
                if rewards:
                    print(f"  {difficulty}: {len(rewards)} scenarios, avg reward: {np.mean(rewards):.2f}")
            
            # Learning phases
            print(f"\nTraining phases completed: {len(set(d['phase'] for d in self.training_stats['decisions']))}")
            
            # Decision type distribution
            exploration_count = sum(1 for d in self.training_stats['decisions'] if d['decision_type'] == 'exploration')
            exploitation_count = len(self.training_stats['decisions']) - exploration_count
            print(f"Exploration: {exploration_count}, Exploitation: {exploitation_count}")
            
            print("="*80)
    
    def keep_visualization_alive(self):
        """Keep visualization window open after training"""
        print("\nTraining complete! Visualization will remain open.")
        print("You can:")
        print("- Examine the final training results")
        print("- Save the current model using the 'Save Model' button")
        print("- Close the window when done")
        print("\nVisualization controls:")
        print("- Save Model: Save current model state")
        print("- Close the window to exit")
        
        # Add a completion message to the visualization
        if self.fig and self.axes is not None:
            # Add completion banner
            self.fig.suptitle('Extended Powerup Training - COMPLETED', fontsize=18, color='green')
            
            # Update phase info to show completion
            ax = self.axes[2, 3]
            ax.clear()
            ax.axis('off')
            
            completion_text = [
                "TRAINING COMPLETED!",
                "",
                f"Episodes: {len(self.training_stats['rewards'])}",
                f"Final Epsilon: {self.powerup_agent.epsilon:.3f}",
                f"Best Performance: {self.best_performance:.1f}",
                f"Phase: {self.current_phase}",
                "",
                "Window will stay open",
                "for result examination"
            ]
            
            full_text = '\n'.join(completion_text)
            ax.text(0.05, 0.95, full_text, transform=ax.transAxes,
                    verticalalignment='top', fontfamily='monospace', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            plt.draw()
        
        # Keep the visualization alive
        try:
            # This will keep the window open until manually closed
            plt.show(block=True)
        except KeyboardInterrupt:
            print("Visualization closed by user")
        except Exception as e:
            print(f"Visualization error: {e}")
            # Fallback - just keep the program running
            print("Press Ctrl+C to exit...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Exiting...")

# Integration with existing code
def run_extended_powerup_training():
    """Main function to run extended training"""
    # Import your existing components
    try:
        from powerup_dqn_network import FinalPowerUpDQNAgent
    except ImportError:
        print("Error: Could not import FinalPowerUpDQNAgent")
        print("Make sure powerup_dqn_network.py is in the same directory")
        return
    
    # Initialize agent with extended training configuration
    print("Initializing PowerUp DQN Agent for extended training...")
    powerup_agent = FinalPowerUpDQNAgent(
        state_size=7,
        learning_rate=0.001,
        epsilon=0.9,  # Higher initial epsilon for longer exploration
        epsilon_decay=0.9995,  # Slower decay for extended training
        epsilon_min=0.05,  # Slightly higher minimum
        memory_size=50000,  # Larger memory for extended training
        batch_size=64,  # Larger batch size
        target_update_freq=2000  # Less frequent updates for stability
    )
    
    # Enhanced training configuration
    training_config = {
        'total_episodes': 25000,  # Extended training
        'batch_episodes': 500,    # Smaller batches for frequent updates
        'visualization_interval': 50,  # More frequent visualization
        'save_interval': 1000,    # Save checkpoints every 1000 episodes
        'curriculum_learning': True,
        'adaptive_epsilon': True
    }
    
    # Create trainer
    trainer = ExtendedPowerupTrainer(powerup_agent, training_config=training_config)
    
    # Run extended training
    dataset_path = "large_realistic_tetris_dataset.pkl"
    trainer.extended_training_loop(dataset_path)

# Enhanced usage example with better dataset
def create_comprehensive_dataset():
    """Create a comprehensive dataset for training"""
    print("Creating comprehensive Tetris dataset...")
    
    # Try to use enhanced generator if available
    try:
        from enhanced_game_state_generator import RealisticTetrisGameGenerator, EnhancedGameStateBuffer
        
        generator = RealisticTetrisGameGenerator()
        buffer = EnhancedGameStateBuffer(max_size=300000)
        
        # Generate diverse scenarios
        num_games = 10000
        difficulties = ['easy', 'medium', 'hard', 'expert']
        
        print(f"Generating {num_games} diverse games...")
        
        for i in tqdm(range(num_games), desc="Creating dataset"):
            # Vary difficulty and length
            difficulty = random.choice(difficulties)
            max_pieces = random.randint(80, 300)
            
            # Generate realistic game
            trajectory = generator.generate_diverse_game_trajectory(
                max_pieces=max_pieces,
                difficulty=difficulty
            )
            
            buffer.add_game_trajectory(trajectory)
            
            # Progress update
            if (i + 1) % 1000 == 0:
                stats = buffer.get_statistics()
                print(f"Progress: {i+1}/{num_games} games, {stats['total_states']} states")
        
        # Save dataset
        buffer.save_to_file("comprehensive_tetris_dataset.pkl")
        
        print("Dataset creation complete!")
        print(f"Total games: {buffer.metadata['total_games']}")
        print(f"Total states: {buffer.metadata['total_states']}")
        print(f"Average game length: {buffer.metadata['avg_game_length']:.1f}")
        
        return buffer
        
    except ImportError:
        print("Enhanced generator not available, using basic approach")
        return None

# Complete training pipeline
def complete_training_pipeline():
    """Complete training pipeline with dataset creation and extended training"""
    print("Starting complete powerup training pipeline...")
    
    # Step 1: Create comprehensive dataset
    dataset = create_comprehensive_dataset()
    
    # Step 2: Run extended training
    print("\nStarting extended training...")
    run_extended_powerup_training()
    
    print("Complete training pipeline finished!")

if __name__ == "__main__":
    # You can run either individual components or the complete pipeline
    
    # For just extended training (if you already have a dataset):
    # run_extended_powerup_training()
    
    # For complete pipeline (dataset creation + training):
    complete_training_pipeline()