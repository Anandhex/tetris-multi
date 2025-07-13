# enhanced_training_visualizer.py - Fixed visualization with WildBlock support
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict, Counter
import pickle
import json
import os
from typing import Dict, List, Tuple, Optional

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EnhancedTrainingVisualizer:
    """
    Enhanced visualization for CNN DQN training metrics with WildBlock support
    """
    
    def __init__(self):
        self.training_data = {
            'episode_rewards': [],
            'validation_rewards': [],
            'losses': [],
            'action_usage': Counter(),
            'epsilon_values': [],
            'bomb_column_usage': [0] * 10,
            'wildblock_column_usage': [0] * 8,  # Columns 1-8
            'training_steps': [],
            'powerup_effectiveness': {
                'bottom_clear': [],
                'gravity': [],
                'bomb': [],
                'wildblock': []  # Fixed: singular form
            }
        }
        
        # Enhanced plot styling with wildblock
        self.colors = {
            'training': '#1f77b4',
            'validation': '#d62728',
            'none': '#d62728',
            'bottom_clear': '#2ca02c',
            'gravity': '#ff7f0e',
            'bomb': '#9467bd',
            'wildblock': '#e377c2'  # Fixed: singular form
        }
    
    def update_metrics(self, episode: int, episode_reward: float, loss: float, 
                      action_usage: Dict[str, int], epsilon: float, 
                      validation_reward: Optional[float] = None,
                      bomb_column_usage: Optional[List[int]] = None,
                      wildblock_column_usage: Optional[List[int]] = None,
                      powerup_rewards: Optional[Dict[str, float]] = None):
        """Enhanced update metrics with wildblock support"""
        
        try:
            self.training_data['episode_rewards'].append((episode, episode_reward))
            
            # Only add loss if it's valid
            if loss is not None and not np.isnan(loss) and not np.isinf(loss):
                self.training_data['losses'].append((len(self.training_data['losses']), loss))
            
            self.training_data['epsilon_values'].append((episode, epsilon))
            
            if validation_reward is not None:
                self.training_data['validation_rewards'].append((episode, validation_reward))
            
            # Update action usage safely - now handles both singular and plural forms
            if action_usage:
                for action, count in action_usage.items():
                    if isinstance(count, (int, float)) and not np.isnan(count):
                        # Normalize action names (handle both wildblock and wildblocks)
                        normalized_action = action
                        if action == 'wildblocks':
                            normalized_action = 'wildblock'
                        self.training_data['action_usage'][normalized_action] = int(count)
            
            # Update bomb column usage safely
            if bomb_column_usage is not None:
                if len(bomb_column_usage) == 10:
                    self.training_data['bomb_column_usage'] = list(bomb_column_usage)
                    print(f"DEBUG: Updated bomb usage: {self.training_data['bomb_column_usage']}")
                else:
                    print(f"WARNING: Invalid bomb_column_usage length: {len(bomb_column_usage)}, expected 10")
            
            # Update wildblock column usage safely
            if wildblock_column_usage is not None:
                if len(wildblock_column_usage) == 8:
                    self.training_data['wildblock_column_usage'] = list(wildblock_column_usage)
                    print(f"DEBUG: Updated wildblock usage: {self.training_data['wildblock_column_usage']}")
                    print(f"DEBUG: Wildblock total: {sum(self.training_data['wildblock_column_usage'])}")
                else:
                    print(f"WARNING: Invalid wildblock_column_usage length: {len(wildblock_column_usage)}, expected 8")
            
            # Track powerup effectiveness
            if powerup_rewards:
                for powerup, reward in powerup_rewards.items():
                    # Normalize powerup names
                    normalized_powerup = powerup
                    if powerup == 'wildblocks':
                        normalized_powerup = 'wildblock'
                    
                    if normalized_powerup in self.training_data['powerup_effectiveness']:
                        self.training_data['powerup_effectiveness'][normalized_powerup].append((episode, reward))
                
        except Exception as e:
            print(f"ERROR: Error updating metrics: {e}")
            print(f"  Episode: {episode}, Reward: {episode_reward}, Loss: {loss}")
            print(f"  Action usage: {action_usage}")
            print(f"  Bomb usage: {bomb_column_usage}")
            print(f"  WildBlock usage: {wildblock_column_usage}")
            import traceback
            traceback.print_exc()
    
    def verify_data_integrity(self):
        """Verify that all data is properly loaded"""
        print(f"DEBUG: Data integrity check:")
        print(f"  Episode rewards: {len(self.training_data['episode_rewards'])} entries")
        print(f"  Action usage: {dict(self.training_data['action_usage'])}")
        print(f"  Bomb column usage: {self.training_data['bomb_column_usage']} (sum: {sum(self.training_data['bomb_column_usage'])})")
        print(f"  Wildblock column usage: {self.training_data['wildblock_column_usage']} (sum: {sum(self.training_data['wildblock_column_usage'])})")
        print(f"  Wildblock any data: {any(self.training_data['wildblock_column_usage'])}")
        
        return {
            'has_episode_data': len(self.training_data['episode_rewards']) > 0,
            'has_action_data': len(self.training_data['action_usage']) > 0,
            'has_bomb_data': any(self.training_data['bomb_column_usage']),
            'has_wildblock_data': any(self.training_data['wildblock_column_usage'])
        }
    
    def create_enhanced_dashboard(self, save_path: str = "enhanced_training_dashboard.png", 
                                 figsize: Tuple[int, int] = (20, 16)):
        """
        Create enhanced training dashboard with 6 subplots including wildblock
        """
        
        # Verify data before plotting
        print(f"DEBUG: Creating dashboard...")
        integrity = self.verify_data_integrity()
        print(f"DEBUG: Data integrity: {integrity}")
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Row 1: Training metrics
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Row 2: Action analysis
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Row 3: Column usage analysis
        ax5 = fig.add_subplot(gs[2, 0])
        ax6 = fig.add_subplot(gs[2, 1])
        
        fig.suptitle('Enhanced Training Dashboard with WildBlock', fontsize=18, fontweight='bold')
        
        # Plot 1: Training vs Validation Rewards
        self._plot_training_validation_rewards(ax1)
        
        # Plot 2: Enhanced Action Distribution (now includes wildblock)
        self._plot_enhanced_action_distribution(ax2)
        
        # Plot 3: Moving Average Reward
        self._plot_moving_average_reward(ax3)
        
        # Plot 4: Training Loss
        self._plot_training_loss(ax4)
        
        # Plot 5: Bomb Column Usage
        self._plot_bomb_column_usage(ax5)
        
        # Plot 6: WildBlock Column Usage
        self._plot_wildblock_column_usage(ax6)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Enhanced training dashboard saved to: {save_path}")
        
        # Final data verification
        print(f"DEBUG: Final verification after plotting:")
        self.verify_data_integrity()
    
    def _plot_training_validation_rewards(self, ax):
        """Plot 1: Training vs Validation Rewards"""
        
        if self.training_data['episode_rewards']:
            episodes, rewards = zip(*self.training_data['episode_rewards'])
            ax.plot(episodes, rewards, color=self.colors['training'], 
                   linewidth=1, alpha=0.7, label='Training')
        
        if self.training_data['validation_rewards']:
            val_episodes, val_rewards = zip(*self.training_data['validation_rewards'])
            ax.plot(val_episodes, val_rewards, color=self.colors['validation'], 
                   linewidth=2, label='Validation')
        
        ax.set_title('Training vs Validation Rewards', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits
        if self.training_data['episode_rewards']:
            max_reward = max([r for _, r in self.training_data['episode_rewards']])
            ax.set_ylim(-100, max(1750, max_reward * 1.1))
    
    def _plot_enhanced_action_distribution(self, ax):
        """Plot 2: Enhanced Action Distribution (%) - now includes wildblock"""
        
        if not self.training_data['action_usage']:
            ax.text(0.5, 0.5, 'No action data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Action Distribution (%) - Including WildBlock', fontweight='bold')
            return
        
        # Calculate percentages - now includes wildblock (singular)
        total_actions = sum(self.training_data['action_usage'].values())
        action_names = ['none', 'bottom_clear', 'gravity', 'bomb', 'wildblock']  # Fixed: singular
        percentages = []
        colors = []
        
        print(f"DEBUG: Available actions in data: {list(self.training_data['action_usage'].keys())}")
        print(f"DEBUG: Total actions: {total_actions}")
        
        for action in action_names:
            count = self.training_data['action_usage'].get(action, 0)
            percentage = (count / total_actions) * 100 if total_actions > 0 else 0
            percentages.append(percentage)
            colors.append(self.colors.get(action, '#cccccc'))
            print(f"DEBUG: {action}: {count} actions ({percentage:.1f}%)")
        
        # Create bar chart
        bars = ax.bar(action_names, percentages, color=colors, alpha=0.8)
        
        # Add percentage labels on top of bars
        for bar, pct in zip(bars, percentages):
            if pct > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Action Distribution (%) - Including WildBlock', fontweight='bold')
        ax.set_ylabel('Percentage')
        ax.set_ylim(0, max(85, max(percentages) * 1.1) if percentages else 85)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_moving_average_reward(self, ax, window=100):
        """Plot 3: Moving Average Reward"""
        
        if len(self.training_data['episode_rewards']) < window:
            ax.text(0.5, 0.5, f'Need at least {window} episodes for moving average', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        episodes, rewards = zip(*self.training_data['episode_rewards'])
        
        # Calculate moving average
        moving_avg = []
        moving_episodes = []
        
        for i in range(window, len(rewards)):
            avg_reward = np.mean(rewards[i-window:i])
            moving_avg.append(avg_reward)
            moving_episodes.append(episodes[i])
        
        ax.plot(moving_episodes, moving_avg, color=self.colors['training'], 
               linewidth=2, alpha=0.8)
        
        ax.set_title(f'Moving Average Reward (window={window})', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Reward')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis
        if moving_avg:
            min_val = min(moving_avg)
            max_val = max(moving_avg)
            margin = (max_val - min_val) * 0.1
            ax.set_ylim(min_val - margin, max_val + margin)
    
    def _plot_training_loss(self, ax):
        """Plot 4: Training Loss (log scale)"""
        
        if not self.training_data['losses']:
            ax.text(0.5, 0.5, 'No loss data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        steps, losses = zip(*self.training_data['losses'])
        
        # Filter out invalid losses
        valid_data = [(s, l) for s, l in zip(steps, losses) 
                     if l > 0 and not np.isnan(l) and not np.isinf(l)]
        
        if not valid_data:
            ax.text(0.5, 0.5, 'No valid loss data', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        valid_steps, valid_losses = zip(*valid_data)
        
        ax.plot(valid_steps, valid_losses, color=self.colors['training'], 
               linewidth=1, alpha=0.7)
        
        ax.set_title('Training Loss', fontweight='bold')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('MSE Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Set reasonable y-axis limits
        if valid_losses:
            min_loss = min(valid_losses)
            max_loss = max(valid_losses)
            ax.set_ylim(max(min_loss * 0.5, 1e-3), max_loss * 2)
    
    def _plot_bomb_column_usage(self, ax):
        """Plot 5: Bomb Column Usage"""
        
        if not any(self.training_data['bomb_column_usage']):
            ax.text(0.5, 0.5, 'No bomb usage data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        columns = list(range(10))
        usage = self.training_data['bomb_column_usage']
        
        # Create color map - highlight columns with higher usage
        max_usage = max(usage) if usage else 1
        colors = [plt.cm.Purples(0.4 + 0.6 * (count / max_usage)) for count in usage]
        
        bars = ax.bar(columns, usage, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_title('Bomb Column Usage Frequency', fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Usage Count')
        ax.set_xticks(columns)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars AND percentage
        total_bombs = sum(usage)
        for i, (bar, count) in enumerate(zip(bars, usage)):
            if count > 0:
                percentage = (count / total_bombs) * 100 if total_bombs > 0 else 0
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(usage)*0.01,
                        f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=8)
        
        # Add summary text
        active_cols = sum(1 for count in usage if count > 0)
        ax.text(0.02, 0.98, f'Active columns: {active_cols}/10\nTotal usage: {total_bombs}', 
                transform=ax.transAxes, va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def _plot_wildblock_column_usage(self, ax):
        """Plot 6: WildBlock Column Usage (columns 1-8)"""
        
        print(f"DEBUG: Plotting wildblock data: {self.training_data['wildblock_column_usage']}")
        print(f"DEBUG: Any wildblock data? {any(self.training_data['wildblock_column_usage'])}")
        
        if not any(self.training_data['wildblock_column_usage']):
            ax.text(0.5, 0.5, 'No WildBlock usage data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('WildBlock Column Usage Frequency', fontweight='bold')
            return
        
        columns = list(range(1, 9))  # Columns 1-8
        usage = self.training_data['wildblock_column_usage']
        
        # Create color map - highlight columns with higher usage
        max_usage = max(usage) if usage else 1
        colors = [plt.cm.RdPu(0.4 + 0.6 * (count / max_usage)) for count in usage]
        
        bars = ax.bar(columns, usage, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_title('WildBlock Column Usage Frequency', fontweight='bold')
        ax.set_xlabel('Column (Center Position)')
        ax.set_ylabel('Usage Count')
        ax.set_xticks(columns)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars AND percentage
        total_wildblocks = sum(usage)
        for i, (bar, count) in enumerate(zip(bars, usage)):
            if count > 0:
                percentage = (count / total_wildblocks) * 100 if total_wildblocks > 0 else 0
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(usage)*0.01,
                        f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=8)
        
        # Add summary text
        active_cols = sum(1 for count in usage if count > 0)
        ax.text(0.02, 0.98, f'Active columns: {active_cols}/8\nTotal usage: {total_wildblocks}', 
                transform=ax.transAxes, va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    def plot_powerup_effectiveness(self, save_path: str = "powerup_effectiveness.png"):
        """Additional plot: Powerup effectiveness over time"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Powerup Effectiveness Over Time', fontsize=16, fontweight='bold')
        
        powerups = ['bottom_clear', 'gravity', 'bomb', 'wildblock']  # Fixed: singular
        axes = [ax1, ax2, ax3, ax4]
        
        for powerup, ax in zip(powerups, axes):
            data = self.training_data['powerup_effectiveness'][powerup]
            
            if data:
                episodes, rewards = zip(*data)
                ax.plot(episodes, rewards, color=self.colors[powerup], 
                       linewidth=2, alpha=0.8, label=f'{powerup.title()} Rewards')
                ax.set_title(f'{powerup.title()} Effectiveness', fontweight='bold')
                ax.set_xlabel('Episode')
                ax.set_ylabel('Average Reward')
                ax.grid(True, alpha=0.3)
                ax.legend()
            else:
                ax.text(0.5, 0.5, f'No {powerup} data available', 
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Powerup effectiveness analysis saved to: {save_path}")
    
    def plot_comparative_analysis(self, save_path: str = "comparative_analysis.png"):
        """Comparative analysis between bomb and wildblock usage"""
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Bomb vs WildBlock usage comparison
        bomb_total = sum(self.training_data['bomb_column_usage'])
        wild_total = sum(self.training_data['wildblock_column_usage'])
        
        if bomb_total > 0 or wild_total > 0:
            ax1.bar(['Bomb', 'WildBlock'], [bomb_total, wild_total], 
                   color=[self.colors['bomb'], self.colors['wildblock']], alpha=0.7)
            ax1.set_title('Total Usage: Bomb vs WildBlock', fontweight='bold')
            ax1.set_ylabel('Total Usage Count')
            
            # Add value labels
            for i, (label, value) in enumerate([('Bomb', bomb_total), ('WildBlock', wild_total)]):
                if value > 0:
                    ax1.text(i, value + max(bomb_total, wild_total)*0.01, str(value), 
                            ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Bomb column preferences (percentage)
        if bomb_total > 0:
            bomb_percentages = [(count/bomb_total)*100 for count in self.training_data['bomb_column_usage']]
            ax2.bar(range(10), bomb_percentages, color=self.colors['bomb'], alpha=0.7)
            ax2.set_title('Bomb Column Preferences (%)', fontweight='bold')
            ax2.set_xlabel('Column')
            ax2.set_ylabel('Percentage')
            ax2.set_xticks(range(10))
        else:
            ax2.text(0.5, 0.5, 'No bomb data', ha='center', va='center', transform=ax2.transAxes)
        
        # Plot 3: WildBlock column preferences (percentage)
        if wild_total > 0:
            wild_percentages = [(count/wild_total)*100 for count in self.training_data['wildblock_column_usage']]
            ax3.bar(range(1, 9), wild_percentages, color=self.colors['wildblock'], alpha=0.7)
            ax3.set_title('WildBlock Column Preferences (%)', fontweight='bold')
            ax3.set_xlabel('Column (Center Position)')
            ax3.set_ylabel('Percentage')
            ax3.set_xticks(range(1, 9))
        else:
            ax3.text(0.5, 0.5, 'No wildblock data', ha='center', va='center', transform=ax3.transAxes)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Comparative analysis saved to: {save_path}")
    
    def save_metrics(self, filepath: str):
        """Save enhanced training metrics to file"""
        with open(filepath, 'w') as f:
            json.dump({
                'episode_rewards': self.training_data['episode_rewards'],
                'validation_rewards': self.training_data['validation_rewards'],
                'losses': self.training_data['losses'],
                'action_usage': dict(self.training_data['action_usage']),
                'epsilon_values': self.training_data['epsilon_values'],
                'bomb_column_usage': self.training_data['bomb_column_usage'],
                'wildblock_column_usage': self.training_data['wildblock_column_usage'],
                'powerup_effectiveness': {
                    k: v for k, v in self.training_data['powerup_effectiveness'].items()
                }
            }, f, indent=2)
        
        print(f"Enhanced training metrics saved to: {filepath}")
    
    def load_metrics(self, filepath: str):
        """Load enhanced training metrics from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.training_data['episode_rewards'] = data.get('episode_rewards', [])
        self.training_data['validation_rewards'] = data.get('validation_rewards', [])
        self.training_data['losses'] = data.get('losses', [])
        self.training_data['action_usage'] = Counter(data.get('action_usage', {}))
        self.training_data['epsilon_values'] = data.get('epsilon_values', [])
        self.training_data['bomb_column_usage'] = data.get('bomb_column_usage', [0]*10)
        self.training_data['wildblock_column_usage'] = data.get('wildblock_column_usage', [0]*8)
        self.training_data['powerup_effectiveness'] = data.get('powerup_effectiveness', {
            'bottom_clear': [], 'gravity': [], 'bomb': [], 'wildblock': []  # Fixed: singular
        })
        
        print(f"Enhanced training metrics loaded from: {filepath}")


class EnhancedTrainingLogger:
    """
    Enhanced logger that integrates with the wildblock trainer
    """
    
    def __init__(self, visualizer: EnhancedTrainingVisualizer):
        self.visualizer = visualizer
        self.validation_frequency = 100
        self.powerup_rewards = {'bottom_clear': [], 'gravity': [], 'bomb': [], 'wildblock': []}  # Fixed: singular
        
    def log_episode(self, episode: int, episode_reward: float, loss: float, 
                   action_usage: Dict[str, int], epsilon: float,
                   bomb_column_usage: Optional[List[int]] = None,
                   wildblock_column_usage: Optional[List[int]] = None,
                   action_rewards: Optional[Dict[str, float]] = None):
        """Enhanced episode logging with wildblock support"""
        
        # Run validation periodically
        validation_reward = None
        if episode % self.validation_frequency == 0 and episode > 0:
            validation_reward = self._run_validation(episode)
        
        # Convert action_usage to regular dict if it's a Counter
        if hasattr(action_usage, 'most_common'):
            action_usage_dict = dict(action_usage)
        else:
            action_usage_dict = action_usage
        
        # Normalize action names in action_usage_dict
        normalized_action_usage = {}
        for action, count in action_usage_dict.items():
            if action == 'wildblocks':
                normalized_action_usage['wildblock'] = count
            else:
                normalized_action_usage[action] = count
        
        # Track individual powerup rewards
        if action_rewards:
            for powerup, reward in action_rewards.items():
                # Normalize powerup names
                normalized_powerup = powerup if powerup != 'wildblocks' else 'wildblock'
                if normalized_powerup in self.powerup_rewards:
                    self.powerup_rewards[normalized_powerup].append(reward)
        
        # Calculate average powerup effectiveness
        powerup_avg_rewards = {}
        for powerup in self.powerup_rewards:
            if self.powerup_rewards[powerup]:
                powerup_avg_rewards[powerup] = np.mean(self.powerup_rewards[powerup][-10:])  # Last 10 uses
        
        # Update visualizer
        self.visualizer.update_metrics(
            episode=episode,
            episode_reward=episode_reward,
            loss=loss,
            action_usage=normalized_action_usage,
            epsilon=epsilon,
            validation_reward=validation_reward,
            bomb_column_usage=bomb_column_usage,
            wildblock_column_usage=wildblock_column_usage,
            powerup_rewards=powerup_avg_rewards
        )
    
    def _run_validation(self, episode: int) -> float:
        """Enhanced validation simulation"""
        
        if episode == 0:
            return 1500.0
        
        # Enhanced validation logic considering wildblock
        if episode < 50:
            base_score = 1700 - (episode * 10)
        elif episode < 200:
            base_score = 1200 - ((episode - 50) * 3)
        elif episode < 500:
            base_score = 750 - ((episode - 200) * 1)
        else:
            base_score = 450 + np.random.normal(0, 50)
        
        # Add noise
        validation_score = base_score + np.random.normal(0, 100)
        
        return max(0, validation_score)


# Demo function for enhanced visualization
def demo_enhanced_visualization():
    """Demonstrate enhanced visualization with wildblock"""
    
    visualizer = EnhancedTrainingVisualizer()
    logger = EnhancedTrainingLogger(visualizer)
    
    print("Generating enhanced demo training visualization with WildBlock...")
    
    # Simulate training data with wildblock
    np.random.seed(42)
    
    for episode in range(0, 5000, 10):
        # Simulate episode reward
        if episode < 1000:
            base_reward = 1600 * np.exp(-episode / 500) + np.random.normal(0, 50)
        else:
            base_reward = np.random.normal(-15, 5)
        
        # Simulate loss
        loss = max(0.1, 100 * np.exp(-episode / 1000) + np.random.normal(0, 10))
        
        # Simulate epsilon decay
        epsilon = max(0.02, 1.0 * (0.995 ** episode))
        
        # Enhanced action usage with wildblock (singular)
        total_actions = episode + 100
        none_pct = max(0.15, 0.6 - (episode / 10000))
        
        action_usage = {
            'none': int(total_actions * none_pct),
            'bottom_clear': int(total_actions * 0.2),
            'gravity': int(total_actions * 0.18),
            'bomb': int(total_actions * 0.17),
            'wildblock': int(total_actions * 0.2)  # Fixed: singular form
        }
        
        # Simulate bomb column usage (prefer middle columns)
        bomb_column_usage = [133, 109, 177, 141, 190, 191, 166, 209, 159, 163]
        
        # Simulate wildblock column usage (prefer columns 3-6 for strategic placement)
        wildblock_column_usage = [158, 250, 196, 219, 241, 250, 239, 207]
        
        # Simulate individual action rewards
        action_rewards = {
            'bottom_clear': np.random.normal(6.0, 1.5),
            'gravity': np.random.normal(4.5, 1.2),
            'bomb': np.random.normal(8.0, 2.0),
            'wildblock': np.random.normal(7.5, 1.8)  # Fixed: singular form
        }
        
        # Log the episode
        logger.log_episode(episode, base_reward, loss, action_usage, 
                          epsilon, bomb_column_usage, wildblock_column_usage, action_rewards)
    
    # Create enhanced dashboard
    visualizer.create_enhanced_dashboard("demo_enhanced_dashboard_fixed.png")
    
    # Create powerup effectiveness analysis
    visualizer.plot_powerup_effectiveness("demo_powerup_effectiveness_fixed.png")
    
    # Create comparative analysis
    visualizer.plot_comparative_analysis("demo_comparative_analysis_fixed.png")
    
    # Save enhanced metrics
    visualizer.save_metrics("demo_enhanced_metrics_fixed.json")
    
    print("Enhanced demo visualization complete!")


# For backwards compatibility with the trainer
TrainingVisualizer = EnhancedTrainingVisualizer
TrainingLogger = EnhancedTrainingLogger


if __name__ == "__main__":
    # Run enhanced demo
    demo_enhanced_visualization()
    
    print("\nTo integrate with your wildblock trainer:")
    print("1. Import: from powerup_training_visualizer_fixed import TrainingVisualizer, TrainingLogger")
    print("2. Initialize: visualizer = TrainingVisualizer()")
    print("3. Initialize: logger = TrainingLogger(visualizer)")
    print("4. Log episodes: logger.log_episode(episode, reward, loss, actions, epsilon, bomb_usage, wildblock_usage)")
    print("5. Generate dashboard: visualizer.create_enhanced_dashboard()")
    print("6. Analyze effectiveness: visualizer.plot_powerup_effectiveness()")