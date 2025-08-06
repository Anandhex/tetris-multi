import numpy as np
import pickle
import random

class TetrisDatasetGenerator:
    """Generate realistic Tetris board configurations for training with wildblock support"""
    
    def __init__(self, board_height=20, board_width=10):
        self.height = board_height
        self.width = board_width
        self.all_powerups = ['bottom_clear', 'gravity', 'bomb', 'wildblock']
    
    def generate_random_board(self, fill_ratio: float = 0.3) -> np.ndarray:
        """Generate a random but realistic board configuration"""
        board = np.zeros((self.height, self.width), dtype=np.int32)
        
        # Start from bottom and work up
        for row in range(self.height - 1, -1, -1):
            # Probability of placing blocks decreases with height
            height_factor = (self.height - row) / self.height
            adjusted_ratio = fill_ratio * height_factor
            
            for col in range(self.width):
                if random.random() < adjusted_ratio:
                    board[row, col] = 1
        
        return board
    
    def generate_structured_board(self) -> np.ndarray:
        """Generate a more structured board with realistic patterns"""
        board = np.zeros((self.height, self.width), dtype=np.int32)
        
        # Create some filled layers at bottom
        bottom_layers = random.randint(0, 3)
        for layer in range(bottom_layers):
            row = self.height - 1 - layer
            # Leave some gaps
            for col in range(self.width):
                if random.random() < 0.8:  # 80% chance of block
                    board[row, col] = 1
        
        # Add some scattered blocks above
        for row in range(self.height - bottom_layers - 1, max(0, self.height - 10), -1):
            blocks_in_row = random.randint(0, 6)
            positions = random.sample(range(self.width), min(blocks_in_row, self.width))
            for col in positions:
                board[row, col] = 1
        
        return board
    
    def create_problem_board(self) -> np.ndarray:
        """Create boards with specific problems (holes, high stacks, etc.)"""
        board = np.zeros((self.height, self.width), dtype=np.int32)
        problem_type = random.choice(['holes', 'high_stack', 'uneven', 'wildblock_target'])
        
        if problem_type == 'holes':
            # Create board with many holes
            for row in range(self.height - 5, self.height):
                for col in range(self.width):
                    if random.random() < 0.7:
                        board[row, col] = 1
            # Create holes by removing some blocks
            for _ in range(random.randint(3, 8)):
                row = random.randint(self.height - 5, self.height - 1)
                col = random.randint(0, self.width - 1)
                board[row, col] = 0
        
        elif problem_type == 'high_stack':
            # Create very uneven heights
            for col in range(self.width):
                height = random.randint(5, 15)
                for row in range(self.height - height, self.height):
                    board[row, col] = 1
        
        elif problem_type == 'uneven':
            # Create very bumpy surface
            heights = [random.randint(0, 12) for _ in range(self.width)]
            for col, height in enumerate(heights):
                for row in range(self.height - height, self.height):
                    if random.random() < 0.9:
                        board[row, col] = 1
        
        elif problem_type == 'wildblock_target':
            # Create board that's good target for wildblock (moderate height, some gaps)
            for col in range(self.width):
                if random.random() < 0.8:  # 80% chance this column has blocks
                    height = random.randint(8, 14)
                    for row in range(self.height - height, self.height):
                        if random.random() < 0.85:  # Leave some internal gaps
                            board[row, col] = 1
        
        return board
    
    def generate_powerup_combination(self) -> dict:
        """Generate a balanced powerup combination ensuring even distribution"""
        
        # Define powerup scenarios with different numbers of available powerups
        scenario_types = [
            'single',    # 1 powerup
            'double',    # 2 powerups  
            'triple',    # 3 powerups
            'all'        # all 4 powerups
        ]
        
        # Weight scenarios to ensure good distribution
        scenario_weights = [0.25, 0.35, 0.25, 0.15]  # Favor 2-3 powerups
        scenario = random.choices(scenario_types, weights=scenario_weights)[0]
        
        if scenario == 'single':
            # Exactly one powerup available (evenly distributed)
            available_powerup = random.choice(self.all_powerups)
            powerups = {powerup: powerup == available_powerup for powerup in self.all_powerups}
            
        elif scenario == 'double':
            # Exactly two powerups available
            available_powerups = random.sample(self.all_powerups, 2)
            powerups = {powerup: powerup in available_powerups for powerup in self.all_powerups}
            
        elif scenario == 'triple':
            # Exactly three powerups available
            available_powerups = random.sample(self.all_powerups, 3)
            powerups = {powerup: powerup in available_powerups for powerup in self.all_powerups}
            
        else:  # 'all'
            # All four powerups available
            powerups = {powerup: True for powerup in self.all_powerups}
        
        return powerups, scenario
    
    def generate_dataset(self, num_samples: int, save_path: str):
        """Generate and save a complete dataset with guaranteed powerup availability and even distribution"""
        dataset = []
        
        print(f"Generating {num_samples} board configurations with wildblock support...")
        
        # Track statistics for even distribution
        powerup_counts = {powerup: 0 for powerup in self.all_powerups}
        scenario_counts = {'single': 0, 'double': 0, 'triple': 0, 'all': 0}
        board_type_counts = {'random': 0, 'structured': 0, 'problem': 0}
        
        for i in range(num_samples):
            # Generate board with even distribution of types
            board_type = random.choice(['random', 'structured', 'problem'])
            board_type_counts[board_type] += 1
            
            if board_type == 'random':
                board = self.generate_random_board()
            elif board_type == 'structured':
                board = self.generate_structured_board()
            else:
                board = self.create_problem_board()
            
            # Generate powerup combination with balanced distribution
            powerups, scenario = self.generate_powerup_combination()
            scenario_counts[scenario] += 1
            
            # Count powerup availability for statistics
            for powerup, available in powerups.items():
                if available:
                    powerup_counts[powerup] += 1
            
            # SAFETY CHECK: Ensure at least one powerup is True
            if not any(powerups.values()):
                # This should never happen with above logic, but just in case
                forced_powerup = random.choice(self.all_powerups)
                powerups[forced_powerup] = True
                powerup_counts[forced_powerup] += 1
                print(f"  SAFETY: Forced {forced_powerup} to be available")
            
            dataset.append({
                'board': board.tolist(),
                'powerups': powerups,
                'board_type': board_type,
                'powerup_scenario': scenario
            })
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_samples} boards")
        
        # VERIFY DATASET DISTRIBUTION
        none_scenarios = sum(1 for sample in dataset if not any(sample['powerups'].values()))
        
        print(f"\n📊 DATASET STATISTICS:")
        print(f"  Total samples: {num_samples}")
        print(f"  Scenarios with NO powerups: {none_scenarios} (should be 0)")
        
        print(f"\n  📈 Powerup Availability (Target: ~{num_samples/2:.0f} each for even distribution):")
        for powerup, count in powerup_counts.items():
            percentage = (count / num_samples) * 100
            print(f"    {powerup:12}: {count:4d} ({percentage:5.1f}%)")
        
        print(f"\n  📋 Scenario Distribution:")
        for scenario, count in scenario_counts.items():
            percentage = (count / num_samples) * 100
            print(f"    {scenario:8}: {count:4d} ({percentage:5.1f}%)")
        
        print(f"\n  🎯 Board Type Distribution:")
        for board_type, count in board_type_counts.items():
            percentage = (count / num_samples) * 100
            print(f"    {board_type:10}: {count:4d} ({percentage:5.1f}%)")
        
        # Calculate distribution balance
        powerup_percentages = [(count / num_samples) * 100 for count in powerup_counts.values()]
        balance_score = 100 - (max(powerup_percentages) - min(powerup_percentages))
        print(f"\n  ⚖️  Powerup Balance Score: {balance_score:.1f}% (higher is more balanced)")
        
        # Additional statistics for wildblock-specific scenarios
        wildblock_available = sum(1 for sample in dataset if sample['powerups']['wildblock'])
        wildblock_only = sum(1 for sample in dataset 
                            if sample['powerups']['wildblock'] and 
                            sum(sample['powerups'].values()) == 1)
        
        print(f"\n  🔥 Wildblock-Specific Stats:")
        print(f"    Wildblock available: {wildblock_available} ({(wildblock_available/num_samples)*100:.1f}%)")
        print(f"    Wildblock only: {wildblock_only} ({(wildblock_only/num_samples)*100:.1f}%)")
        
        # Save dataset
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"\n💾 Dataset saved to {save_path}")
        print("✅ GUARANTEED: Every scenario has at least 1 powerup available")
        print("✅ BALANCED: All powerups have even distribution opportunities")
        
        return dataset

    def load_and_analyze_dataset(self, file_path: str):
        """Load and analyze an existing dataset"""
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
        
        print(f"📁 Analyzing dataset: {file_path}")
        print(f"  Total samples: {len(dataset)}")
        
        # Analyze powerup distribution
        powerup_counts = {powerup: 0 for powerup in self.all_powerups}
        scenario_counts = {}
        
        for sample in dataset:
            for powerup, available in sample['powerups'].items():
                if available and powerup in powerup_counts:
                    powerup_counts[powerup] += 1
            
            scenario = sample.get('powerup_scenario', 'unknown')
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        
        print(f"\n  Powerup Distribution:")
        for powerup, count in powerup_counts.items():
            percentage = (count / len(dataset)) * 100
            print(f"    {powerup:12}: {count:4d} ({percentage:5.1f}%)")
        
        print(f"\n  Scenario Distribution:")
        for scenario, count in scenario_counts.items():
            percentage = (count / len(dataset)) * 100
            print(f"    {scenario:8}: {count:4d} ({percentage:5.1f}%)")
        
        return dataset


# Usage example and testing
if __name__ == "__main__":
    generator = TetrisDatasetGenerator()
    
    # Generate a balanced dataset with wildblock support
    print("🚀 Generating Tetris dataset with wildblock support...")
    dataset = generator.generate_dataset(20000, "tetris_boards2.pkl")
    
    # Test loading and analysis
    print("\n" + "="*50)
    print("🔍 Testing dataset loading and analysis...")
    loaded_dataset = generator.load_and_analyze_dataset("tetris_boards2.pkl")
    
    # Show some sample configurations
    print("\n" + "="*50)
    print("📋 Sample configurations:")
    for i in range(3):
        sample = dataset[i]
        available_powerups = [p for p, avail in sample['powerups'].items() if avail]
        print(f"  Sample {i+1}: {sample['board_type']} board, powerups: {available_powerups}")
        
    print("\n✅ Dataset generation with wildblock support complete!")