import numpy as np
import pickle
import random

class TetrisDatasetGenerator:
    """Generate realistic Tetris board configurations with proper gravity and connectivity"""
    
    def __init__(self, board_height=20, board_width=10):
        self.height = board_height
        self.width = board_width
        self.all_powerups = ['bottom_clear', 'gravity', 'bomb', 'wildblock']
    
    def apply_gravity(self, board: np.ndarray) -> np.ndarray:
        """Apply gravity - all blocks fall to the bottom with no floating blocks"""
        result = np.zeros_like(board, dtype=np.int32)
        
        for col in range(self.width):
            # Count blocks in this column
            block_count = np.sum(board[:, col])
            
            # Place all blocks at the bottom of the column
            for i in range(int(block_count)):
                result[self.height - 1 - i, col] = 1
        
        return result
    
    def generate_realistic_board(self, target_density: float = 0.3) -> np.ndarray:
        """Generate a realistic board by building from bottom up"""
        board = np.zeros((self.height, self.width), dtype=np.int32)
        
        # Start from the bottom and work up
        for row in range(self.height - 1, -1, -1):
            # Higher rows have lower probability of blocks
            row_height = self.height - row
            height_factor = max(0, 1 - (row_height / self.height) * 1.5)
            row_probability = target_density * height_factor
            
            if row_probability <= 0:
                break
            
            # Decide how many blocks in this row
            max_blocks = max(1, int(self.width * row_probability))
            num_blocks = random.randint(0, max_blocks)
            
            if num_blocks > 0:
                # Randomly place blocks, but ensure they're somewhat clustered
                positions = random.sample(range(self.width), min(num_blocks, self.width))
                for col in positions:
                    board[row, col] = 1
        
        # Apply gravity to ensure no floating blocks
        return self.apply_gravity(board)
    
    def generate_layered_board(self) -> np.ndarray:
        """Generate board with realistic layers (like cleared lines with gaps)"""
        board = np.zeros((self.height, self.width), dtype=np.int32)
        
        # Create several layers with different fill ratios
        current_row = self.height - 1
        
        while current_row > self.height - 15 and current_row >= 0:
            # Decide layer thickness and density
            layer_thickness = random.randint(1, 4)
            layer_density = random.uniform(0.4, 0.9)
            
            # Fill this layer
            for layer_row in range(max(0, current_row - layer_thickness + 1), current_row + 1):
                if layer_row >= 0:
                    num_blocks = int(self.width * layer_density)
                    if num_blocks > 0:
                        positions = random.sample(range(self.width), num_blocks)
                        for col in positions:
                            board[layer_row, col] = 1
            
            # Leave some empty space before next layer
            current_row -= layer_thickness + random.randint(0, 2)
        
        # Apply gravity to make it realistic
        return self.apply_gravity(board)
    
    def generate_uneven_board(self) -> np.ndarray:
        """Generate board with uneven column heights"""
        board = np.zeros((self.height, self.width), dtype=np.int32)
        
        # Generate different heights for each column
        for col in range(self.width):
            # Random height with tendency for middle columns to be higher
            base_height = random.randint(0, 12)
            
            # Add some variation
            if 2 <= col <= 7:  # Middle columns
                base_height += random.randint(0, 4)
            
            # Fill column from bottom up with some gaps
            blocks_to_place = max(0, base_height)
            
            for i in range(blocks_to_place):
                row = self.height - 1 - i
                if row >= 0:
                    # 90% chance of placing block (creates some internal gaps)
                    if random.random() < 0.9:
                        board[row, col] = 1
        
        # Apply gravity to ensure connectivity
        return self.apply_gravity(board)
    
    def generate_problem_board(self) -> np.ndarray:
        """Generate boards with specific problem patterns and complex scenarios"""
        board_type = random.choice([
            'high_sides', 'pyramid', 'valleys', 'sparse', 'deep_holes', 
            'overhangs', 'narrow_wells', 'checkerboard', 'stairs', 'critical_height'
        ])
        board = np.zeros((self.height, self.width), dtype=np.int32)
        
        if board_type == 'high_sides':
            # High columns on sides, low in middle
            for col in range(self.width):
                if col <= 2 or col >= 7:  # Side columns
                    height = random.randint(10, 16)
                else:  # Middle columns
                    height = random.randint(3, 7)
                
                for i in range(height):
                    row = self.height - 1 - i
                    if row >= 0 and random.random() < 0.9:
                        board[row, col] = 1
        
        elif board_type == 'pyramid':
            # Pyramid shape - higher in middle
            for col in range(self.width):
                distance_from_center = abs(col - 4.5)
                max_height = max(3, 15 - int(distance_from_center * 2.5))
                height = random.randint(max_height // 2, max_height)
                
                for i in range(height):
                    row = self.height - 1 - i
                    if row >= 0 and random.random() < 0.92:
                        board[row, col] = 1
        
        elif board_type == 'valleys':
            # Create multiple valley patterns
            valley_positions = random.sample(range(1, self.width-1), random.randint(2, 3))
            for col in range(self.width):
                if col in valley_positions:  # Valley positions
                    height = random.randint(2, 5)
                else:
                    height = random.randint(8, 14)
                
                for i in range(height):
                    row = self.height - 1 - i
                    if row >= 0 and random.random() < 0.88:
                        board[row, col] = 1
        
        elif board_type == 'deep_holes':
            # Create board with deep internal holes
            # First fill most of the bottom area
            for row in range(self.height - 8, self.height):
                for col in range(self.width):
                    if random.random() < 0.85:
                        board[row, col] = 1
            
            # Create deep holes by removing vertical sections
            num_holes = random.randint(2, 4)
            for _ in range(num_holes):
                hole_col = random.randint(1, self.width - 2)
                hole_depth = random.randint(4, 7)
                hole_start = random.randint(self.height - 8, self.height - hole_depth)
                
                for row in range(hole_start, hole_start + hole_depth):
                    if row < self.height:
                        board[row, hole_col] = 0
        
        elif board_type == 'overhangs':
            # Create overhanging structures
            for col in range(self.width):
                base_height = random.randint(4, 8)
                # Fill base
                for i in range(base_height):
                    row = self.height - 1 - i
                    if row >= 0:
                        board[row, col] = 1
                
                # Add overhangs randomly
                if random.random() < 0.6:
                    overhang_height = random.randint(2, 5)
                    overhang_start = self.height - base_height - overhang_height
                    if overhang_start >= 0:
                        for i in range(overhang_height):
                            row = overhang_start + i
                            if row >= 0 and random.random() < 0.7:
                                board[row, col] = 1
        
        elif board_type == 'narrow_wells':
            # Create narrow wells (single column gaps)
            well_positions = random.sample(range(1, self.width-1), random.randint(2, 3))
            
            # Fill everything first
            for row in range(self.height - 12, self.height):
                for col in range(self.width):
                    if random.random() < 0.88:
                        board[row, col] = 1
            
            # Create narrow wells
            for well_col in well_positions:
                well_depth = random.randint(6, 10)
                well_start = self.height - well_depth
                for row in range(well_start, self.height):
                    if row >= 0:
                        board[row, well_col] = 0
        
        elif board_type == 'checkerboard':
            # Create checkerboard-like pattern with complexity
            base_height = random.randint(6, 10)
            for row in range(self.height - base_height, self.height):
                for col in range(self.width):
                    # Checkerboard with some randomness
                    if (row + col) % 2 == 0:
                        if random.random() < 0.8:
                            board[row, col] = 1
                    else:
                        if random.random() < 0.3:
                            board[row, col] = 1
        
        elif board_type == 'stairs':
            # Create stair-like patterns
            direction = random.choice(['up', 'down'])
            for col in range(self.width):
                if direction == 'up':
                    height = 4 + col
                else:
                    height = 4 + (self.width - 1 - col)
                
                height = min(height, 15)
                for i in range(height):
                    row = self.height - 1 - i
                    if row >= 0 and random.random() < 0.9:
                        board[row, col] = 1
        
        elif board_type == 'critical_height':
            # Dangerous high boards (close to game over)
            for col in range(self.width):
                height = random.randint(16, 19)  # Very high
                for i in range(height):
                    row = self.height - 1 - i
                    if row >= 0 and random.random() < 0.85:
                        board[row, col] = 1
                
                # Leave some strategic gaps
                if random.random() < 0.4:
                    gap_row = random.randint(self.height - height + 2, self.height - 3)
                    board[gap_row, col] = 0
        
        else:  # sparse - keep original
            total_blocks = random.randint(15, 35)
            for _ in range(total_blocks):
                col = random.randint(0, self.width - 1)
                for row in range(self.height - 1, -1, -1):
                    if board[row, col] == 0:
                        board[row, col] = 1
                        break
        
        return self.apply_gravity(board)
    
    def generate_extreme_board(self) -> np.ndarray:
        """Generate extremely challenging board configurations"""
        extreme_type = random.choice([
            'multiple_wells', 'scattered_holes', 'fortress', 'maze_like', 'unstable_tower'
        ])
        board = np.zeros((self.height, self.width), dtype=np.int32)
        
        if extreme_type == 'multiple_wells':
            # Multiple wells of different depths
            well_count = random.randint(3, 5)
            well_positions = random.sample(range(1, self.width-1), well_count)
            
            # Fill base layer
            for row in range(self.height - 10, self.height):
                for col in range(self.width):
                    board[row, col] = 1
            
            # Create wells of varying depths
            for i, well_col in enumerate(well_positions):
                well_depth = random.randint(5, 9)
                for row in range(self.height - well_depth, self.height):
                    board[row, well_col] = 0
        
        elif extreme_type == 'scattered_holes':
            # Heavily filled with many scattered holes
            # Fill most positions
            for row in range(self.height - 14, self.height):
                for col in range(self.width):
                    if random.random() < 0.9:
                        board[row, col] = 1
            
            # Create scattered holes
            hole_count = random.randint(8, 15)
            for _ in range(hole_count):
                hole_col = random.randint(0, self.width - 1)
                hole_row = random.randint(self.height - 12, self.height - 1)
                board[hole_row, hole_col] = 0
        
        elif extreme_type == 'fortress':
            # Fortress-like structure with walls and internal spaces
            wall_positions = [1, 3, 6, 8]
            for col in wall_positions:
                height = random.randint(12, 16)
                for i in range(height):
                    row = self.height - 1 - i
                    if row >= 0:
                        board[row, col] = 1
            
            # Fill some internal areas
            for col in range(self.width):
                if col not in wall_positions:
                    height = random.randint(6, 10)
                    for i in range(height):
                        row = self.height - 1 - i
                        if row >= 0 and random.random() < 0.7:
                            board[row, col] = 1
        
        elif extreme_type == 'maze_like':
            # Complex maze-like structure
            for row in range(self.height - 12, self.height):
                for col in range(self.width):
                    # Create maze pattern
                    if (row % 3 == 0 and col % 2 == 0) or (row % 3 == 2 and col % 2 == 1):
                        if random.random() < 0.8:
                            board[row, col] = 1
        
        elif extreme_type == 'unstable_tower':
            # Tall unstable tower structures
            tower_positions = random.sample(range(1, self.width-1), random.randint(2, 4))
            for tower_col in tower_positions:
                tower_height = random.randint(14, 18)
                for i in range(tower_height):
                    row = self.height - 1 - i
                    if row >= 0:
                        # Make tower slightly unstable with gaps
                        if random.random() < 0.9:
                            board[row, tower_col] = 1
                        
                        # Add some horizontal extensions
                        if random.random() < 0.3 and i > 5:
                            for dc in [-1, 1]:
                                extend_col = tower_col + dc
                                if 0 <= extend_col < self.width:
                                    board[row, extend_col] = 1
        
        return self.apply_gravity(board)
    
    def remove_complete_lines(self, board: np.ndarray) -> np.ndarray:
        """Remove any complete lines to make board more realistic"""
        new_board = []
        lines_removed = 0
        
        for row in range(self.height):
            if np.sum(board[row, :]) < self.width:  # Not a complete line
                new_board.append(board[row, :])
            else:
                lines_removed += 1
        
        # Add empty rows at the top
        while len(new_board) < self.height:
            new_board.insert(0, np.zeros(self.width))
        
        return np.array(new_board)
    
    def generate_powerup_combination(self) -> tuple:
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
        """Generate and save a complete dataset with realistic Tetris physics"""
        dataset = []
        
        print(f"Generating {num_samples} realistic Tetris board configurations...")
        
        # Track statistics for even distribution
        powerup_counts = {powerup: 0 for powerup in self.all_powerups}
        scenario_counts = {'single': 0, 'double': 0, 'triple': 0, 'all': 0}
        board_type_counts = {'realistic': 0, 'layered': 0, 'uneven': 0, 'problem': 0, 'extreme': 0}
        
        for i in range(num_samples):
            # Generate board with even distribution of types, including extreme cases
            board_type = random.choice(['realistic', 'layered', 'uneven', 'problem', 'extreme'])
            board_type_counts[board_type] = board_type_counts.get(board_type, 0) + 1
            
            if board_type == 'realistic':
                board = self.generate_realistic_board()
            elif board_type == 'layered':
                board = self.generate_layered_board()
            elif board_type == 'uneven':
                board = self.generate_uneven_board()
            elif board_type == 'problem':
                board = self.generate_problem_board()
            else:  # extreme
                board = self.generate_extreme_board()
            
            # Remove any complete lines to make it more realistic
            board = self.remove_complete_lines(board)
            
            # Final gravity application to ensure everything is properly settled
            board = self.apply_gravity(board)
            
            # Ensure board is integer type
            board = board.astype(np.int32)
            
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
        
        print(f"\n📊 REALISTIC TETRIS DATASET STATISTICS:")
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
        
        # Physics validation
        floating_blocks = 0
        for sample in dataset:
            board = np.array(sample['board'])
            gravity_applied = self.apply_gravity(board)
            if not np.array_equal(board, gravity_applied):
                floating_blocks += 1
        
        print(f"\n  🔬 Physics Validation:")
        print(f"    Boards with floating blocks: {floating_blocks} (should be 0)")
        print(f"    Realistic physics compliance: {((num_samples-floating_blocks)/num_samples)*100:.1f}%")
        
        # Save dataset
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"\n💾 Dataset saved to {save_path}")
        print("✅ GUARANTEED: Every scenario has at least 1 powerup available")
        print("✅ BALANCED: All powerups have even distribution opportunities")
        print("✅ REALISTIC: All blocks follow proper Tetris physics (no floating blocks)")
        
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

    def visualize_board(self, board: np.ndarray):
        """Simple text visualization of a board"""
        print("  " + "="*self.width)
        for row in board:
            print("  " + "".join("█" if cell else "·" for cell in row))
        print("  " + "="*self.width)


# Usage example and testing
if __name__ == "__main__":
    generator = TetrisDatasetGenerator()
    
    # Generate a realistic dataset with wildblock support
    print("🚀 Generating realistic Tetris dataset with wildblock support...")
    dataset = generator.generate_dataset(20000, "tetris_boards3.pkl")
    
    # Test loading and analysis
    print("\n" + "="*50)
    print("🔍 Testing dataset loading and analysis...")
    loaded_dataset = generator.load_and_analyze_dataset("tetris_boards3.pkl")
    
    # Show some sample configurations
    print("\n" + "="*50)
    print("📋 Sample realistic board configurations:")
    for i in range(3):
        sample = dataset[i]
        available_powerups = [p for p, avail in sample['powerups'].items() if avail]
        print(f"\nSample {i+1}: {sample['board_type']} board, powerups: {available_powerups}")
        board = np.array(sample['board'])
        generator.visualize_board(board)
        
    print("\n✅ Realistic Tetris dataset generation complete!")
    print("   All blocks properly connected and follow gravity!")