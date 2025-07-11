import numpy as np
import pickle
import random

class TetrisDatasetGenerator:
    """Generate realistic Tetris board configurations for training"""
    
    def __init__(self, board_height=20, board_width=10):
        self.height = board_height
        self.width = board_width
    
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
        problem_type = random.choice(['holes', 'high_stack', 'uneven'])
        
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
        
        return board
    
    def generate_dataset(self, num_samples: int, save_path: str):
        """Generate and save a complete dataset"""
        dataset = []
        
        print(f"Generating {num_samples} board configurations...")
        
        for i in range(num_samples):
            board_type = random.choice(['random', 'structured', 'problem'])
            
            if board_type == 'random':
                board = self.generate_random_board()
            elif board_type == 'structured':
                board = self.generate_structured_board()
            else:
                board = self.create_problem_board()
            
            # Add powerup availability (random for training diversity)
            powerups = {
                'bottom_clear': random.choice([True, False]),
                'gravity': random.choice([True, False]),
                'bomb': random.choice([True, False])
            }
            
            dataset.append({
                'board': board.tolist(),  # Convert to list for JSON serialization
                'powerups': powerups,
                'board_type': board_type
            })
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_samples} boards")
        
        # Save dataset
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"Dataset saved to {save_path}")
        return dataset