import numpy as np
import random
import pickle
from collections import deque, defaultdict
from tqdm import tqdm

class RealisticTetrisGameGenerator:
    """Enhanced game state generator that creates realistic Tetris scenarios with holes"""
    
    def __init__(self, board_width=10, board_height=20):
        self.board_width = board_width
        self.board_height = board_height
        
        # All 7 Tetris pieces with all rotations
        self.pieces = {
            'I': [
                [[1,1,1,1]],
                [[1],[1],[1],[1]]
            ],
            'O': [
                [[1,1],[1,1]]
            ],
            'T': [
                [[0,1,0],[1,1,1]],
                [[1,0],[1,1],[1,0]],
                [[1,1,1],[0,1,0]],
                [[0,1],[1,1],[0,1]]
            ],
            'S': [
                [[0,1,1],[1,1,0]],
                [[1,0],[1,1],[0,1]]
            ],
            'Z': [
                [[1,1,0],[0,1,1]],
                [[0,1],[1,1],[1,0]]
            ],
            'J': [
                [[1,0,0],[1,1,1]],
                [[1,1],[1,0],[1,0]],
                [[1,1,1],[0,0,1]],
                [[0,1],[0,1],[1,1]]
            ],
            'L': [
                [[0,0,1],[1,1,1]],
                [[1,0],[1,0],[1,1]],
                [[1,1,1],[1,0,0]],
                [[1,1],[0,1],[0,1]]
            ]
        }
        
        # Piece frequencies (some pieces are more common)
        self.piece_weights = {
            'I': 1.0, 'O': 1.0, 'T': 1.2, 'S': 1.0, 
            'Z': 1.0, 'J': 1.0, 'L': 1.0
        }
        
        # Game difficulty progression
        self.difficulty_levels = {
            'easy': {'hole_probability': 0.1, 'bad_placement_chance': 0.1},
            'medium': {'hole_probability': 0.2, 'bad_placement_chance': 0.2},
            'hard': {'hole_probability': 0.3, 'bad_placement_chance': 0.3},
            'expert': {'hole_probability': 0.4, 'bad_placement_chance': 0.4}
        }
    
    def create_realistic_base_board(self, difficulty='medium'):
        """Create a realistic board with some existing blocks and holes"""
        board = np.zeros((self.board_height, self.board_width), dtype=int)
        
        # Add some base layers with holes
        base_height = random.randint(2, 8)
        difficulty_params = self.difficulty_levels[difficulty]
        
        for row in range(self.board_height - base_height, self.board_height):
            for col in range(self.board_width):
                # Create holes with some probability
                if random.random() > difficulty_params['hole_probability']:
                    board[row, col] = random.randint(1, 7)
                    
            # Ensure each row has at least some blocks
            filled_cells = np.sum(board[row, :] != 0)
            if filled_cells < 3:
                # Add some random blocks
                empty_cols = [c for c in range(self.board_width) if board[row, c] == 0]
                fill_count = min(3 - filled_cells, len(empty_cols))
                for col in random.sample(empty_cols, fill_count):
                    board[row, col] = random.randint(1, 7)
        
        return board
    
    def find_all_valid_placements(self, board, piece):
        """Find all valid placements for a piece"""
        valid_placements = []
        piece_height = len(piece)
        piece_width = len(piece[0])
        
        for col in range(self.board_width - piece_width + 1):
            # Find the landing row for this column
            landing_row = self._find_landing_row(board, piece, col)
            
            if landing_row >= 0 and landing_row + piece_height <= self.board_height:
                # Check if placement is valid
                if self._can_place_piece(board, piece, landing_row, col):
                    valid_placements.append((landing_row, col))
        
        return valid_placements
    
    def _find_landing_row(self, board, piece, col):
        """Find where piece lands when dropped in column"""
        piece_height = len(piece)
        piece_width = len(piece[0])
        
        for row in range(self.board_height - piece_height + 1):
            if not self._can_place_piece(board, piece, row, col):
                return max(0, row - 1)
        
        return self.board_height - piece_height
    
    def _can_place_piece(self, board, piece, row, col):
        """Check if piece can be placed at position"""
        piece_height = len(piece)
        piece_width = len(piece[0])
        
        for pr in range(piece_height):
            for pc in range(piece_width):
                if piece[pr][pc] != 0:
                    board_row = row + pr
                    board_col = col + pc
                    
                    if (board_row >= self.board_height or 
                        board_col >= self.board_width or
                        board_row < 0 or board_col < 0 or
                        board[board_row, board_col] != 0):
                        return False
        
        return True
    
    def evaluate_placement(self, board, piece, row, col):
        """Evaluate placement quality using multiple heuristics"""
        # Create test board
        test_board = board.copy()
        self._place_piece(test_board, piece, row, col)
        
        # Clear lines and get metrics
        test_board, lines_cleared = self._clear_lines(test_board)
        
        # Calculate heuristics
        height_score = -self._get_max_height(test_board) * 4
        hole_score = -self._count_holes(test_board) * 10
        bumpiness_score = -self._calculate_bumpiness(test_board) * 2
        line_score = lines_cleared * 100
        
        # Bonus for placing low
        placement_height = self.board_height - row
        height_bonus = max(0, (20 - placement_height) * 2)
        
        # Penalty for creating unreachable holes
        deep_hole_penalty = -self._count_deep_holes(test_board) * 20
        
        total_score = (height_score + hole_score + bumpiness_score + 
                      line_score + height_bonus + deep_hole_penalty)
        
        return total_score, {
            'height': height_score,
            'holes': hole_score,
            'bumpiness': bumpiness_score,
            'lines': line_score,
            'placement_height': height_bonus,
            'deep_holes': deep_hole_penalty
        }
    
    def _count_deep_holes(self, board):
        """Count holes that are hard to reach (surrounded by blocks)"""
        deep_holes = 0
        for col in range(self.board_width):
            blocks_above = 0
            for row in range(self.board_height):
                if board[row, col] != 0:
                    blocks_above += 1
                elif blocks_above > 0:  # This is a hole
                    # Check if it's deep (many blocks above)
                    if blocks_above >= 3:
                        deep_holes += 1
        return deep_holes
    
    def place_piece_intelligently(self, board, difficulty='medium'):
        """Place piece using intelligent strategy with some randomness"""
        # Get random piece
        piece_type = random.choices(
            list(self.pieces.keys()),
            weights=list(self.piece_weights.values())
        )[0]
        
        # Get random rotation
        rotations = self.pieces[piece_type]
        piece = random.choice(rotations)
        
        # Find all valid placements
        valid_placements = self.find_all_valid_placements(board, piece)
        
        if not valid_placements:
            return board, 0, piece_type  # Game over
        
        # Evaluate all placements
        placement_scores = []
        for row, col in valid_placements:
            score, details = self.evaluate_placement(board, piece, row, col)
            placement_scores.append((score, row, col, details))
        
        # Sort by score
        placement_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Choose placement based on difficulty
        difficulty_params = self.difficulty_levels[difficulty]
        bad_placement_chance = difficulty_params['bad_placement_chance']
        
        if random.random() < bad_placement_chance:
            # Make suboptimal choice
            chosen_idx = random.randint(len(placement_scores)//2, len(placement_scores)-1)
        else:
            # Make good choice (top 25%)
            chosen_idx = random.randint(0, max(0, len(placement_scores)//4))
        
        score, row, col, details = placement_scores[chosen_idx]
        
        # Place the piece
        new_board = board.copy()
        self._place_piece(new_board, piece, row, col)
        
        # Clear lines
        new_board, lines_cleared = self._clear_lines(new_board)
        
        return new_board, lines_cleared, piece_type
    
    def _place_piece(self, board, piece, row, col):
        """Place piece on board"""
        piece_height = len(piece)
        piece_width = len(piece[0])
        
        for pr in range(piece_height):
            for pc in range(piece_width):
                if piece[pr][pc] != 0:
                    board_row = row + pr
                    board_col = col + pc
                    if (0 <= board_row < self.board_height and 
                        0 <= board_col < self.board_width):
                        board[board_row, board_col] = piece[pr][pc]
    
    def _clear_lines(self, board):
        """Clear completed lines"""
        lines_cleared = 0
        row = self.board_height - 1
        
        while row >= 0:
            if np.all(board[row, :] != 0):  # Line is full
                board = np.delete(board, row, axis=0)
                board = np.vstack([np.zeros((1, self.board_width)), board])
                lines_cleared += 1
            else:
                row -= 1
        
        return board, lines_cleared
    
    def _get_max_height(self, board):
        """Get maximum height of board"""
        for row in range(self.board_height):
            if np.any(board[row, :] != 0):
                return self.board_height - row
        return 0
    
    def _count_holes(self, board):
        """Count holes in board"""
        holes = 0
        for col in range(self.board_width):
            found_block = False
            for row in range(self.board_height):
                if board[row, col] != 0:
                    found_block = True
                elif found_block and board[row, col] == 0:
                    holes += 1
        return holes
    
    def _calculate_bumpiness(self, board):
        """Calculate bumpiness (height differences between columns)"""
        heights = []
        for col in range(self.board_width):
            height = 0
            for row in range(self.board_height):
                if board[row, col] != 0:
                    height = self.board_height - row
                    break
            heights.append(height)
        
        bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))
        return bumpiness
    
    def calculate_board_features(self, board):
        """Calculate comprehensive board features"""
        lines_cleared = 0  # This would be tracked during gameplay
        holes = self._count_holes(board)
        bumpiness = self._calculate_bumpiness(board)
        height = self._get_max_height(board)
        
        return [lines_cleared, holes, bumpiness, height]
    
    def generate_diverse_game_trajectory(self, max_pieces=150, difficulty='medium'):
        """Generate a diverse game trajectory with varying difficulty"""
        # Start with realistic base
        board = self.create_realistic_base_board(difficulty)
        trajectory = []
        
        # Track game statistics
        total_lines_cleared = 0
        piece_count = 0
        
        for piece_num in range(max_pieces):
            # Store current state
            board_features = self.calculate_board_features(board)
            board_features[0] = total_lines_cleared  # Update actual lines cleared
            
            state = {
                'piece_number': piece_num,
                'board_2d': board.copy(),
                'board_features': board_features,
                'timestamp': piece_num,
                'difficulty': difficulty,
                'game_stats': {
                    'total_lines': total_lines_cleared,
                    'piece_count': piece_count
                }
            }
            trajectory.append(state)
            
            # Place next piece
            board, lines_cleared, piece_type = self.place_piece_intelligently(board, difficulty)
            total_lines_cleared += lines_cleared
            piece_count += 1
            
            # Check game over condition
            if self._get_max_height(board) >= self.board_height - 2:
                break
            
            # Dynamically adjust difficulty
            if piece_num > 50 and piece_num % 30 == 0:
                complexity = sum(board_features[1:])  # holes + bumpiness + height
                if complexity < 20:
                    difficulty = 'hard'  # Make it harder
                elif complexity > 50:
                    difficulty = 'easy'  # Make it easier
        
        return trajectory

class EnhancedGameStateBuffer:
    """Enhanced buffer with better sampling and statistics"""
    
    def __init__(self, max_size=100000):
        self.max_size = max_size
        self.states = deque(maxlen=max_size)
        self.difficulty_indices = defaultdict(list)
        self.complexity_indices = defaultdict(list)
        
        self.metadata = {
            'total_games': 0,
            'total_states': 0,
            'avg_game_length': 0,
            'difficulty_distribution': defaultdict(int),
            'complexity_distribution': defaultdict(int)
        }
    
    def add_game_trajectory(self, game_states):
        """Add game trajectory with indexing"""
        start_idx = len(self.states)
        
        for i, state in enumerate(game_states):
            self.states.append(state)
            
            # Index by difficulty
            difficulty = state.get('difficulty', 'medium')
            self.difficulty_indices[difficulty].append(start_idx + i)
            
            # Index by complexity
            complexity = self._assess_complexity(state['board_features'])
            self.complexity_indices[complexity].append(start_idx + i)
        
        # Update metadata
        self.metadata['total_games'] += 1
        self.metadata['total_states'] = len(self.states)
        self.metadata['avg_game_length'] = len(self.states) / self.metadata['total_games']
        
        # Update distributions
        for state in game_states:
            difficulty = state.get('difficulty', 'medium')
            self.metadata['difficulty_distribution'][difficulty] += 1
            
            complexity = self._assess_complexity(state['board_features'])
            self.metadata['complexity_distribution'][complexity] += 1
    
    def _assess_complexity(self, board_features):
        """Assess complexity level of board state"""
        lines, holes, bumpiness, height = board_features
        
        complexity_score = holes * 2 + bumpiness + height
        
        if complexity_score < 15:
            return 'low'
        elif complexity_score < 35:
            return 'medium'
        elif complexity_score < 60:
            return 'high'
        else:
            return 'extreme'
    
    def sample_powerup_scenario(self, difficulty=None, complexity=None, powerup_type=None):
        """Sample scenario with filtering options"""
        if not self.states:
            return None
        
        # Filter by difficulty
        if difficulty and difficulty in self.difficulty_indices:
            candidate_indices = self.difficulty_indices[difficulty]
        else:
            candidate_indices = list(range(len(self.states)))
        
        # Filter by complexity
        if complexity and complexity in self.complexity_indices:
            complexity_indices = set(self.complexity_indices[complexity])
            candidate_indices = [i for i in candidate_indices if i in complexity_indices]
        
        if not candidate_indices:
            candidate_indices = list(range(len(self.states)))
        
        # Sample random state
        state_idx = random.choice(candidate_indices)
        state = self.states[state_idx].copy()
        
        # Add powerup context
        if powerup_type:
            state['available_powerup'] = powerup_type
        else:
            state['available_powerup'] = random.choice(['bottom_line_clear', 'gravity', 'bomb'])
        
        state['blocks_since_powerup'] = random.randint(0, 10)
        
        return state
    
    def get_balanced_sample(self, count=1000):
        """Get balanced sample across difficulties and complexities"""
        samples = []
        
        # Ensure balanced sampling
        difficulties = list(self.difficulty_indices.keys())
        complexities = list(self.complexity_indices.keys())
        
        if not difficulties or not complexities:
            # Fallback to random sampling
            for _ in range(count):
                sample = self.sample_powerup_scenario()
                if sample:
                    samples.append(sample)
            return samples
        
        samples_per_category = count // (len(difficulties) * len(complexities))
        
        for difficulty in difficulties:
            for complexity in complexities:
                for _ in range(samples_per_category):
                    sample = self.sample_powerup_scenario(difficulty, complexity)
                    if sample:
                        samples.append(sample)
        
        return samples
    
    def get_statistics(self):
        """Get buffer statistics"""
        return self.metadata.copy()
    
    def save_to_file(self, filepath):
        """Save buffer with indices"""
        data = {
            'states': list(self.states),
            'metadata': self.metadata,
            'difficulty_indices': dict(self.difficulty_indices),
            'complexity_indices': dict(self.complexity_indices)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Enhanced buffer saved to {filepath}")
    
    def load_from_file(self, filepath):
        """Load buffer with indices"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.states = deque(data['states'], maxlen=self.max_size)
            self.metadata = data['metadata']
            self.difficulty_indices = defaultdict(list, data.get('difficulty_indices', {}))
            self.complexity_indices = defaultdict(list, data.get('complexity_indices', {}))
            
            print(f"Enhanced buffer loaded from {filepath}")
            print(f"States: {len(self.states)}, Games: {self.metadata['total_games']}")
            return True
        except Exception as e:
            print(f"Failed to load enhanced buffer: {e}")
            return False

# Usage example for generating large dataset
if __name__ == "__main__":
    # Create enhanced generator
    generator = RealisticTetrisGameGenerator()
    buffer = EnhancedGameStateBuffer(max_size=200000)
    
    # Generate large diverse dataset
    num_games = 5000
    print(f"Generating {num_games} realistic Tetris games...")
    
    difficulties = ['easy', 'medium', 'hard', 'expert']
    
    for i in tqdm(range(num_games), desc="Generating games"):
        # Vary difficulty
        difficulty = random.choice(difficulties)
        
        # Generate trajectory
        trajectory = generator.generate_diverse_game_trajectory(
            max_pieces=random.randint(100, 200),
            difficulty=difficulty
        )
        
        # Add to buffer
        buffer.add_game_trajectory(trajectory)
        
        # Print progress
        if (i + 1) % 500 == 0:
            stats = buffer.get_statistics()
            print(f"\nProgress: {i+1}/{num_games} games")
            print(f"Total states: {stats['total_states']}")
            print(f"Avg game length: {stats['avg_game_length']:.1f}")
            print(f"Difficulty distribution: {dict(stats['difficulty_distribution'])}")
    
    # Save the dataset
    buffer.save_to_file("large_realistic_tetris_dataset.pkl")
    
    # Print final statistics
    print("\nFinal Dataset Statistics:")
    stats = buffer.get_statistics()
    print(f"Total games: {stats['total_games']}")
    print(f"Total states: {stats['total_states']}")
    print(f"Average game length: {stats['avg_game_length']:.1f}")
    print(f"Difficulty distribution: {dict(stats['difficulty_distribution'])}")
    print(f"Complexity distribution: {dict(stats['complexity_distribution'])}")