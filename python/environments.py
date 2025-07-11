from abc import ABC, abstractmethod
import pickle
import random
import numpy as np
from typing import Dict, Tuple
from feature_extractor import UniversalFeatureExtractor

class PowerupEnvironment(ABC):
    """Abstract interface for both training and Unity environments"""
    
    @abstractmethod
    def get_board_state(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_powerup_availability(self) -> Dict[str, bool]:
        pass
    
    @abstractmethod
    def get_features(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def apply_powerup(self, action: Dict) -> Tuple[np.ndarray, float]:
        pass
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        pass


class TrainingEnvironment(PowerupEnvironment):
    """Training environment using .pkl dataset"""
    
    def __init__(self, dataset_path: str):
        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)
        self.feature_extractor = UniversalFeatureExtractor()
        self.current_board = None
        self.current_powerups = None
        print(f"Loaded dataset with {len(self.dataset)} board configurations")
    
    def get_board_state(self) -> np.ndarray:
        """Return raw 2D board"""
        return self.current_board
    
    def get_powerup_availability(self) -> Dict[str, bool]:
        """Return current powerup availability"""
        return self.current_powerups
    
    def get_features(self) -> np.ndarray:
        """Extract features using universal extractor"""
        return self.feature_extractor.extract_features(
            self.current_board, 
            self.current_powerups
        )
    
    def apply_powerup(self, action: Dict) -> Tuple[np.ndarray, float]:
        """Simulate powerup application and calculate reward"""
        old_board = self.current_board.copy()
        new_board = self._simulate_powerup(old_board, action)
        reward = self._calculate_reward(old_board, new_board, action)
        
        self.current_board = new_board
        # Use up the powerup
        if action['type'] != 'none':
            self.current_powerups[action['type']] = False
        
        return new_board, reward
    
    def reset(self) -> np.ndarray:
        """Load random board from dataset"""
        sample = random.choice(self.dataset)
        
        if isinstance(sample, dict):
            self.current_board = np.array(sample['board'], dtype=np.int32)
            self.current_powerups = sample.get('powerups', {
                'bottom_clear': random.choice([True, False]),
                'gravity': random.choice([True, False]),
                'bomb': random.choice([True, False])
            })
        else:
            self.current_board = np.array(sample, dtype=np.int32)
            self.current_powerups = {
                'bottom_clear': random.choice([True, False]),
                'gravity': random.choice([True, False]),
                'bomb': random.choice([True, False])
            }
        
        return self.current_board
    
    def _simulate_powerup(self, board: np.ndarray, action: Dict) -> np.ndarray:
        """Simulate powerup effects"""
        new_board = board.copy()
        
        if action['type'] == 'bottom_clear':
            new_board[-1, :] = 0  # Clear bottom row
            
        elif action['type'] == 'gravity':
            new_board = self._apply_gravity(new_board)
            
        elif action['type'] == 'bomb':
            row, col = action['row'], action['col']
            # Clear 3x3 area
            for r in range(max(0, row), min(board.shape[0], row + 3)):
                for c in range(max(0, col), min(board.shape[1], col + 3)):
                    new_board[r, c] = 0
        
        return new_board
    
    def _apply_gravity(self, board: np.ndarray) -> np.ndarray:
        """Apply gravity effect - move blocks down to fill holes"""
        new_board = board.copy()
        rows, cols = new_board.shape
        
        for col in range(cols):
            # Extract all blocks in this column
            blocks = []
            for row in range(rows-1, -1, -1):  # Bottom to top
                if new_board[row, col] == 1:
                    blocks.append(1)
                    new_board[row, col] = 0
            
            # Place blocks at bottom
            for i, block in enumerate(blocks):
                new_board[rows-1-i, col] = block
        
        return new_board
    
    def _calculate_reward(self, old_board: np.ndarray, new_board: np.ndarray, action: Dict) -> float:
        """Calculate reward for powerup usage"""
        # Board quality improvement
        old_quality = self._evaluate_board_quality(old_board)
        new_quality = self._evaluate_board_quality(new_board)
        improvement = new_quality - old_quality
        
        base_reward = improvement * 10
        
        # Action-specific bonuses
        if action['type'] == 'bottom_clear':
            blocks_cleared = np.sum(old_board[-1, :])
            base_reward += blocks_cleared * 5 + 20
            
        elif action['type'] == 'gravity':
            old_holes = self.feature_extractor._count_holes(old_board)
            new_holes = self.feature_extractor._count_holes(new_board)
            holes_filled = old_holes - new_holes
            base_reward += holes_filled * 15 + 10
            
        elif action['type'] == 'bomb':
            blocks_destroyed = np.sum(old_board) - np.sum(new_board)
            base_reward += blocks_destroyed * 3 + 15
            
        elif action['type'] == 'none':
            base_reward += 1  # Small reward for conservation
        
        return float(base_reward)
    
    def _evaluate_board_quality(self, board: np.ndarray) -> float:
        """Evaluate overall board quality (higher is better)"""
        holes = self.feature_extractor._count_holes(board)
        bumpiness = self.feature_extractor._calculate_bumpiness(board)
        max_height = self.feature_extractor._get_max_height(board)
        
        # Lower values are better, so negate them
        quality = -holes * 2 - bumpiness * 0.5 - max_height * 0.1
        return quality