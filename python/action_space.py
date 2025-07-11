import numpy as np
from typing import Dict, Tuple
from feature_extractor import UniversalFeatureExtractor

class ActionSpace:
    """Manages action encoding/decoding for powerups"""
    
    def __init__(self):
        self.action_types = {
            0: 'none',
            1: 'bottom_clear', 
            2: 'gravity',
            3: 'bomb'
        }
        self.num_actions = len(self.action_types)
    
    def encode_action(self, action_type: str) -> int:
        """Convert action type to integer for DQN"""
        for key, value in self.action_types.items():
            if value == action_type:
                return key
        return 0  # Default to 'none'
    
    def decode_action(self, action_id: int, board: np.ndarray) -> Dict:
        """Convert DQN output to executable action"""
        if action_id == 0:
            return {'type': 'none'}
        elif action_id == 1:
            return {'type': 'bottom_clear'}
        elif action_id == 2:
            return {'type': 'gravity'}
        elif action_id == 3:
            # Find best bomb position dynamically
            best_pos = self._find_best_bomb_position(board)
            if best_pos:
                return {'type': 'bomb', 'row': best_pos[0], 'col': best_pos[1]}
            else:
                return {'type': 'none'}  # No valid bomb positions
        
        return {'type': 'none'}
    
    def _find_best_bomb_position(self, board: np.ndarray, bomb_size: int = 3) -> Tuple[int, int]:
        """Find the most effective bomb position"""
        rows, cols = board.shape
        best_pos = None
        max_effectiveness = 0
        
        extractor = UniversalFeatureExtractor()
        
        for row in range(rows - bomb_size + 1):
            for col in range(cols - bomb_size + 1):
                effectiveness = extractor._calculate_bomb_effectiveness(board, row, col, bomb_size)
                if effectiveness > max_effectiveness:
                    max_effectiveness = effectiveness
                    best_pos = (row, col)
        
        return best_pos