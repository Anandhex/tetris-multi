import numpy as np
from typing import Dict, List, Tuple

class UniversalFeatureExtractor:
    """Feature extractor that works with raw 2D boards from any source"""
    
    def __init__(self):
        self.feature_names = [
            'holes_count', 'bumpiness', 'max_height', 'avg_height',
            'lines_ready_to_clear', 'deep_wells', 'column_transitions',
            'bottom_line_benefit', 'gravity_benefit', 'best_bomb_effectiveness',
            'powerup_bottom_available', 'powerup_gravity_available', 'powerup_bomb_available'
        ]
    
    def extract_features(self, board: np.ndarray, powerup_availability: Dict[str, bool]) -> np.ndarray:
        """
        Extract features from raw 2D board
        
        Args:
            board: 2D numpy array (rows x cols) where 1=block, 0=empty
            powerup_availability: dict with keys ['bottom_clear', 'gravity', 'bomb']
        
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Basic board analysis features
        features.append(self._count_holes(board))
        features.append(self._calculate_bumpiness(board))
        features.append(self._get_max_height(board))
        features.append(self._get_avg_height(board))
        features.append(self._lines_ready_to_clear(board))
        features.append(self._count_deep_wells(board))
        features.append(self._column_transitions(board))
        
        # Powerup-specific features
        features.append(self._bottom_line_benefit(board) if powerup_availability.get('bottom_clear', False) else 0)
        features.append(self._gravity_benefit(board) if powerup_availability.get('gravity', False) else 0)
        features.append(self._best_bomb_effectiveness(board) if powerup_availability.get('bomb', False) else 0)
        
        # Powerup availability flags
        features.append(1.0 if powerup_availability.get('bottom_clear', False) else 0.0)
        features.append(1.0 if powerup_availability.get('gravity', False) else 0.0)
        features.append(1.0 if powerup_availability.get('bomb', False) else 0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _count_holes(self, board: np.ndarray) -> float:
        """Count holes (empty cells with blocks above them)"""
        holes = 0
        rows, cols = board.shape
        
        for col in range(cols):
            block_found = False
            for row in range(rows):
                if board[row, col] == 1:
                    block_found = True
                elif block_found and board[row, col] == 0:
                    holes += 1
        return float(holes)
    
    def _calculate_bumpiness(self, board: np.ndarray) -> float:
        """Calculate surface bumpiness"""
        heights = self._get_column_heights(board)
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return float(bumpiness)
    
    def _get_column_heights(self, board: np.ndarray) -> List[int]:
        """Get height of each column"""
        rows, cols = board.shape
        heights = []
        
        for col in range(cols):
            height = 0
            for row in range(rows):
                if board[row, col] == 1:
                    height = rows - row
                    break
            heights.append(height)
        return heights
    
    def _get_max_height(self, board: np.ndarray) -> float:
        """Get maximum column height"""
        heights = self._get_column_heights(board)
        return float(max(heights)) if heights else 0.0
    
    def _get_avg_height(self, board: np.ndarray) -> float:
        """Get average column height"""
        heights = self._get_column_heights(board)
        return float(np.mean(heights)) if heights else 0.0
    
    def _lines_ready_to_clear(self, board: np.ndarray) -> float:
        """Count complete lines ready to clear"""
        rows, cols = board.shape
        complete_lines = 0
        
        for row in range(rows):
            if np.sum(board[row, :]) == cols:
                complete_lines += 1
        return float(complete_lines)
    
    def _count_deep_wells(self, board: np.ndarray) -> float:
        """Count deep wells (columns significantly lower than neighbors)"""
        heights = self._get_column_heights(board)
        wells = 0
        
        for i in range(len(heights)):
            left_height = heights[i-1] if i > 0 else 0
            right_height = heights[i+1] if i < len(heights)-1 else 0
            current_height = heights[i]
            
            if current_height < left_height - 2 and current_height < right_height - 2:
                wells += 1
        return float(wells)
    
    def _column_transitions(self, board: np.ndarray) -> float:
        """Count transitions between filled and empty cells in columns"""
        rows, cols = board.shape
        transitions = 0
        
        for col in range(cols):
            for row in range(rows - 1):
                if board[row, col] != board[row + 1, col]:
                    transitions += 1
        return float(transitions)
    
    def _bottom_line_benefit(self, board: np.ndarray) -> float:
        """Calculate benefit of clearing bottom line"""
        rows, cols = board.shape
        bottom_filled = np.sum(board[-1, :])
        return float(bottom_filled / cols)  # Percentage of bottom line filled
    
    def _gravity_benefit(self, board: np.ndarray) -> float:
        """Calculate how many holes gravity would fill"""
        holes_fillable = 0
        rows, cols = board.shape
        
        for col in range(cols):
            blocks_above = 0
            holes_below = 0
            
            for row in range(rows):
                if board[row, col] == 1:
                    blocks_above += 1
                else:
                    # Check if this empty cell has blocks above it
                    has_blocks_above = any(board[r, col] == 1 for r in range(row))
                    if has_blocks_above:
                        holes_below += 1
            
            # Gravity can fill holes up to the number of movable blocks
            fillable_in_column = min(blocks_above, holes_below)
            holes_fillable += fillable_in_column
            
        return float(holes_fillable)
    
    def _best_bomb_effectiveness(self, board: np.ndarray, bomb_size: int = 3) -> float:
        """Calculate effectiveness of best bomb placement"""
        rows, cols = board.shape
        max_effectiveness = 0
        
        for row in range(rows - bomb_size + 1):
            for col in range(cols - bomb_size + 1):
                effectiveness = self._calculate_bomb_effectiveness(board, row, col, bomb_size)
                max_effectiveness = max(max_effectiveness, effectiveness)
        
        return float(max_effectiveness)
    
    def _calculate_bomb_effectiveness(self, board: np.ndarray, row: int, col: int, size: int) -> float:
        """Calculate comprehensive bomb effectiveness"""
        bomb_area = board[row:row+size, col:col+size]
        blocks_destroyed = np.sum(bomb_area)
        
        if blocks_destroyed == 0:
            return 0
        
        # Simulate board after bomb
        temp_board = board.copy()
        temp_board[row:row+size, col:col+size] = 0
        
        # Calculate improvements
        old_holes = self._count_holes(board)
        new_holes = self._count_holes(temp_board)
        hole_reduction = old_holes - new_holes
        
        old_bumpiness = self._calculate_bumpiness(board)
        new_bumpiness = self._calculate_bumpiness(temp_board)
        bumpiness_reduction = old_bumpiness - new_bumpiness
        
        # Weighted effectiveness score
        effectiveness = (
            blocks_destroyed * 1.0 +
            hole_reduction * 2.0 +
            bumpiness_reduction * 0.5
        )
        
        return effectiveness