import numpy as np
from typing import Dict, Tuple, List

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
    
    def _find_best_bomb_position(self, board: np.ndarray) -> Tuple[int, int]:
        """Find the most effective bomb position (only surface blocks)"""
        surface_blocks = self._get_surface_blocks(board)
        best_pos = None
        max_effectiveness = 0
        
        for hit_row, hit_col in surface_blocks:
            effectiveness = self._calculate_bomb_effectiveness_on_surface(board, hit_row, hit_col)
            if effectiveness > max_effectiveness:
                max_effectiveness = effectiveness
                best_pos = (hit_row, hit_col)
        
        return best_pos
    
    def _get_surface_blocks(self, board: np.ndarray) -> List[Tuple[int, int]]:
        """Get all surface blocks (blocks that can be hit by bomb from above)"""
        rows, cols = board.shape
        surface_blocks = []
        
        for col in range(cols):
            for row in range(rows):
                if board[row, col] == 1:  # Found a block
                    # This is a surface block (first block from top in this column)
                    surface_blocks.append((row, col))
                    break  # Only the topmost block in each column is reachable
        
        return surface_blocks
    
    def _calculate_bomb_effectiveness_on_surface(self, board: np.ndarray, hit_row: int, hit_col: int) -> float:
        """
        Calculate bomb effectiveness when bomb hits a surface block
        
        Explosion pattern: 3x3 area centered on hit block
        Relative offsets from hit point: (-1,-1) to (1,1)
        """
        rows, cols = board.shape
        
        # Define 3x3 explosion pattern around hit point
        explosion_offsets = [
            (-1, -1), (-1, 0), (-1, 1),  # Row above
            (0, -1),  (0, 0),  (0, 1),   # Same row as hit
            (1, -1),  (1, 0),  (1, 1)    # Row below
        ]
        
        # Count blocks that would be destroyed in explosion
        blocks_destroyed = 0
        destroyed_positions = []
        
        for row_offset, col_offset in explosion_offsets:
            explosion_row = hit_row + row_offset
            explosion_col = hit_col + col_offset
            
            # Check if position is valid (within board bounds)
            if 0 <= explosion_row < rows and 0 <= explosion_col < cols:
                if board[explosion_row, explosion_col] == 1:
                    blocks_destroyed += 1
                    destroyed_positions.append((explosion_row, explosion_col))
        
        if blocks_destroyed == 0:
            return 0
        
        # Simulate board after explosion to calculate improvements
        temp_board = board.copy()
        for destroy_row, destroy_col in destroyed_positions:
            temp_board[destroy_row, destroy_col] = 0
        
        # Calculate hole reduction (important benefit)
        old_holes = self._count_holes(board)
        new_holes = self._count_holes(temp_board)
        holes_reduced = old_holes - new_holes
        
        # Calculate bumpiness improvement
        old_bumpiness = self._calculate_bumpiness(board)
        new_bumpiness = self._calculate_bumpiness(temp_board)
        bumpiness_reduced = old_bumpiness - new_bumpiness
        
        # Weighted effectiveness score
        effectiveness = (
            blocks_destroyed * 1.0 +      # Base value for blocks destroyed
            holes_reduced * 2.0 +         # High bonus for hole reduction
            bumpiness_reduced * 0.5       # Bonus for surface smoothing
        )
        
        return effectiveness
    
    def _count_holes(self, board: np.ndarray) -> int:
        """Count holes in board"""
        holes = 0
        rows, cols = board.shape
        for col in range(cols):
            block_found = False
            for row in range(rows):
                if board[row, col] == 1:
                    block_found = True
                elif block_found and board[row, col] == 0:
                    holes += 1
        return holes
    
    def _calculate_bumpiness(self, board: np.ndarray) -> int:
        """Calculate surface bumpiness"""
        heights = []
        rows, cols = board.shape
        for col in range(cols):
            height = 0
            for row in range(rows):
                if board[row, col] == 1:
                    height = rows - row
                    break
            heights.append(height)
        
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness