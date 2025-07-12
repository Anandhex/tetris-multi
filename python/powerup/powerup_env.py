import numpy as np
import matplotlib.pyplot as plt
import random


class PowerupEnv:
    def __init__(self):
        self.board_width = 10
        self.board_height = 20
        self.num_powerups = 4
        self.max_steps = 20
        self.reset()

    def reset(self):
        self.steps = 0
        self.done = False
        while True:
            self.player_board = self.generate_random_board(protected_top_rows=0)
            if np.sum(self.player_board) > 0:
                break

        while True:
            self.opponent_board = self.generate_random_board(protected_top_rows=5)
            if np.sum(self.opponent_board) > 0:
                break
        
        self.powerups = self.generate_powerup_vector()
        return self.get_state()
    

    def show_boards_side_by_side(self,player_board, opponent_board,action=None):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

        axes[0].imshow(player_board, cmap='gray_r')
        axes[0].set_title("Player Board")
        axes[0].axis('off')  # optional: hide axes

        axes[1].imshow(opponent_board, cmap='gray_r')
        axes[1].set_title("Opponent Board")
        axes[1].axis('off')  # optional: hide axes

        if(action is not None):
            plt.title(f"action used: {action}")

        plt.tight_layout()
        plt.show()

    def generate_random_board(self, density=None, protected_top_rows=5):
        if density is None:
            density = random.uniform(0.2, 0.6)
        board = np.zeros((self.board_height, self.board_width), dtype=np.float32)
        for y in range(protected_top_rows, self.board_height):  # start from row 5
            if np.random.rand() < density:
                row = np.ones(self.board_width)
                holes = np.random.randint(0, 4)
                for _ in range(holes):
                    row[np.random.randint(0, self.board_width)] = 0
                board[y] = row
        return board

    def generate_powerup_vector(self):
        vector = np.zeros(self.num_powerups, dtype=np.float32)
        available = np.random.choice(self.num_powerups, size=np.random.randint(0, self.num_powerups + 1), replace=False)
        vector[available] = 1.0
        return vector
    
    def extract_board_features(self, board):
        heights = np.zeros(board.shape[1])
        holes = 0

        for col in range(board.shape[1]):
            column = board[:, col]
            nonzero_indices = np.where(column != 0)[0]
            if len(nonzero_indices) == 0:
                heights[col] = 0
            else:
                heights[col] = self.board_height - nonzero_indices[0]

            block_found = False
            col_holes = 0
            for cell in column:
                if cell != 0:
                    block_found = True
                elif block_found and cell == 0:
                    col_holes += 1
            holes += col_holes

        bumpiness = np.sum(np.abs(np.diff(heights)))
        return heights.sum(), holes, bumpiness
    
    def clear_full_lines(self, board):
        full_lines = np.all(board == 1, axis=1)
        num_full = np.sum(full_lines)
        if num_full == 0:
            return board  # no full lines, return as is

        # Keep only rows that are NOT full
        new_board = board[~full_lines]

        # Add empty rows on top to maintain board height
        empty_rows = np.zeros((num_full, board.shape[1]), dtype=board.dtype)
        new_board = np.vstack((empty_rows, new_board))

        return new_board
    def count_full_lines(self, board):
        return np.sum(np.all(board == 1, axis=1))

    def get_state(self):
        # player_board_features = self.extract_board_features(self.player_board)
        # opponent_board_features = self.extract_board_features(self.opponent_board)
        return np.concatenate([self.player_board.copy().flatten(), self.opponent_board.copy().flatten(), self.powerups.copy().flatten()])
        return np.concatenate([
    np.array(player_board_features, dtype=np.float32), 
    np.array(opponent_board_features, dtype=np.float32), 
    self.powerups.copy().flatten()
]) 

    def board_quality(self, board):
        heights = np.zeros(board.shape[1])
        holes = 0

        for col in range(board.shape[1]):
            column = board[:, col]
            nonzero_indices = np.where(column != 0)[0]
            if len(nonzero_indices) == 0:
                heights[col] = 0
            else:
                heights[col] = self.board_height - nonzero_indices[0]

            block_found = False
            col_holes = 0
            for cell in column:
                if cell != 0:
                    block_found = True
                elif block_found and cell == 0:
                    col_holes += 1
            holes += col_holes

        aggregate_height = np.sum(heights)
        bumpiness = np.sum(np.abs(np.diff(heights)))

        score = -0.5 * aggregate_height - 0.7 * holes - 0.3 * bumpiness
        return score

    def step(self, action):
        quality_multiplier = 1.0
        opponent_multiplier = 1.0
        if self.done:
            return self.get_state(), 0, True, {}

        self.steps += 1

        reward = 0
        powerup_used = False
        powerup_had_effect = False
        reward_components = {}

        old_quality = self.board_quality(self.player_board)
        old_opponent_quality = self.board_quality(self.opponent_board)

        # No-op
        if action == 0:
            reward+=-1
            reward_components["noop"] = 0

        elif action == 1 and self.powerups[0] == 1:
            powerup_used = True
            before = np.sum(self.player_board[0, :])
            self.clear_bottom_line()
            self.powerups[0] = 0
            after = np.sum(self.player_board[0, :])
            powerup_had_effect = before > after
            reward_components["clear_bottom_line"] = 0

        elif action == 2 and self.powerups[1] == 1:
            powerup_used = True
            before_board = self.player_board.copy()
            old_quality = self.board_quality(self.player_board)
            old_line_count = self.count_full_lines(self.player_board)

            self.gravity_push()
            self.player_board = self.clear_full_lines(self.player_board)

            new_quality = self.board_quality(self.player_board)
            new_line_count = self.count_full_lines(self.player_board)

            lines_cleared = old_line_count - new_line_count

            powerup_had_effect = not np.array_equal(before_board, self.player_board)

            if lines_cleared > 0:
                reward += lines_cleared * 1.0
                reward_components["gravity_lines_cleared"] = lines_cleared * 1.0

        elif 3 <= action <= 12 and self.powerups[2] == 1:
            powerup_used = True
            col = action - 3
            cleared_block = self.place_bomb(col)
            self.powerups[2] = 0
            powerup_had_effect = cleared_block > 0
            if cleared_block > 0:
                bomb_reward = cleared_block * 0.3
                reward += bomb_reward
                reward_components["bomb"] = bomb_reward

        elif 13 <= action <= 22 and self.powerups[3] == 1:
            powerup_used = True
            col = action - 13
            before = self.opponent_board.copy()
            self.place_wild_card_opponent(col)
            self.powerups[3] = 0
            powerup_had_effect = not np.array_equal(before, self.opponent_board)

            if np.any(self.opponent_board[-1, :]):
                self.opponent_board = self.generate_random_board(protected_top_rows=5)
                reward += 10
                reward_components["wildcard_full"] = 10

        # Ineffective use penalty
        if powerup_used and not powerup_had_effect:
            reward -= 1.0
            reward_components["ineffective_penalty"] = -1.0

        # Player board quality improvement (always applies)
        new_quality = self.board_quality(self.player_board)
        quality_reward = (new_quality - old_quality) * quality_multiplier
        reward += quality_reward
        reward_components["player_quality_improvement"] = quality_reward

        # Opponent penalty
        if 13 <= action <= 22 and not self.powerups[3]:
            new_opponent_quality = self.board_quality(self.opponent_board)
            opponent_penalty = (new_opponent_quality-old_opponent_quality ) * opponent_multiplier
            reward += opponent_penalty
            reward_components["opponent_quality_penalty"] = opponent_penalty

        if np.sum(self.powerups) == 0:
            self.powerups = self.generate_powerup_vector()

        if np.sum(self.player_board) == 0:
            self.done = True
            reward += 10
            reward_components["board_clear_bonus"] = 10
            print(f"Step {self.steps} | Action: {action} | Final Reward: {reward:.2f} | Components: {reward_components}")
            return self.get_state(), reward, self.done, {}

        self.done = self.steps >= self.max_steps
        print(f"Step {self.steps} | Action: {action} | Final Reward: {reward:.2f} | Components: {reward_components}")
        reward = np.clip(reward, -500, 500)
        return self.get_state(), reward, self.done, {}



    def clear_bottom_line(self):
        # Clear bottom line
        self.player_board[0, :] = 0
        # Shift all rows above down by 1
        self.player_board[1:, :] = np.roll(self.player_board[1:, :], shift=-1, axis=0)
        # Set top row to zero to avoid duplication after roll
        self.player_board[-1, :] = 0

    def gravity_push(self):
        for col in range(self.board_width):
            column = self.player_board[:, col]
            blocks = column[column != 0]
            empty = np.zeros(len(column) - len(blocks))
            self.player_board[:, col] = np.concatenate((empty, blocks))

    def place_bomb(self, col):
        bomb_size = 3
        center_col = col
        col_start = max(0, center_col - 1)
        col_end = min(self.board_width, center_col + 2)

        # Drop bomb from bottom up to simulate gravity
        for row in range(self.board_height - bomb_size + 1):
            area = self.player_board[row:row + bomb_size, col_start:col_end]
            if np.any(area != 0):
                landing_row = max(0, row - 1)
                break
        else:
            landing_row = self.board_height - bomb_size


        # Clamp in bounds
        landing_row = row
        row_start = landing_row
        row_end = row_start + bomb_size

        cleared_area = self.player_board[row_start:row_end, col_start:col_end]
        before = np.sum(cleared_area)

       

        self.player_board[row_start:row_end, col_start:col_end] = 0

       
        
        return before



    def place_wild_card_opponent(self, col):
        block_height = 3
        block_width = 3

        # Define horizontal range of the block
        col_start = max(0, col - 1)
        col_end = min(self.board_width, col + 2)  # exclusive
        actual_block_width = col_end - col_start

        # Extract column area for collision detection
        for drop_row in range(self.board_height - block_height + 1):
            # Check 3x3 area below to see if there's any existing block
            area_below = self.opponent_board[drop_row:drop_row + block_height, col_start:col_end]
            # Check the row below the bottom of the block
            if drop_row + block_height >= self.board_height or \
            np.any(self.opponent_board[drop_row + block_height, col_start:col_end]):
                # Found collision or floor — place block above
                target_row = drop_row
                self.opponent_board[target_row:target_row + block_height, col_start:col_end] = 1
                return

        # If no collision found, place block at the bottom
        self.opponent_board[-block_height:, col_start:col_end] = 1

