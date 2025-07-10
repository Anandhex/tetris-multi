import numpy as np

class PowerupEnv:
    def __init__(self):
        self.board_width = 10
        self.board_height = 20
        self.num_powerups = 4
        self.max_steps = 200
        self.reset()

    def reset(self):
        self.steps = 0
        self.done = False
        self.player_board = self.generate_random_board()
        self.opponent_board = self.generate_random_board()
        self.powerups = self.generate_powerup_vector()
        return self.get_state()

    def generate_random_board(self, density=0.2):
        board = np.zeros((self.board_height, self.board_width), dtype=np.float32)
        for y in range(self.board_height):
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

    def get_state(self):
        return np.concatenate([
            self.player_board.flatten(),
            self.opponent_board.flatten(),
            self.powerups
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
        if self.done:
            return self.get_state(), 0, True, {}

        self.steps += 1

        old_quality = self.board_quality(self.player_board)

        reward = 0

        if action == 0:
            # No-op
            if np.sum(self.player_board) == 0:
                reward = 1  # correct no-op on empty board
            else:
                reward = -0.1
        elif action == 1 and self.powerups[0] == 1:
            self.clear_bottom_line()
            self.powerups[0] = 0
            reward = 5
        elif action == 2 and self.powerups[1] == 1:
            self.gravity_push()
            self.powerups[1] = 0
            reward = 4
        elif 3 <= action <= 12:
            col = action - 3
            if self.powerups[2] == 1:
                self.place_bomb(col)
                self.powerups[2] = 0
                reward = 6
            else:
                reward = -2
        elif 13 <= action <= 22:
            col = action - 13
            if self.powerups[3] == 1:
                self.place_wild_card_opponent(col)
                self.powerups[3] = 0
                reward = 6
            else:
                reward = -2
        else:
            reward = -1

        new_quality = self.board_quality(self.player_board)
        quality_diff = new_quality - old_quality

        reward += quality_diff * 10  # scale improvement impact

        self.done = self.steps >= self.max_steps
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
        row_start = 0
        row_end = min(3, self.board_height)
        col_start = max(0, col - 1)
        col_end = min(self.board_width, col + 2)
        self.player_board[row_start:row_end, col_start:col_end] = 0

    def place_wild_card_opponent(self, col):
        rows_to_fill = min(3, self.board_height)
        self.opponent_board[0:rows_to_fill, col] = 1

