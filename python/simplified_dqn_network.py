import torch
from keras.models import Sequential
from keras.layers import Dense
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
import numpy as np
import os
from datetime import datetime

class TetrisPiece:
    """Tetris piece definitions with rotations and shapes"""
    
    PIECES = {
        'I': [
            [[1, 1, 1, 1]],  # Horizontal
            [[1], [1], [1], [1]]  # Vertical
        ],
        'O': [
            [[1, 1], [1, 1]]  # Square (no rotation)
        ],
        'T': [
            [[0, 1, 0], [1, 1, 1]],  # Normal
            [[1, 0], [1, 1], [1, 0]],  # Right
            [[1, 1, 1], [0, 1, 0]],  # Upside down
            [[0, 1], [1, 1], [0, 1]]  # Left
        ],
        'S': [
            [[0, 1, 1], [1, 1, 0]],  # Normal
            [[1, 0], [1, 1], [0, 1]]  # Vertical
        ],
        'Z': [
            [[1, 1, 0], [0, 1, 1]],  # Normal
            [[0, 1], [1, 1], [1, 0]]  # Vertical
        ],
        'J': [
            [[1, 0, 0], [1, 1, 1]],  # Normal
            [[1, 1], [1, 0], [1, 0]],  # Right
            [[1, 1, 1], [0, 0, 1]],  # Upside down
            [[0, 1], [0, 1], [1, 1]]  # Left
        ],
        'L': [
            [[0, 0, 1], [1, 1, 1]],  # Normal
            [[1, 0], [1, 0], [1, 1]],  # Right
            [[1, 1, 1], [1, 0, 0]],  # Upside down
            [[1, 1], [0, 1], [0, 1]]  # Left
        ]
    }
    
    @staticmethod
    def get_piece_shape(piece_type, rotation):
        """Get the shape matrix for a piece at given rotation"""
        if piece_type not in TetrisPiece.PIECES:
            return [[1]]  # Default fallback
        
        rotations = TetrisPiece.PIECES[piece_type]
        rotation_index = rotation % len(rotations)
        return rotations[rotation_index]

class EnhancedTetrisEnvironment:
    """Enhanced environment with proper game simulation"""
    
    def __init__(self):
        self.piece_types = ['I', 'O', 'T', 'J', 'L', 'S', 'Z']
    
    def decode_action(self, action, piece_type,board_width=10):
        rotation = action // board_width
        left_target_col = action % board_width

        piece_shape = TetrisPiece.get_piece_shape(piece_type, rotation)

    # Find column index of leftmost '1' in the shape
        leftmost_offset = min([
            col for row in piece_shape for col, val in enumerate(row) if val == 1
        ])

        # Adjust so the piece’s leftmost block aligns with the desired column
        position = left_target_col - leftmost_offset

        return position, rotation
    
    def can_place_piece(self, board, piece_shape, position, board_height, board_width):
        """
        Check if a piece can be placed at the given position and simulate gravity.
        Returns (can_place, final_row) where final_row is where the piece lands.
        """
        piece_height = len(piece_shape)
        piece_width = len(piece_shape[0]) if piece_shape else 0

        # Convert board to 2D
        board_2d = np.array(board).reshape(board_height, board_width)

        # Check horizontal bounds
        if position < 0 or position + piece_width > board_width:
            return False, -1

        # Check if piece would go out of bounds vertically
        if piece_height > board_height:
            return False, -1

        # Try dropping from the top, row by row
        for drop_row in range(board_height - piece_height + 1):
            collision = False
            
            # Check if piece collides at this drop_row position
            for p_row in range(piece_height):
                for p_col in range(piece_width):
                    if piece_shape[p_row][p_col] == 1:  # Only check filled cells
                        b_row = drop_row + p_row
                        b_col = position + p_col
                        
                        # Check bounds (shouldn't happen due to earlier checks, but safety)
                        if b_row >= board_height or b_col >= board_width:
                            collision = True
                            break
                        
                        # Check collision with existing blocks
                        if board_2d[b_row, b_col] != 0:
                            collision = True
                            break
                
                if collision:
                    break
            
            if collision:
                # If we collide on the first row (top), piece can't be placed at all
                if drop_row == 0:
                    return False, -1
                else:
                    # Piece lands at the previous valid row
                    return True, drop_row - 1
        
        # If no collision occurred during the entire drop, piece lands at the bottom
        return True, board_height - piece_height


    def place_piece(self, board, piece_shape, position, row, board_height, board_width):
        """
        Place a piece on the board at the specified position and row.
        Returns the new board state.
        """
        board_2d = np.array(board).reshape(board_height, board_width)
        new_board = board_2d.copy()
        
        piece_height = len(piece_shape)
        piece_width = len(piece_shape[0]) if piece_shape else 0
        
        # Place the piece
        for p_row in range(piece_height):
            for p_col in range(piece_width):
                if piece_shape[p_row][p_col] == 1:  # Only place filled cells
                    board_row = row + p_row
                    board_col = position + p_col
                    
                    # Double-check bounds (should be guaranteed by can_place_piece)
                    if 0 <= board_row < board_height and 0 <= board_col < board_width:
                        new_board[board_row, board_col] = 1
        
        return new_board.flatten().tolist()
    
    def clear_lines(self, board, board_height, board_width):
        """Clear completed lines and return new board and lines cleared"""
        board_2d = np.array(board).reshape(board_height, board_width)
        lines_cleared = 0
        
        # Check each row from bottom to top
        new_board = []
        for row in range(board_height - 1, -1, -1):
            if np.all(board_2d[row] != 0):  # Line is complete
                lines_cleared += 1
            else:
                new_board.append(board_2d[row].tolist())
        
        # Add empty rows at the top
        while len(new_board) < board_height:
            new_board.append([0] * board_width)
        
        # Reverse to get correct order (top to bottom)
        new_board.reverse()
        
        return np.array(new_board).flatten().tolist(), lines_cleared
    
    def calculate_column_heights(self, board, board_height, board_width):
        """Calculate height of each column"""
        board_2d = np.array(board).reshape(board_height, board_width)
        heights = []
        
        for col in range(board_width):
            col_height = 0
            for row in range(board_height):
                if board_2d[row, col] != 0:
                    col_height = board_height - row
                    break
            heights.append(col_height)
        
        return heights
    
    def calculate_holes(self, board, board_height, board_width):
        """Calculate number of holes in the board"""
        holes = 0
        board_2d = np.array(board).reshape(board_height, board_width)
        
        for col in range(board_width):
            # Find the topmost filled cell in this column
            top_filled = -1
            for row in range(board_height):
                if board_2d[row, col] != 0:
                    top_filled = row
                    break
            
            # Count holes below the topmost filled cell
            if top_filled != -1:
                for row in range(top_filled + 1, board_height):
                    if board_2d[row, col] == 0:
                        holes += 1
        
        return holes
    
    def calculate_bumpiness(self, heights):
        """Calculate bumpiness (sum of height differences between adjacent columns)"""
        if len(heights) < 2:
            return 0
        
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        
        return bumpiness
    
   
class EnhancedDQNAgent:
    """Enhanced DQN Agent with proper action simulation"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 state_size=4, mem_size=10000, discount=0.95,
                 epsilon=1, epsilon_min=0, epsilon_stop_episode=0,
                 n_neurons=[32, 32], activations=['relu', 'relu', 'linear'],
                 loss='mse', optimizer='adam', replay_start_size=None,tensorboard_log_dir=None, modelFile=None):
        self.device = device
        self.state_size = state_size
        self.mem_size = mem_size
        self.memory = deque(maxlen=self.mem_size)
        self.discount=discount
        if epsilon_stop_episode > 0:
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = (self.epsilon - self.epsilon_min) / (epsilon_stop_episode)
        else:
            self.epsilon = 0

        self.n_neurons = n_neurons
        self.activations = activations
        self.loss = loss
        self.optimizer = optimizer
        if not replay_start_size:
            self.replay_start_size = self.mem_size / 2
        self.replay_start_size = replay_start_size
        
  
       
        self.piece_types = ['I', 'O', 'T', 'J', 'L', 'S', 'Z']

       
        # create a new model
        self.model = self._build_model()

        
        # TensorBoard writer
        if tensorboard_log_dir is None:
            tensorboard_log_dir = f"runs/enhanced_tetris_dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(tensorboard_log_dir)
        
       
        
        # Enhanced environment
        self.env = EnhancedTetrisEnvironment()
        
        print(f"Enhanced DQN Agent initialized on {device}")
        print(f"TensorBoard logs: {tensorboard_log_dir}")
    
    def simulate_action(self, state, action):
        """
        Enhanced action simulation with proper piece placement.
        Now also tags the resulting state with 'action'.
        """
        board_height = int(state.get('curriculumBoardHeight', 20))
        board_width  = 10

        # copy original board
        board = state.get('board', [0] * (board_height * board_width))[:]

        # figure out which piece we're placing
        piece_info = state.get('currentPiece', [2, 0, 0, 0])
        piece_index = int(piece_info[0])
        piece_types = self.env.piece_types  # ['I','O','T','S','Z','J','L']
        if 0 <= piece_index < len(piece_types):
            current_piece = piece_types[piece_index]
        else:
            current_piece = 'T'

        # decode the raw action id → (position, rotation)
        position, rotation = self.env.decode_action(action, current_piece,board_width)

        # get the shape at that rotation
        piece_shape = TetrisPiece.get_piece_shape(current_piece, rotation)

        # see if it even fits
        can_place, drop_row = self.env.can_place_piece(
            board, piece_shape, position, board_height, board_width
        )

        if not can_place:
            # invalid: copy old state + penalty flag + tag action
            sim = state.copy()
            sim.update({
                'linesCleared':   0,
                'invalid_action': True,
                'action':         action
            })
            return sim

        # place, clear lines, recompute features…
        new_board     = self.env.place_piece(board, piece_shape, position, drop_row, board_height, board_width)
        final_board, lines_cleared = self.env.clear_lines(new_board, board_height, board_width)
        new_heights   = self.env.calculate_column_heights(final_board, board_height, board_width)
        new_holes     = self.env.calculate_holes(final_board, board_height, board_width)
        new_bumpiness = self.env.calculate_bumpiness(new_heights)

        # build the new state dict and tag which action led to it
        simulated_state = {
            'board':                 final_board,
            'curriculumBoardHeight': board_height,
            'heights':               new_heights,
            'linesCleared':          lines_cleared,
            'holesCount':            new_holes,
            'bumpiness':             new_bumpiness,
            'validActions':          state.get('validActions', []),
            'currentPiece':          current_piece,
            'invalid_action':        False,
            'action':                action,
        }
        return simulated_state


    def actions_to_states(self, state, actions=None):
        """
        Given a state and (optionally) a list of actions,
        return the list of resulting simulated states.
        If no actions list is provided, uses state['validActions'].
        """
        if actions is None:
            actions = state.get('validActions', [])
        return [ self.simulate_action(state, a) for a in actions ]


    def states_to_actions(self, simulated_states):
        """
        Given a list of simulated states (as from actions_to_states),
        return the list of actions that produced them.
        """
        return [ s.get('action') for s in simulated_states ]
    

    def _build_model(self):
        '''Builds a Keras deep neural network model'''
        model = Sequential()
        model.add(Dense(self.n_neurons[0], input_dim=self.state_size, activation=self.activations[0]))

        for i in range(1, len(self.n_neurons)):
            model.add(Dense(self.n_neurons[i], activation=self.activations[i]))

        model.add(Dense(1, activation=self.activations[-1]))

        model.compile(loss=self.loss, optimizer=self.optimizer)
        
        return model



    def state_to_action(self, simulated_state):
        """
        Given a single simulated state, return its action.
        """
        return simulated_state.get('action')
    
    def evaluate_action(self, state, action):
        """
        Simulate the given action and return the four raw features:
          [lines_cleared, holes, total_bumpiness, sum_of_heights]
        If the action is invalid, returns [0,0,0,0] as a penalty.
        """
        sim = self.simulate_action(state, action)

        if sim.get('invalid_action', False):
            return None

        lines_cleared  = sim.get('linesCleared', 0)
        holes          = sim.get('holesCount', 0)
        total_bumpiness= sim.get('bumpiness', 0)
        heights        = sim.get('heights', [])
        sum_heights    = sum(heights) if heights else 0

        return [lines_cleared, holes, total_bumpiness, sum_heights]
    
    def get_possible_states(self, current_state):
        """Generate all possible next states for valid actions (rotation-aware)."""
        piece_index = int(current_state['currentPiece'][0])
        piece_type = self.piece_types[piece_index]

        board_width = 10
        board_height = int(current_state.get('curriculumBoardHeight', 20))
        board = current_state.get('board', [0] * (board_height * board_width))
        
        valid_actions = []
        possible_states = []

        # Use correct number of rotations from TetrisPiece.PIECES
        rotation_range = len(TetrisPiece.PIECES[piece_type])

        for rotation in range(rotation_range):
            shape = TetrisPiece.get_piece_shape(piece_type, rotation)
            piece_width = len(shape[0]) if shape else 0
            piece_height = len(shape)

            # Find the leftmost block offset in the piece
            leftmost_offset = float('inf')
            for row in shape:
                for col, val in enumerate(row):
                    if val == 1:
                        leftmost_offset = min(leftmost_offset, col)
            
            # If no blocks found (shouldn't happen), skip
            if leftmost_offset == float('inf'):
                continue

            # Generate actions for each possible left target column
            for left_target_col in range(board_width):
                # Calculate actual position where piece will be placed
                position = left_target_col - leftmost_offset
                
                # Check if piece fits horizontally
                if position < 0 or (position + piece_width) > board_width:
                    continue
                
                # Check if piece can be placed (including gravity simulation)
                can_place, drop_row = self.env.can_place_piece(
                    board, shape, position, board_height, board_width
                )
                
                if not can_place:
                    continue
                
                # Generate action ID: rotation * board_width + left_target_col
                action = rotation * board_width + left_target_col
                valid_actions.append(action)
                
                # Evaluate the action to get features
                features = self.evaluate_action(current_state, action)
                if features is not None:  # Only add if action is valid
                    possible_states.append((action, features))

        return possible_states
    
    def predict_value(self,state):
        return self.model.predict(state,verbose=0)[0]
    
    def best_state(self,states,steps):
        max_value =None
        best_state =None
        self.writer.add_scalar("score/epsilon",self.epsilon,steps)
        if states is None:
            return None
        if random.random() <= self.epsilon:
            return random.choice(list(states))
        else:
            for state in states:
                value = self.predict_value(np.reshape(state,[1,self.state_size]))
                if not max_value or value> max_value:
                    max_value = value
                    best_state = state
        return best_state       
    def add_to_memory(self, current_state, next_state, reward, done):
        '''Adds a play to the replay memory buffer'''
        self.memory.append((current_state, next_state, reward, done))      
    
    def random_value(self):
        '''Random score for a certain action'''
        return random.random()
  
    def act(self, state):
        '''Returns the expected score of a certain state'''
        state = np.reshape(state, [1, self.state_size])
        if random.random() <= self.epsilon:
            return self.random_value()
        else:
            return self.predict_value(state)

 

    def train(self, batch_size=32, epochs=3):
        '''Trains the agent'''
        if batch_size > self.mem_size:
            print('WARNING: batch size is bigger than mem_size. The agent will not be trained.')

        n = len(self.memory)
    
        if n >= self.replay_start_size and n >= batch_size:

            batch = random.sample(self.memory, batch_size)

            # Get the expected score for the next states, in batch (better performance)
            next_states = np.array([x[1] for x in batch])
            next_qs = [x[0] for x in self.model.predict(next_states)]

            x = []
            y = []

            # Build xy structure to fit the model in batch (better performance)
            for i, (state, _, reward, done) in enumerate(batch):
                if not done:
                    # Partial Q formula
                    new_q = reward + self.discount * next_qs[i]
                else:
                    new_q = reward

                x.append(state)
                y.append(new_q)

            # Fit the model to the given values
            self.model.fit(np.array(x), np.array(y), batch_size=batch_size, epochs=epochs, verbose=0)
            # Update the exploration variable
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay    
    
    def end_episode(self, episode_reward, episode_length, episode_score):
        """Called at the end of each episode"""
        self.episodes += 1
        
        # Log episode metrics
        self.writer.add_scalar('Episode/Reward', episode_reward, self.episodes)
        self.writer.add_scalar('Episode/Length', episode_length, self.episodes)
        self.writer.add_scalar('Episode/Score', episode_score, self.episodes)
        
        print(f"Episode {self.episodes}: Score={episode_score}, Reward={episode_reward:.2f}, ")
    
    def save_model(self, name):
        '''Saves the current model.
        It is recommended to name the file with the ".keras" extension.'''
        self.model.save(name)
    
    def load(self, filepath):
        """Load the model"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.episodes = checkpoint.get('episodes', 0)
            self.steps = checkpoint.get('steps', 0)
            print(f"Model loaded from {filepath}")
            return True
        return False
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()


    def log_episode_metrics(self, episode, episode_reward, episode_length, episode_score, 
                          episode_lines, game_metrics):
        """Log episode metrics to TensorBoard"""
        # FIX 4: Increment episode counter
        self.training_episodes += 1
        
        self.writer.add_scalar('Episode/Reward', episode_reward, episode)
        self.writer.add_scalar('Episode/Length', episode_length, episode)
        self.writer.add_scalar('Episode/Score', episode_score, episode)
        self.writer.add_scalar('Episode/Lines_Cleared', episode_lines, episode)
        
        # Game-specific metrics
        self.writer.add_scalar('Game/Holes_Count', game_metrics.get('holes_count', 0), episode)
        self.writer.add_scalar('Game/Stack_Height', game_metrics.get('stack_height', 0), episode)
        self.writer.add_scalar('Game/Perfect_Clear', 1 if game_metrics.get('perfect_clear', False) else 0, episode)
        
        # Running averages
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        if len(self.episode_rewards) >= 100:
            avg_reward_100 = np.mean(self.episode_rewards[-100:])
            avg_length_100 = np.mean(self.episode_lengths[-100:])
            self.writer.add_scalar('Episode/Avg_Reward_100', avg_reward_100, episode)
            self.writer.add_scalar('Episode/Avg_Length_100', avg_length_100, episode)
        
        if len(self.episode_rewards) >= 10:
            avg_reward_10 = np.mean(self.episode_rewards[-10:])
            avg_length_10 = np.mean(self.episode_lengths[-10:])
            self.writer.add_scalar('Episode/Avg_Reward_10', avg_reward_10, episode)
            self.writer.add_scalar('Episode/Avg_Length_10', avg_length_10, episode)        


