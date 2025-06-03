import socket
import json
import numpy as np
import time
import threading
from queue import Queue

class UnityTetrisClient:
    def __init__(self, host='127.0.0.1', port=12345):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.game_state_queue = Queue()
        self.running = False
        self.action_space_size = 40
        self.board_width = 10
        self.num_rotations = 4
        
    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            self.running = True
            
            self.receive_thread = threading.Thread(target=self._receive_loop)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            
            print(f"Connected to Unity at {self.host}:{self.port}")
            print(f"Action space: {self.action_space_size} actions ({self.board_width} columns × {self.num_rotations} rotations)")
            return True
            
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def _receive_loop(self):
        buffer = ""
        while self.running and self.connected:
            try:
                data = self.socket.recv(4096).decode('utf-8')
                if not data:
                    break
                    
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            game_state = json.loads(line)
                            self.game_state_queue.put(game_state)
                        except json.JSONDecodeError as e:
                            print(f"JSON decode error: {e}")
                            
            except Exception as e:
                print(f"Receive error: {e}")
                break
                
        self.connected = False
    
    def send_action(self, action_index):
        """Send action to Unity
        action_index: 0-39 representing column (0-9) and rotation (0-3)
        Formula: action = column * 4 + rotation
        """
        if not self.connected:
            return False
            
        if action_index < 0 or action_index >= self.action_space_size:
            print(f"Invalid action index: {action_index}. Must be 0-{self.action_space_size-1}")
            return False
        
        # Decode action for logging
        column = action_index // 4
        rotation = action_index % 4
        
        command = {
            "type": "action",
            "action": {
                "actionIndex": action_index
            }
        }
        
        success = self._send_command(command)
        if success:
            print(f"Sent action {action_index}: Place at column {column} with rotation {rotation}")
        
        return success
    
    def action_to_column_rotation(self, action_index):
        """Convert action index to column and rotation"""
        column = action_index // 4
        rotation = action_index % 4
        return column, rotation
    
    def column_rotation_to_action(self, column, rotation):
        """Convert column and rotation to action index"""
        if column < 0 or column >= self.board_width or rotation < 0 or rotation >= self.num_rotations:
            raise ValueError(f"Invalid column {column} or rotation {rotation}")
        return column * 4 + rotation
    
    def get_valid_actions(self, game_state):
        """Get list of all valid actions (always all 40 for now, but can be filtered based on game state)"""
        # For now, all actions are always valid
        # You could add logic here to filter invalid placements
        return list(range(40))
    
    def send_curriculum_change(self, board_height=20, board_preset=0, tetromino_types=7):
        """Send curriculum parameters to Unity"""
        command = {
            "type": "curriculum_change",
            "curriculum": {
                "boardHeight": board_height,
                "boardPreset": board_preset,
                "allowedTetrominoTypes": tetromino_types
            }
        }
        
        return self._send_command(command)
    
    def send_reset(self):
        """Reset the game"""
        command = {
            "type": "reset",
            "reset": {
                "resetBoard": True
            }
        }
        
        return self._send_command(command)
    
    def _send_command(self, command):
        if not self.connected:
            return False
            
        try:
            message = json.dumps(command)
            self.socket.send(message.encode('utf-8'))
            return True
        except Exception as e:
            print(f"Send error: {e}")
            return False
    
    def get_game_state(self, timeout=1.0):
        """Get the latest game state"""
        try:
            return self.game_state_queue.get(timeout=timeout)
        except:
            return None
    
    def get_board_state(self, game_state):
        """Extract and reshape board state"""
        board_flat = game_state.get('board', [])
        if len(board_flat) == 200:  # 10x20 board
            return np.array(board_flat).reshape(20, 10)
        else:
            # Dynamic board size based on curriculum
            height = len(board_flat) // 10
            return np.array(board_flat).reshape(height, 10)
    
    def get_current_piece_info(self, game_state):
        """Extract current piece information"""
        piece_info = game_state.get('currentPiece', [0, 0, 0, 0])
        return {
            'type': piece_info[0],
            'rotation': piece_info[1],
            'x': piece_info[2],
            'y': piece_info[3]
        }
    
    def disconnect(self):
        self.running = False
        self.connected = False
        if self.socket:
            self.socket.close()

# Example with intelligent action selection
def main():
    client = UnityTetrisClient()
    
    if not client.connect():
        return
    
    try:
        print("Starting training loop with direct placement actions...")
        episode = 0
        step_count = 0
        
        while True:
            state = client.get_game_state()
            if state is None:
                continue
            
            # Only process if waiting for action
            if not state.get('waitingForAction', False):
                continue
                
            print(f"Episode {episode}, Step {step_count}, Score: {state.get('score', 0)}")
            
            if state.get('gameOver', False):
                print(f"Episode {episode} ended with score {state.get('score', 0)}. Resetting...")
                client.send_reset()
                episode += 1
                step_count = 0
                
                # Progressive curriculum
                if episode % 10 == 0:
                    new_height = min(20, 10 + episode // 10)
                    client.send_curriculum_change(board_height=new_height)
                    print(f"Changed board height to {new_height}")
                
            else:
                # Select action using your policy
                action = select_action_heuristic(client, state)
                client.send_action(action)
                step_count += 1
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        client.disconnect()

def select_action_heuristic(client, state):
    """Heuristic action selection - replace with your RL policy"""
    import random
    
    # Get board and piece info
    board = client.get_board_state(state)
    piece_info = client.get_current_piece_info(state)
    
    # Simple heuristic: try to place piece in lowest position
    best_action = 0
    best_score = float('-inf')
    
    for action in range(40):
        column, rotation = client.action_to_column_rotation(action)
        
        # Simple scoring: prefer lower columns in center, avoid high stacks
        column_height = get_column_height(board, column)
        
        score = 0
        
        # Prefer center columns
        score += 10 - abs(column - 4.5)
        
        # Penalty for high stacks
        score -= column_height * 2
        
        # Small random factor
        score += random.random()
        
        if score > best_score:
            best_score = score
            best_action = action
    
    return best_action

def get_column_height(board, column):
    """Get the height of pieces in a specific column"""
    height = 0
    for row in range(len(board)):
        if board[row][column] == 1:
            height = len(board) - row
            break
    return height

# Example for RL training integration
class TetrisEnvironment:
    def __init__(self):
        self.client = UnityTetrisClient()
        self.connected = False
    
    def connect(self):
        self.connected = self.client.connect()
        return self.connected
    
    def reset(self):
        """Reset environment and return initial observation"""
        if not self.connected:
            return None
            
        self.client.send_reset()
        
        # Wait for initial state
        for _ in range(50):  # Max 5 seconds
            state = self.client.get_game_state(timeout=0.1)
            if state and state.get('waitingForAction', False):
                return self.state_to_observation(state)
        
        return None
    
    def step(self, action):
        """Take action and return (observation, reward, done, info)"""
        if not self.connected:
            return None, 0, True, {}
        
        # Send action
        success = self.client.send_action(action)
        if not success:
            return None, 0, True, {}
        
        # Wait for result
        for _ in range(100):  # Max 10 seconds
            state = self.client.get_game_state(timeout=0.1)
            if state:
                obs = self.state_to_observation(state)
                reward = state.get('reward', 0)
                done = state.get('gameOver', False)
                info = {
                    'score': state.get('score', 0),
                    'lines_cleared': state.get('linesCleared', 0),
                    'holes': state.get('holesCount', 0),
                    'stack_height': state.get('stackHeight', 0)
                }
                
                # If waiting for action or game over, return result
                if state.get('waitingForAction', False) or done:
                    return obs, reward, done, info
        
        # Timeout
        return None, 0, True, {}
    
    def state_to_observation(self, state):
        """Convert game state to observation for RL agent"""
        board = np.array(state.get('board', []))
        current_piece = np.array(state.get('currentPiece', [0, 0, 0, 0]))
        next_piece = np.array(state.get('nextPiece', [0]))
        
        # You can customize this based on your RL algorithm's needs
        return {
            'board': board,
            'current_piece': current_piece,
            'next_piece': next_piece,
            'score': state.get('score', 0),
            'holes': state.get('holesCount', 0),
            'stack_height': state.get('stackHeight', 0)
        }
    
    def close(self):
        self.client.disconnect()

if __name__ == "__main__":
    main()