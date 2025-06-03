# tetris_client.py
import socket
import json
import numpy as np
import time
import threading
from queue import Queue
import random
from collections import deque
import pickle
import os

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
        
    def connect(self, max_retries=5, retry_delay=2.0):
        """Connect to Unity with retries"""
        for attempt in range(max_retries):
            try:
                print(f"Attempting to connect to Unity... (attempt {attempt + 1}/{max_retries})")
                
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(10.0)
                self.socket.connect((self.host, self.port))
                
                self.connected = True
                self.running = True
                
                self.receive_thread = threading.Thread(target=self._receive_loop)
                self.receive_thread.daemon = True
                self.receive_thread.start()
                
                print(f"✓ Connected to Unity at {self.host}:{self.port}")
                return True
                
            except Exception as e:
                print(f"✗ Connection failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                
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
                if self.running:
                    print(f"Receive error: {e}")
                break
                
        self.connected = False
    
    def send_action(self, action_index):
        if not self.connected:
            return False
            
        if action_index < 0 or action_index >= self.action_space_size:
            return False
        
        command = {
            "type": "action",
            "action": {"actionIndex": action_index}
        }
        
        return self._send_command(command)
    
    def send_curriculum_change(self, board_height=20, board_preset=0, tetromino_types=7):
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
        command = {
            "type": "reset",
            "reset": {"resetBoard": True}
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
        try:
            return self.game_state_queue.get(timeout=timeout)
        except:
            return None
    
    def get_board_state(self, game_state):
        board_flat = game_state.get('board', [])
        if len(board_flat) == 200:  # 10x20 board
            return np.array(board_flat).reshape(20, 10)
        else:
            height = len(board_flat) // 10
            return np.array(board_flat).reshape(height, 10)
    
    def get_current_piece_info(self, game_state):
        piece_info = game_state.get('currentPiece', [0, 0, 0, 0])
        return {
            'type': piece_info[0],
            'rotation': piece_info[1],
            'x': piece_info[2],
            'y': piece_info[3]
        }
    
    def action_to_column_rotation(self, action_index):
        column = action_index // 4
        rotation = action_index % 4
        return column, rotation
    
    def disconnect(self):
        self.running = False
        self.connected = False
        if self.socket:
            self.socket.close()