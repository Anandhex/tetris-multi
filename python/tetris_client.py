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
#12347 without noise
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
    
    def get_curriculum_info(self, game_state):
        """Extract curriculum information from game state"""
        return {
            'board_height': game_state.get('curriculumBoardHeight', 20),
            'board_preset': game_state.get('curriculumBoardPreset', 0),
            'allowed_tetromino_types': game_state.get('allowedTetr ominoTypes', 7)
        }
    
    def is_game_over(self, game_state):
        """Check if the game is over"""
        return game_state.get('gameOver', False) or game_state.get('episodeEnd', False)
    
    def get_action_space_info(self, game_state):
        """Get action space information from game state"""
        return {
            'action_space_size': game_state.get('actionSpaceSize', 40),
            'action_space_type': game_state.get('actionSpaceType', 'column_rotation'),
            'is_executing_action': game_state.get('isExecutingAction', False),
            'waiting_for_action': game_state.get('waitingForAction', False)
        }
    
    def get_board_metrics(self, game_state):
        """Get board analysis metrics"""
        return {
            'holes_count': game_state.get('holesCount', 0),
            'stack_height': game_state.get('stackHeight', 0),
            'perfect_clear': game_state.get('perfectClear', False),
            'lines_cleared': game_state.get('linesCleared', 0)
        }
    
    def wait_for_game_ready(self, timeout=10.0, check_interval=0.1):
        """Wait until the game is ready to receive actions"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            state = self.get_game_state(timeout=check_interval)
            if state:
                action_info = self.get_action_space_info(state)
                if action_info['waiting_for_action'] and not action_info['is_executing_action']:
                    return state
            time.sleep(check_interval)
        
        return None
    
    def send_action_and_wait(self, action, timeout=5.0):
        """Send action and wait for the result state"""
        if not self.send_action(action):
            return None
        
        # Wait for the action to complete and get the resulting state
        start_time = time.time()
        while time.time() - start_time < timeout:
            state = self.get_game_state(timeout=0.1)
            if state:
                return state
        
        return None
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
    
    def send_action(self, action):
        if not self.connected:
            return False
            
        
        command = {
            "type": "action",
            "action": {"col": action["col"],"rot":action["rot"]}
        }
        
        return self._send_command(command)
    
    def send_curriculum_change(self, board_height=20, board_preset=0, tetromino_types=7,stage_name=None):
        command = {
            "type": "curriculum_change",
            "curriculum": {
                "boardHeight": board_height,
                "boardPreset": board_preset,
                "allowedTetrominoTypes": tetromino_types,
                "stageName": stage_name or "Unknown",
                "timestamp": time.time()
            }
        }
        return self._send_command(command)
    def send_curriculum_status_request(self):
        """Request current curriculum status from Unity"""
        command = {
            "type": "curriculum_status_request"
        }
        return self._send_command(command)
    
    def get_curriculum_confirmation(self, timeout=2.0):
        """Wait for curriculum change confirmation"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            state = self.get_game_state(timeout=0.1)
        if state and 'curriculumConfirmed' in state:
                return state
        return None
    # def send_reset(self):
    #     command = {
    #         "type": "reset",
    #         "reset": {"resetBoard": True}
    #     }
    #     return self._send_command(command)
    
    # def _send_command(self, command):
    #     if not self.connected:
    #         return False
            
    #     try:
    #         message = json.dumps(command)
    #         self.socket.send(message.encode('utf-8'))
    #         return True
    #     except Exception as e:
    #         print(f"Send error: {e}")
    #         return False
    
    def get_game_state(self, timeout=1.0):
        try:
            return self.game_state_queue.get(timeout=timeout)
        except:
            return None
    
    def get_board_state(self, timeout=2.0, poll_interval=0.05):
        """
        Ask Unity for all valid (col,rot) placements and their [lines, holes, bumpiness, height].
        Blocks until it receives a message of type 'possible_states' or times out.
        Returns a dict { 'col:rot': [lines, holes, bumpiness, height], ... }
        """
        # 1) send the request
        cmd = { "type": "board_state" }
        if not self._send_command(cmd):
            return {}

        # 2) drain queue until we find the response
        start = time.time()
        while time.time() - start < timeout:
            msg = self.get_game_state(timeout=poll_interval)
            print(msg)
            if not msg:
                continue
            # we expect Unity to send back {"type":"possible_states","payload":{...}}
            if msg.get("type") == "possible_states" and "payload" in msg:
                return msg["payload"]
        # timed out
        return {}
    
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

    def send_reset(self):
        """Send reset command to Unity to reset the board"""
        if not self.connected:
            return False
        command = {"type": "reset", "reset": {"resetBoard": True, "clearPowerups": True}}
        return self._send_command(command)
    
    def get_possible_states(self, timeout=2.0, poll_interval=0.05):
        """
        Ask Unity for all valid (col,rot) placements and their [lines, holes, bumpiness, height].
        Blocks until it receives a message of type 'possible_states' or times out.
        Returns a dict { 'col:rot': [lines, holes, bumpiness, height], ... }
        """
        # 1) send the request
        cmd = { "type": "request_states" }
        if not self._send_command(cmd):
            return {}

        # 2) drain queue until we find the response
        start = time.time()
        while time.time() - start < timeout:
            msg = self.get_game_state(timeout=poll_interval)
            if not msg:
                continue
            # we expect Unity to send back {"type":"possible_states","payload":{...}}
            if msg.get("type") == "possible_states" and "payload" in msg:
                return msg["payload"]
        # timed out
        return {}
    

    def env_reset(self, timeout=10.0, check_interval=0.1):
        """
        Resets the game board in Unity and waits until the environment is ready for actions.
        Returns the initial game state dict.
        """
        print(f"🐛 DEBUG: Starting environment reset...")
        # Send reset signal
        if not self.send_reset():
            raise RuntimeError("Failed to send reset command to Unity.")
            
        # Clear any old game states from queue
        old_states_cleared = 0
        while not self.game_state_queue.empty():
            try:
                self.game_state_queue.get_nowait()
                old_states_cleared += 1
            except:
                break
        
        if old_states_cleared > 0:
            print(f"🐛 DEBUG: Cleared {old_states_cleared} old game states from queue")
                    
        # Wait for Unity to finish resetting and be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            state = self.get_game_state(timeout=check_interval)
            if state:
                if state.get('type') == 'reset_confirmed':
                    print(f"🐛 DEBUG: Reset confirmed by Unity")
                    continue
                # Check if ready for new action
                action_info = state.get('waitingForAction', False)
                executing = state.get('isExecutingAction', False)
                if action_info and not executing:
                    return state
            time.sleep(check_interval)
        raise TimeoutError("Timeout waiting for game reset and ready state.")          
    
    def execute_powerup_decision(self, decision_result, timeout=5.0):
        """
        Execute powerup decision with enhanced debugging
        """
        decision_data = decision_result['decision_data']
        action = decision_data['action']
        
        print(f"🐛 DEBUG: Executing powerup decision:")
        print(f"   Action: {action}")
        print(f"   PowerUp Type: {decision_data['powerup_type']}")
        print(f"   Q-Value: {decision_result['q_value']:.3f}")
        
        if action == 'wait':
            command = {
                "type": "hold_powerup",
                "powerup_type": decision_data['powerup_type'],
                "ai_confidence": decision_result['q_value'],
                "timestamp": time.time()
            }
            print(f"🐛 DEBUG: Sending hold command: {json.dumps(command, indent=2)}")
            
        elif action == 'use_bomb':
            command = {
                "type": "execute_bomb_drop",
                "bomb": {
                    "column": decision_data['column'],
                    "predicted_impact": decision_data['impact'],
                    "ai_confidence": decision_result['q_value'],
                    "timestamp": time.time()
                }
            }
            print(f"🐛 DEBUG: Sending bomb command: {json.dumps(command, indent=2)}")
            
        elif action == 'use_gravity':
            command = {
                "type": "execute_gravity",
                "gravity": {
                    "predicted_impact": decision_data['impact'],
                    "ai_confidence": decision_result['q_value'],
                    "timestamp": time.time()
                }
            }
            print(f"🐛 DEBUG: Sending gravity command: {json.dumps(command, indent=2)}")
            
        elif action == 'use_bottom_clear':
            command = {
                "type": "execute_bottom_clear",
                "bottom_clear": {
                    "predicted_impact": decision_data['impact'],
                    "ai_confidence": decision_result['q_value'],
                    "timestamp": time.time()
                }
            }
            print(f"🐛 DEBUG: Sending bottom clear command: {json.dumps(command, indent=2)}")
        
        else:
            print(f"🐛 ERROR: Unknown action: {action}")
            return {'success': False, 'error': f'Unknown action: {action}'}
        
        # Send command with debugging
        print(f"🐛 DEBUG: Attempting to send command...")
        if not self._send_command(command):
            print(f"🐛 ERROR: Failed to send command to Unity!")
            return {'success': False, 'error': 'Failed to send command'}
        
        print(f"🐛 DEBUG: Command sent successfully, waiting for response...")
        
        # Wait for response with detailed logging
        start_time = time.time()
        response_count = 0
        
        while time.time() - start_time < timeout:
            state = self.get_game_state(timeout=0.1)
            if state:
                response_count += 1
                print(f"🐛 DEBUG: Received response #{response_count}: type='{state.get('type', 'UNKNOWN')}'")
                
                # Check for expected response types
                expected_types = {
                    'use_bomb': 'bomb_executed',
                    'use_gravity': 'gravity_executed', 
                    'use_bottom_clear': 'bottom_clear_executed'
                }
                
                expected_type = expected_types.get(action)
                if expected_type and state.get('type') == expected_type:
                    print(f"🐛 DEBUG: Found expected response type: {expected_type}")
                    print(f"🐛 DEBUG: Response content: {json.dumps(state, indent=2)}")
                    
                    return {
                        'success': state.get('success', False),
                        'action': action,
                        'powerup_type': decision_data['powerup_type'],
                        'impact_metrics': state.get('impact_metrics', {}),
                        'board_state_before': state.get('board_before'),
                        'board_state_after': state.get('board_after'),
                        'ui_updates': state.get('ui_updates', {}),
                        'error': state.get('error', None)
                    }
                else:
                    # Log unexpected responses
                    print(f"🐛 DEBUG: Unexpected response type. Expected: {expected_type}, Got: {state.get('type')}")
                    if 'error' in state:
                        print(f"🐛 DEBUG: Error in response: {state['error']}")
        
        print(f"🐛 ERROR: Timeout after {timeout}s. Received {response_count} responses but none matched expected type.")
        return {'success': False, 'error': f'Timeout waiting for {action} execution'}

    # Also add debugging to _send_command method
    def _send_command(self, command):
        """Enhanced _send_command with debugging"""
        if not self.connected:
            print("🐛 ERROR: Not connected to Unity!")
            return False
            
        try:
            message = json.dumps(command)

            # Add newline terminator for Unity parsing
            message_with_newline = message + '\n'            
            print(f"🐛 DEBUG: Sending {len(message_with_newline)} bytes to Unity")
            # print(f"🐛 DEBUG: Sending {len(message)} bytes to Unity")
            print(f"🐛 DEBUG: Raw message: {message}")
            
            self.socket.send(message.encode('utf-8'))
            print(f"🐛 DEBUG: Message sent successfully")
            return True
            
        except Exception as e:
            print(f"🐛 ERROR: Send error: {e}")
            print(f"🐛 DEBUG: Connection status: connected={self.connected}")
            return False 