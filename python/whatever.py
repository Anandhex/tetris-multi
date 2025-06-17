from simplified_dqn_network import EnhancedDQNAgent
from datetime import datetime
from tetris_client import UnityTetrisClient
from statistics import mean
from tqdm import tqdm
import logging
import numpy as np
import sys
import json
import os
import time


class TetrisTrainer:
    def __init__(self,
                 agent_type: str = 'dqn',
                 load_model: bool = False,
                 model_path: str =None,
                 tensorboard_log_dir: str = None,
                 score_window_size: int = 100):
        # Unity client for Tetris
        self.client = UnityTetrisClient()

        # Training hyperparameters
        self.episodes = 3000
        self.max_steps = None
        self.epsilon_stop_episode = 2000
        self.mem_size = 1000
        self.discount = 0.95
        self.batch_size = 128
        self.epochs = 1
        self.render_every = 50
        self.render_delay = None
        self.log_every = 50
        self.replay_start_size = 1000
        self.train_every = 1
        self.n_neurons = [32, 32, 32]
        self.activations = ['relu', 'relu', 'relu', 'linear']
        self.save_best_model = True
        self.start_at = f"model_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.model_path = model_path if model_path is not None else f"model_{datetime.now().strftime('%Y%m%d-%H%M%S')}.h5" 
        self.BOARD_HEIGHT = 20
        self.BOARD_WIDTH = 10

        # Logging and checkpoint
        self.start_episode = 1
        self.best_score = -float('inf')
        self.total_steps = 0

        # Build tensorboard path
        log_dir = tensorboard_log_dir or (
            f"logs/{self.start_at}/tetris-nn={self.n_neurons}-mem={self.mem_size}-bs={self.batch_size}-"
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

        # Initialize DQN agent
        self.agent = EnhancedDQNAgent(
            n_neurons=self.n_neurons,
            activations=self.activations,
            epsilon_stop_episode=self.epsilon_stop_episode,
            mem_size=self.mem_size,
            discount=self.discount,
            replay_start_size=self.replay_start_size,
            tensorboard_log_dir=log_dir,
            epsilon=0
        )

        # Load pretrained weights if requested
        print(self.model_path)
        if load_model and os.path.exists(self.model_path):
            self.agent.model.load_weights(self.model_path)
            print(f"Loaded model weights from {self.model_path}")
        else:
            print("model not loaded")    

    def ensure_connection(self, retries: int = 5, delay: float = 2.0):
        for _ in range(retries):
            state = self.client.wait_for_game_ready(timeout=5.0)
            print(state)
            if state is not None:
                return True
            time.sleep(delay)
        raise ConnectionError("Unable to connect to Unity Tetris client.")

    def calculate_reward(self, prev_meta: dict, curr_meta: dict, action: int) -> float:
        # Basic reward: block reward + line clear bonus - game over penalty
        lines_prev = prev_meta.get('linesCleared', 0) if prev_meta else 0
        lines_now = curr_meta.get('linesCleared', 0)
        new_lines = max(0, lines_now - lines_prev)
        reward = 1 + (new_lines ** 2) * self.BOARD_WIDTH
        if curr_meta.get('gameOver', False):
            reward -= 2
        return reward

    def train(self):
        scores = []
        # Connect to Unity Tetris client
        self.client.connect()

        # Progress bar for valid episodes only

        # Loop until we've completed the desired number of valid episodes
        for episode in tqdm(range(self.episodes)):
            # Start a new game
            self.client.env_reset()
            done = False
            steps = 0
            episode_reward = 0.0
            episode_score = 0
            current_state = [0,0,0,0]

            # Play out one episode
            while not done and (self.max_steps is None or steps < self.max_steps):
                # Select an action
                next_states = self.client.get_possible_states()
                if not next_states:
                    action = 0
                else:
                    # 3) Build a map from feature‐tuples → actionIndex
                    # Keys in next_states are strings "col:rot"
                    # Values are lists [lines, holes, bumpiness, height]
                    action_map = {}
                    for key, feats in next_states.items():
                        col_str, rot_str = key.split(":")
                        col_i, rot_i = int(col_str), int(rot_str)
                        action_map[tuple(feats)] = (col_i, rot_i)
                feature_list = list(action_map.keys())
                if not next_states:
                    curr_meta = self.client.send_action_and_wait({"col":-1,"rot":-1}, timeout=30.0)
                    if curr_meta is None:

                        print(f"Episode {episode}: timeout, skipping step")
                        break

                    done = curr_meta.get('gameOver', False)
                    # Compute and store reward
                    reward = curr_meta.get('reward',0)
                    self.agent.add_to_memory(
                        current_state,
                        [0,0,0,0],
                        reward, done
                    )
                else:
                    best_state = self.agent.best_state(feature_list, episode)
                    col, rot = action_map[tuple(best_state)]     
                    # Send action and receive new state
                    print(next_states)
                    print(best_state)
                    curr_meta = self.client.send_action_and_wait({"col":col,"rot":rot}, timeout=30.0)
                    if curr_meta is None:
                        print(f"Episode {episode}: timeout, skipping step")
                        break
                    input()    
                    done = curr_meta.get('gameOver', False)
                    # Compute and store reward
                    reward = curr_meta.get('reward',0)
                    self.agent.add_to_memory(
                        current_state,
                        best_state,
                        reward, done
                    )

                # Accumulate metrics
                episode_reward += reward
                episode_score = curr_meta.get('reward', 0)
                current_state = best_state
                steps += 1
                self.total_steps += 1

            # Skip any episode that ended in <= 6 steps
            if steps <= 6:
                print(done,curr_meta,reward)
                logging.info(f"Skipping episode {episode} (only {steps} steps)")
                continue

            # Valid episode: record metrics
            scores.append(episode_reward)
            self.agent.writer.add_scalar('Episode/Reward', episode_reward, episode)
            self.agent.writer.add_scalar('Episode/Score', episode_score, episode)
            logging.info(f"Episode {episode} ended: reward={episode_reward}, score={episode_score}, steps={steps}")

            # Train the agent
            if episode % self.train_every == 0:
                self.agent.train(batch_size=self.batch_size, epochs=self.epochs)

            # Log running stats
            if episode % self.log_every == 0:
                window = scores[-self.log_every:]
                self.agent.writer.add_scalar('Stats/AvgReward', mean(window), episode)
                self.agent.writer.add_scalar('Stats/MinReward', min(window), episode)
                self.agent.writer.add_scalar('Stats/MaxReward', max(window), episode)

            # Save best model if improved
            if self.save_best_model and episode_reward > self.best_score:
                self.best_score = episode_reward
                fname = f"{self.start_at}/v/best_{datetime.now().strftime('%Y%m%d-%H%M%S')}.h5"
                self.agent.save_model(fname)
                print(f"New best model ({episode_reward:.1f}) saved to {fname}")

            # Periodic checkpoint
            if episode % 50 == 0:
                self.agent.save_model(f"{self.start_at}/{self.model_path}")
                with open(f'{self.start_at}/checkpoint.json', 'w') as f:
                    json.dump({'episode': episode, 'total_steps': self.total_steps, 'best_score': self.best_score}, f)

            # Advance episode counter and progress bar
            episode += 1

        # Cleanup
        self.agent.close()
        self.client.disconnect()
        print("Training complete.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    trainer = TetrisTrainer(load_model=True,model_path='/Users/anandpatil/Documents/Projects/tetris-multi/python/model_20250616-071507/model_20250616-071507.h5')
    # trainer = TetrisTrainer(load_model=True)
    trainer.train()
