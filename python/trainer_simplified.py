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
    def __init__(self, agent_type='dqn', load_model=False, model_path='tetris_model.weights.h5', 
                 tensorboard_log_dir=None, score_window_size=100):

        # Environment and hyperparameters
        self.client = UnityTetrisClient()
        self.episodes = 3000 # total number of episodes
        self.max_steps = None # max number of steps per game (None for infinite)
        self.epsilon_stop_episode = 2000 # at what episode the random exploration stops
        self.mem_size = 1000 # maximum number of steps stored by the agent
        self.discount = 0.95 # discount in the Q-learning formula (see DQNAgent)
        self.batch_size = 128 # number of actions to consider in each training
        self.epochs = 1 # number of epochs per training
        self.render_every = 50 # renders the gameplay every x episodes
        self.render_delay = None # delay added to render each frame (None for no delay)
        self.log_every = 50 # logs the current stats every x episodes
        self.replay_start_size = 1000 # minimum steps stored in the agent required to start training
        self.train_every = 1 # train every x episodes
        self.n_neurons = [32, 32, 32] # number of neurons for each activation layer
        self.activations = ['relu', 'relu', 'relu', 'linear']       # delay per frame when rendering
        self.save_best_model = True         # save best model to file
        self.total_steps = 0 
        self.model_path = model_path
        self.BOARD_HEIGHT = 20
        self.BOARD_WIDTH = 10
        self.log_every = 50

        self.start_episode = 1
        self.best_score = -float('inf')
        # if os.path.exists("checkpoint.json"):
        #     with open("checkpoint.json", "r") as f:
        #         data = json.load(f)
        #         self.start_episode = data.get("episode", 1)
        #         self.total_steps = data.get("total_steps", 0)
        #         self.best_score = data.get("best_score", -float('inf'))
        #         print(f"Resuming from episode {self.start_episode}")

        # Neural net architecture
       
        log_dir = (
            f"logs/tetris-nn={self.n_neurons}-mem={self.mem_size}-bs={self.batch_size}-"
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        # Initialize agent
        self.agent = EnhancedDQNAgent(
            n_neurons=self.n_neurons,
            activations=self.activations,
            epsilon_stop_episode=self.epsilon_stop_episode,
            mem_size=self.mem_size,
            discount=self.discount,
            replay_start_size=self.replay_start_size,
            tensorboard_log_dir=log_dir
        )

        # Load model weights if required
        if load_model and os.path.exists(self.model_path):
            self.agent.model.load_weights(self.model_path)
            print(f"✅ Loaded model weights from {self.model_path}")

    def ensure_connection_ready(self, max_retries=10):
        state = self.client.wait_for_game_ready(timeout=5.0)
        if state is not None:
            return state
        self.client.disconnect()
        import time
        time.sleep(10.0)
        self.client.connect()

    def calculate_reward(self, prev_state, current_state, action, step):
        board = current_state.get('board')
        if not isinstance(board, list) or len(board) != self.BOARD_HEIGHT * self.BOARD_WIDTH:
            self.agent.writer.add_scalar('reward/invalid_board', 1, step)
            return -0.5
        if prev_state: 
            lines_prev = prev_state.get('linesCleared', 0)
        else: 
            lines_prev= 0
        lines = max(0, current_state.get('linesCleared', 0) - lines_prev)
        score_comp = 1 + (lines ** 2) * self.BOARD_WIDTH
        if current_state.get('gameOver', False):
            score_comp -= 2
        reward = score_comp
        return reward

    def train(self):
        scores = []
        best_score = self.best_score

        if not self.client.connect():
            print("Failed to connect to unity")
            return

        try:
            pbar = tqdm(total=self.episodes, desc="Training Episodes", initial=self.start_episode-1)
            episode = self.start_episode
            while episode <= self.episodes:
                # Start episode
                current_state_meta = self.client.wait_for_game_ready(timeout=15.0)

                done = False
                episode_reward = 0
                episode_score = 0
                episode_lines = 0
                steps = 0
                current_state = [0, 0, 0, 0]

                while not done and (self.max_steps is None or steps < self.max_steps):
                    # Action selection
                    if current_state_meta is None:
                         break
                    next_states = self.agent.get_possible_states(current_state_meta)
                    state_to_action = {tuple(f): a for a, f in next_states}
                    if not state_to_action:
                        next_state = self.client.send_action_and_wait(0, timeout=10.0)
                        episode_reward -= 2
                        break

                    best_state = self.agent.best_state(list(state_to_action.keys()), episode)
                    action = state_to_action[tuple(best_state)]
                  

                    # Execute action
                    next_state = self.client.send_action_and_wait(action, timeout=30.0)
                    if not next_state:
                        print(f"Episode {episode}: Timeout waiting for next state, retrying same episode")
                        break
                    done = self.client.is_game_over(next_state)
                    reward = self.calculate_reward(current_state_meta, next_state, action, self.total_steps)
                    self.agent.add_to_memory(current_state, best_state, reward, done)

                    episode_score = next_state.get('score', 0)
                    episode_lines = next_state.get('linesCleared', 0)
                    self.total_steps += 1
                    steps += 1
                    episode_reward += reward

                    if done:
                        print(f"Episode {episode}: Score={episode_score}, Reward={episode_reward:.1f}, "
                            f"Lines={episode_lines}, Steps={steps}")
                        break

                    current_state = best_state
                    current_state_meta = next_state

                # If episode ended immediately, retry same episode
                if steps <= 1:
                    print(f"Episode {episode}: ended in {steps} step(s), retrying same episode")
                    continue

                # Logging & training for successful episodes
                scores.append(episode_reward)
                self.agent.writer.add_scalar("score/score", episode_reward, episode)
                if episode % self.train_every == 0:
                    self.agent.train(batch_size=self.batch_size, epochs=1)

                if self.log_every and episode % self.log_every == 0:
                    recent = scores[-self.log_every:]
                    self.agent.writer.add_scalar("score/avg_score", mean(recent), episode)
                    self.agent.writer.add_scalar("score/min_score", min(recent), episode)
                    self.agent.writer.add_scalar("score/max_score", max(recent), episode)

                # Save
                if self.save_best_model and episode_reward > best_score:
                    print(f"Saving new best model: score={episode_reward} at episode={episode}")
                    best_score = episode_reward
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    best_path = f"best_{ts}.weights.h5"
                    self.save_model(best_path)
                    self.save_checkpoint(episode, best_model_path=best_path)

                if episode % 10 == 0:
                    self.save_checkpoint(episode)
                    self.save_model(f"checkpoint_episode_{episode}.weights.h5")

                episode += 1
                pbar.update(1)

            pbar.close()

            print("Training completed.")
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            logging.info("Training interrupted by user")
        except Exception as e:
            print(f"Training error: {e}")
            logging.error(f"Training error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.save_model()
            self.agent.close()
            self.client.disconnect()
            logging.info("Training session ended")
            print("Training completed!")

    def save_model(self, filename=None):
        if filename is None:
            filename = self.model_path
        self.agent.save_model(filename)

    def save_checkpoint(self, episode, best_model_path=None):
        checkpoint_data = {
            "episode": episode,
            "total_steps": self.total_steps,
            "best_score": self.best_score
        }
        if best_model_path:
            checkpoint_data["best_model_path"] = best_model_path
        with open("checkpoint.json", "w") as f:
            json.dump(checkpoint_data, f)

if __name__ == "__main__":
    trainer = TetrisTrainer(agent_type='dqn', load_model=False, model_path='tetris_model.weights.h5')
    trainer.train()
