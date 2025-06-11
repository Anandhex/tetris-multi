import sys
import logging
import time
from datetime import datetime
from collections import Counter, deque

import numpy as np
import pandas as pd

from tetris_client import UnityTetrisClient
from simplified_dqn_network import DQNAgent


class TetrisTrainer:
    def __init__(
        self,
        agent_type='dqn',
        load_model=False,
        model_path='tetris_model.pth',
        tensorboard_log_dir=None,
        score_window_size=100
    ):
        self.client = UnityTetrisClient()
        self.agent_type = agent_type
        self.model_path = model_path
        self.total_steps = 0

        if tensorboard_log_dir is None:
            tensorboard_log_dir = (
                f"runs/tetris_{agent_type}_"
                f"{datetime.now():%Y%m%d_%H%M%S}"
            )

        # Initialize agent
        if agent_type == 'dqn':
            self.agent = DQNAgent(
                state_size=208,
            )
            if load_model:
                self.agent.load(model_path)

        self.action_counter = Counter()

        # Episode metrics
        self.episode_scores = []
        self.episode_lines = []
        self.episode_lengths = []
        self.episode_rewards = []
        self.best_score = 0

        # Curriculum parameters
        self.current_curriculum_stage = 0
        self.consecutive_good_episodes = 0
        self.curriculum_stages = [
            { 'name':'Very Easy','height':20,'preset':1,'pieces':1,
              'advancement_threshold': 900, 'consecutive_required': 8 },
            { 'name':'Easy','height':20,'preset':2,'pieces':2,
              'advancement_threshold':1100, 'consecutive_required':10 },
            { 'name':'Medium','height':20,'preset':3,'pieces':3,
              'advancement_threshold':1200,'consecutive_required':12 },
            { 'name':'Hard','height':20,'preset':0,'pieces':5,
              'advancement_threshold':2000,'consecutive_required':15 },
            { 'name':'Full Game','height':20,'preset':0,'pieces':7,
              'advancement_threshold':float('inf'),
              'consecutive_required':float('inf') },
        ]

        # Board dimensions
        self.BOARD_HEIGHT = 20
        self.BOARD_WIDTH  = 10

        # Logging setup
        self.setup_logging()

    def setup_logging(self):
        log_file = f"tetris_training_{datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger().addHandler(console)
        logging.info("=== Training session started ===")

    def _reshape_board(self, flat):
        return np.array(flat, dtype=np.uint8).reshape(
            (self.BOARD_HEIGHT, self.BOARD_WIDTH)
        )

    def ensure_connection_ready(self):
        for _ in range(5):
            state = self.client.wait_for_game_ready(timeout=5.0)
            if state:
                return state
            self.client.send_reset()
            time.sleep(2.0)
        return None

    def check_curriculum_advancement(self, score):
        stage = self.curriculum_stages[self.current_curriculum_stage]
        if score >= stage['advancement_threshold']:
            self.consecutive_good_episodes += 1
        else:
            self.consecutive_good_episodes = 0
        return self.consecutive_good_episodes >= stage['consecutive_required']

    def apply_curriculum(self, episode, last_score, force=False):
        if not force and episode > 0:
            if not self.check_curriculum_advancement(last_score):
                return False
            self.current_curriculum_stage += 1
            self.consecutive_good_episodes = 0
            self.agent.epsilon = 1.0
            self.agent.writer.add_scalar(
                'Agent/Epsilon_Reset', self.agent.epsilon, self.total_steps
            )

        stage = self.curriculum_stages[self.current_curriculum_stage]
        logging.info(
            f"🚀 Curriculum → {stage['name']} "
            f"(height={stage['height']}, pieces={stage['pieces']})"
        )
        success = self.client.send_curriculum_change(
            board_height=stage['height'],
            board_preset=stage['preset'],
            tetromino_types=stage['pieces'],
            stage_name=stage['name']
        )
        if success:
            time.sleep(3.0)
            self.client.send_reset()
            time.sleep(1.0)
        return True

    def calculate_reward(self, prev_state, current_state, action, step):
        """
        Matches tetris.py:
          +1 per piece placed
          + (lines_cleared^2 * BOARD_WIDTH)
          -2 if game over
        """
        prev_lines = prev_state.get('linesCleared', 0) if prev_state else 0
        curr_lines = current_state.get('linesCleared', 0)
        lines = max(0, curr_lines - prev_lines)

        reward = 1 + (lines ** 2) * self.BOARD_WIDTH
        if current_state.get('gameOver', False):
            reward -= 2

        self.agent.writer.add_scalar('reward/total', reward, step)
        return float(reward)

    def train(self, episodes=sys.maxsize, save_interval=100):
        if not self.client.connect():
            logging.error("Cannot connect to Unity!")
            return

        logging.info("Starting training loop")
        prev_score = 0

        for ep in range(episodes):
            # Curriculum handling
            if ep == 0:
                self.apply_curriculum(ep, prev_score, force=True)
            else:
                self.apply_curriculum(ep, prev_score)

            state = self.ensure_connection_ready()
            if state is None:
                logging.warning(f"Ep {ep}: failed to get initial state, skipping")
                continue

            ep_reward = 0.0
            ep_score = 0
            ep_lines = 0
            step = 0
            prev_state = None

            while True:
                print(state)
                action = self.agent.act(state)
                nxt = self.client.send_action_and_wait(action, timeout=10.0)
                if nxt is None:
                    logging.warning(f"Ep {ep} step {step}: no response, breaking")
                    break

                done = nxt.get('gameOver', False)
                self.total_steps += 1

                r = self.calculate_reward(prev_state, nxt, action, self.total_steps)
                ep_reward += r
                ep_score  = nxt.get('score', 0)
                ep_lines  = nxt.get('linesCleared', 0)

                self.agent.remember(state, action, r, (None if done else nxt), done)
                if len(self.agent.memory) > self.agent.batch_size:
                    self.agent.replay()

                step += 1
                if done:
                    break

                prev_state = state
                state = nxt

            # End of episode
            self.episode_scores.append(ep_score)
            self.episode_lines .append(ep_lines)
            self.episode_lengths.append(step)
            self.episode_rewards.append(ep_reward)

            self.agent.log_episode_metrics(
                ep, ep_reward, step, ep_score, ep_lines, {}
            )

            if ep_score > self.best_score:
                self.best_score = ep_score
                self.agent.save(f"best_model_{ep_score}.pth")
                logging.info(f"🏆 New best score {ep_score} at episode {ep}")

            if ep % save_interval == 0 and ep > 0:
                self.agent.save(self.model_path)
                self._save_metrics_csv()

            prev_score = ep_score

        # Final save
        self.agent.save(self.model_path)
        self._save_metrics_csv()
        self.agent.close()
        self.client.disconnect()
        logging.info("=== Training complete ===")

    def _save_metrics_csv(self):
        df = pd.DataFrame({
            'episode':       range(len(self.episode_scores)),
            'score':         self.episode_scores,
            'lines_cleared': self.episode_lines,
            'length':        self.episode_lengths,
            'reward':        self.episode_rewards,
        })
        df['score_ma20']  = df['score'].rolling(20, min_periods=1).mean()
        df['reward_ma20'] = df['reward'].rolling(20, min_periods=1).mean()
        df.to_csv('training_metrics.csv', index=False)
        logging.info("Metrics saved to training_metrics.csv")
