from simplified_dqn_network import EnhancedDQNAgent
from datetime import datetime
from tetris_client import UnityTetrisClient
from statistics import mean
from tqdm import tqdm
import logging
import sys

class TetrisTrainer:
    def __init__(self, agent_type='dqn', load_model=False, model_path='tetris_model.keras', 
                 tensorboard_log_dir=None, score_window_size=100):

     # Environment and hyperparameters
        self.client = UnityTetrisClient()
        self.episodes = 3000                # total number of episodes
        self.max_steps = None               # max steps per game (None for unlimited)
        self.epsilon_stop_episode = 2000    # when epsilon exploration stops
        self.memory_size = 1000             # replay buffer size
        self.discount_factor = 0.95         # gamma in Q-learning
        self.batch_size = 128               # minibatch size
        self.train_every = 1                # train frequency (episodes)
        replay_start_size = 1000       # start training after this many steps
        self.render_every = 50              # render every N episodes
        self.render_delay = None            # delay per frame when rendering
        self.save_best_model = True         # save best model to file
        self.total_steps = 0 
        self.model_path = model_path
        self.BOARD_HEIGHT = 20
        self.BOARD_WIDTH = 10

        # Neural net architecture
        hidden_layers = [32, 32, 32]
        activations = ['relu', 'relu', 'relu', 'linear']
        log_dir = (
            f"logs/tetris-nn={hidden_layers}-mem={self.memory_size}-bs={self.batch_size}-"
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        # Initialize agent
        self.agent = EnhancedDQNAgent(
            n_neurons=hidden_layers,
            activations=activations,
            epsilon_stop_episode=self.epsilon_stop_episode,
            mem_size=self.memory_size,
            discount=self.discount_factor,
            replay_start_size=replay_start_size,
            tensorboard_log_dir=log_dir
        )

        # Setup TensorBoard logging

    def ensure_connection_ready(self):
        """Ensure connection is ready before starting episode"""
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                # Try to get initial state
                state = self.client.wait_for_game_ready(timeout=5.0)
                if state is not None:
                    return state
                
                print(f"⚠️ No initial state, attempt {attempt + 1}/{max_attempts}")
                
                # Try reset
                self.client.send_reset()
                import time
                time.sleep(2.0)
                
            except Exception as e:
                print(f"❌ Connection issue: {e}")
                
                # Reconnect if not last attempt
                if attempt < max_attempts - 1:
                    self.client.disconnect()
                    import time
                    time.sleep(2.0)
                    if not self.client.connect():
                        continue
        
        return None 
    def calculate_reward(self, prev_state, current_state, action, step):
        # 1) Validate board
        board = current_state.get('board')
        if not isinstance(board, list) or len(board) != self.BOARD_HEIGHT * self.BOARD_WIDTH:
            self.agent.writer.add_scalar('reward/invalid_board', 1, step)
            return -0.5
        if prev_state: 
            lines_prev = prev_state.get('linesCleared', 0)
        else: 
            lines_prev= 0
        lines = max(0, current_state.get('linesCleared', 0) - lines_prev)

       

        # Score‐based component with a single death penalty
        score_comp =1+(lines ** 2)*self.BOARD_WIDTH
        if current_state.get('gameOver', False):
            score_comp -= 2


      
        reward = score_comp

        # Log each step so you can see the numbers
        w = self.agent.writer
       

        # And the normal logs
        w.add_scalar('reward/total',       reward,        step)

        return reward

    def train(self):
        scores = []
        best_score = -float('inf')

        if not self.client.connect():
            print("Failed to connect to unity")
            return
        

        try:
            # Main training loop
            for episode in tqdm(range(1, self.episodes + 1), desc="Training Episodes"):
                state = self.ensure_connection_ready()
                if state is None:
                    print(f"Episode {episode}: Failed to get initial game state, skipping episode")
                    continue

                done = False
                episode_reward = 0
                episode_score = 0
                episode_lines = 0

                steps = 0
                prev_state = None
                
                # Determine whether to render this episode

                # Episode rollout
                while not done and (self.max_steps is None or steps < self.max_steps):
                    # Get mapping of next states to actions
                    next_states = self.agent.get_possible_states(state)  # returns dict: action->state
                    # Flip mapping to choose best state
                    state_to_action = {
                        tuple(features): action
                    for action, features in next_states
                    }

                    # Agent selects best state
                    best_state = self.agent.best_state(list(state_to_action.keys()))
                    action = state_to_action[tuple(best_state)]

                    # Play the chosen action
                    next_state= self.client.send_action_and_wait(action,timeout=10.0)
                    if next_state is None:
                        print(f"Episode {episode}: Timeout waiting for next state")
                        break
                    done = self.client.is_game_over(next_state)
                    self.total_steps+=1

                    reward = self.calculate_reward(prev_state=prev_state,current_state=next_state,action=action,step=self.total_steps)
                    # Store transition
                    
                    state = best_state
                    steps += 1
                    episode_reward += reward
                    episode_score = next_state.get('score', 0)
                    episode_lines = next_state.get('linesCleared', 0)

                    self.agent.add_to_memory(
                        state, best_state, reward, done
                    )
                    if done:
                            # Enhanced game over logging
                            board_metrics = self.client.get_board_metrics(next_state)
                            pieces_placed = next_state.get('piecesPlaced', 0)
                            efficiency = episode_lines / pieces_placed if pieces_placed > 0 else 0
                            
                            if episode % 1 == 0:
                                print(f"Episode {episode}: Score={episode_score}, "
                                    f"Reward={episode_reward:.1f}, Lines={episode_lines}, "
                                    f"Steps={steps}, Efficiency={efficiency:.3f}, "
                                    f"Height={board_metrics.get('stack_height', 0)}")
                    prev_state = next_state
                    state = next_state            

                # End of episode
                
                scores.append(episode_score)

                # Training step
                if episode % self.train_every == 0:
                    self.agent.train(batch_size=self.batch_size, epochs=1)

                

                # Save best model
                if self.save_best_model and episode_score > best_score:
                    print(f"Saving new best model: score={episode_score} at episode={episode}")
                    best_score = episode_score
                    self.save_model("best.keras")

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
            # Save final model and metrics
            self.save_model()
            self.agent.close()
            self.client.disconnect()
            logging.info("Training session ended")
            print("Training completed!")

    def save_model(self, filename=None):
        """Save the trained model"""
        if filename is None:
            filename = self.model_path
        self.agent.save_model(filename)   


if __name__ == "__main__":
    trainer = TetrisTrainer(agent_type='dqn',load_model=False,model_path='tetris_model.keras',tensorboard_log_dir=None)
    trainer.train()
