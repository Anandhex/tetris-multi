import torch
from keras.models import Sequential
from keras.layers import Dense
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
import numpy as np
import os
from datetime import datetime

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
        
        print(f"Enhanced DQN Agent initialized on {device}")
        print(f"TensorBoard logs: {tensorboard_log_dir}")
    
    

    def _build_model(self):
        '''Builds a Keras deep neural network model'''
        model = Sequential()
        model.add(Dense(self.n_neurons[0], input_dim=self.state_size, activation=self.activations[0]))

        for i in range(1, len(self.n_neurons)):
            model.add(Dense(self.n_neurons[i], activation=self.activations[i]))

        model.add(Dense(1, activation=self.activations[-1]))

        model.compile(loss=self.loss, optimizer=self.optimizer)
        
        return model



   
    
   
    
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


