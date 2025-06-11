import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, input_dim, n_neurons, activations):
        super().__init__()
        layers = []
        in_dim = input_dim
        # Hidden layers
        for idx, out_dim in enumerate(n_neurons):
            layers.append(nn.Linear(in_dim, out_dim))
            act = activations[idx].lower()
            if act == 'relu':
                layers.append(nn.ReLU())
            elif act == 'tanh':
                layers.append(nn.Tanh())
            elif act == 'sigmoid':
                layers.append(nn.Sigmoid())
            # (ignore 'linear')
            in_dim = out_dim
        # Output layer
        layers.append(nn.Linear(in_dim, 1))
        last_act = activations[-1].lower()
        if last_act == 'relu':
            layers.append(nn.ReLU())
        elif last_act == 'tanh':
            layers.append(nn.Tanh())
        elif last_act == 'sigmoid':
            layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DQNAgent:
    def __init__(self,
                 state_size,
                 mem_size=10000,
                 discount=0.95,
                 epsilon=1.0,
                 epsilon_min=0.0,
                 epsilon_stop_episode=0,
                 n_neurons=[32, 32],
                 activations=['relu', 'relu', 'linear'],
                 replay_start_size=None,
                 model_file=None,
                 lr=1e-3,
                 epochs=3):
        # --- Hyperparameters and replay buffer ---
        if len(activations) != len(n_neurons) + 1:
            raise ValueError("activations length must be n_neurons + 1")
        self.state_size = state_size
        self.memory     = deque(maxlen=mem_size)
        self.discount   = discount

        # ε-greedy schedule
        self.epsilon    = epsilon if epsilon_stop_episode > 0 else 0.0
        self.epsilon_min= epsilon_min
        self.epsilon_decay = ((epsilon - epsilon_min) /
                              epsilon_stop_episode) if epsilon_stop_episode > 0 else 0.0

        # warm‐up before training
        self.replay_start_size = (replay_start_size
                                  if replay_start_size is not None
                                  else mem_size / 2)
        self.n_neurons = n_neurons
        self.activations = activations
        self.epochs = epochs

        # --- Build or load network ---
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_network = QNetwork(state_size, n_neurons, activations).to(self.device)
        if model_file is not None:
            self.q_network.load_state_dict(torch.load(model_file, map_location=self.device))

        # --- Optimizer & loss ---
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def add_to_memory(self, current_state, next_state, reward, done):
        """Store transition in replay buffer."""
        self.memory.append((current_state, next_state, reward, done))

    def random_value(self):
        """Random Q-value for exploration."""
        return random.random()

    def predict_value(self, state):
        """Predict Q-value for a single state."""
        self.q_network.eval()
        with torch.no_grad():
            s = np.reshape(state, [1, self.state_size]).astype(np.float32)
            t = torch.from_numpy(s).to(self.device)
            return self.q_network(t).cpu().numpy()[0, 0]

    def act(self, state):
        """Return either a random value (explore) or the network's Q-value (exploit)."""
        if random.random() <= self.epsilon:
            return self.random_value()
        return self.predict_value(state)

    def best_state(self, states):
        """Given a list of candidate states, pick one at random (explore) or the best (exploit)."""
        if random.random() <= self.epsilon:
            return random.choice(states)
        best = max(states, key=self.predict_value)
        return best

    def train(self, batch_size=32):
        """Sample a batch and update the network."""
        n = len(self.memory)
        if n < self.replay_start_size or n < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states      = np.array([x[0] for x in batch], dtype=np.float32)
        next_states = np.array([x[1] for x in batch], dtype=np.float32)
        rewards     = np.array([x[2] for x in batch], dtype=np.float32)
        dones       = np.array([x[3] for x in batch], dtype=bool)

        # Compute next Q-values in one forward pass
        self.q_network.eval()
        with torch.no_grad():
            ns_t = torch.from_numpy(next_states).to(self.device)
            next_qs = self.q_network(ns_t).cpu().numpy().squeeze()

        # Build training data
        x_batch, y_batch = [], []
        for i, (s, _, r, done) in enumerate(batch):
            q_target = r if done else r + self.discount * next_qs[i]
            x_batch.append(s)
            y_batch.append(q_target)

        # Train for a few epochs
        self.q_network.train()
        x_t = torch.from_numpy(np.array(x_batch, dtype=np.float32)).to(self.device)
        y_t = torch.from_numpy(np.array(y_batch, dtype=np.float32)).unsqueeze(1).to(self.device)
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            preds = self.q_network(x_t)
            loss = self.criterion(preds, y_t)
            loss.backward()
            self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def save_model(self, filename):
        """Persist network weights to disk."""
        torch.save(self.q_network.state_dict(), filename)
