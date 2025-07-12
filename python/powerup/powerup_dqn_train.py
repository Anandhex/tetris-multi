from powerup_net import PowerupNet
import torch 
import random
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity

    def push(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

def select_action(model, state, epsilon, action_size,powerups):
    if random.random() < epsilon:
            _, valid_mask = mask_invalid_actions(np.ones(action_size), powerups)
            valid_indices = np.where(valid_mask)[0]
            return int(np.random.choice(valid_indices))

    with torch.no_grad():
            player_board = state[:200].reshape(20, 10)
            opponent_board = state[200:400].reshape(20, 10)
            board_tensor = torch.tensor(np.stack([player_board, opponent_board]), dtype=torch.float32).unsqueeze(0)
            powerup_tensor = torch.tensor(state[400:], dtype=torch.float32).unsqueeze(0)

            q_values = model(board_tensor, powerup_tensor).squeeze(0).numpy()
            q_values, _ = mask_invalid_actions(q_values, powerups)
            return int(np.argmax(q_values))
    
def mask_invalid_actions(q_values, powerups):
    valid = np.ones_like(q_values, dtype=bool)

    if powerups[0] == 0:
        valid[1] = False  # Clear bottom line
    if powerups[1] == 0:
        valid[2] = False  # Gravity
    if powerups[2] == 0:
        valid[3:13] = False  # Bomb (cols 0–9)
    if powerups[3] == 0:
        valid[13:23] = False  # Wild card (cols 0–9)

    q_values[~valid] = -np.inf
    return q_values, valid    

def train_dqn(model, env, episodes=300, batch_size=500, gamma=0.99,timestamp=""):
    target_model = PowerupNet(board_channels=2, powerup_size=4, output_size=23)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    log_dir = f"runs/{timestamp}/dqn_experiment_{timestamp}"
    replay = ReplayBuffer(capacity=1000)
    epsilon = 1.0
    epsilon_decay = 0.95
    min_epsilon = 0.0
    writer = SummaryWriter(log_dir)
    global_step = 0
    epsilon_decay = (epsilon - min_epsilon) / (episodes)
    warmup_steps = int(0.7 * replay.capacity) 

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        actions_this_episode = []



        while not done:
            action = select_action(model, state, epsilon, 23,env.powerups.copy())
            next_state, reward, done, _ = env.step(action)
            replay.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            actions_this_episode.append(action)

            if len(replay.buffer) >= warmup_steps and len(replay.buffer) >= batch_size:
                batch = replay.sample(batch_size)
                s, a, r, s_, d = zip(*batch)

                s = np.array(s)
                s_ = np.array(s_)

                # Split states
                s_board = torch.tensor(s[:, :400].reshape(-1, 2, 20, 10), dtype=torch.float32)
                s_power = torch.tensor(s[:, 400:], dtype=torch.float32)

                s_board_ = torch.tensor(s_[:, :400].reshape(-1, 2, 20, 10), dtype=torch.float32)
                s_power_ = torch.tensor(s_[:, 400:], dtype=torch.float32)

                a = torch.tensor(a, dtype=torch.long).unsqueeze(1)
                r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
                d = torch.tensor(d, dtype=torch.float32).unsqueeze(1)

                # Forward pass
                q_vals = model(s_board, s_power).gather(1, a)
                max_q_next = target_model(s_board_, s_power_).max(1)[0].unsqueeze(1)
                target = r + gamma * max_q_next * (1 - d)

                loss = torch.nn.functional.mse_loss(q_vals, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar('DQN/Loss', loss.item(), global_step)
                global_step += 1

        if ep % 10 == 0:
            target_model.load_state_dict(model.state_dict())
        epsilon -=epsilon_decay
        print(f"Episode {ep+1}: total_reward = {total_reward:.2f}, epsilon = {epsilon:.2f}")
        writer.add_scalar('DQN/Total Reward', total_reward, ep) # Log total reward per episode
        writer.add_scalar('DQN/Epsilon', epsilon, ep) # Log epsilon per episode
        writer.add_histogram('DQN/Action Distribution', np.array(actions_this_episode), ep)

    save_path = f"models/{timestamp}/dqn_powerup_{timestamp}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"DQN model saved to: {save_path}")    
