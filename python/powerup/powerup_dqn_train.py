from powerup_net import PowerupNet
import torch 
import random
import torch.optim as optim
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

def select_action(model, state, epsilon, action_size):
    if random.random() < epsilon:
        return random.randint(0, action_size - 1)
    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return model(state).argmax().item()

def train_dqn(model, env, episodes=300, batch_size=64, gamma=0.99,timestamp=""):
    target_model = PowerupNet(model.net[0].in_features, model.net[-1].out_features)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    log_dir = f"runs/{timestamp}/dqn_experiment_{timestamp}"
    replay = ReplayBuffer()
    epsilon = 1.0
    epsilon_decay = 0.999
    min_epsilon = 0.05
    writer = SummaryWriter(log_dir)
    global_step = 0

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = select_action(model, state, epsilon, 23)
            next_state, reward, done, _ = env.step(action)
            replay.push((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(replay.buffer) >= batch_size:
                batch = replay.sample(batch_size)
                s, a, r, s_, d = zip(*batch)

                s = torch.tensor(s, dtype=torch.float32)
                a = torch.tensor(a, dtype=torch.long).unsqueeze(1)
                r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
                s_ = torch.tensor(s_, dtype=torch.float32)
                d = torch.tensor(d, dtype=torch.float32).unsqueeze(1)

                q_vals = model(s).gather(1, a)
                max_q_next = target_model(s_).max(1)[0].unsqueeze(1)
                target = r + gamma * max_q_next * (1 - d)

                loss = torch.nn.functional.mse_loss(q_vals, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar('DQN/Loss', loss.item(), global_step)
                global_step += 1

        if ep % 10 == 0:
            target_model.load_state_dict(model.state_dict())
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        print(f"Episode {ep+1}: total_reward = {total_reward:.2f}, epsilon = {epsilon:.2f}")
        writer.add_scalar('DQN/Total Reward', total_reward, ep) # Log total reward per episode
        writer.add_scalar('DQN/Epsilon', epsilon, ep) # Log epsilon per episode

    save_path = f"models/{timestamp}/dqn_powerup_{timestamp}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"DQN model saved to: {save_path}")    
