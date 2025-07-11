import torch
import torch.optim as optim
import numpy as np
import random
import os
from torch.utils.tensorboard import SummaryWriter


def generate_supervised_dataset(env, samples=1000):
    X, y = [], []
    for _ in range(samples):
        env.reset()
        state = env.get_state()

        # Heuristic label
        if env.powerups[0] == 1 and np.sum(env.player_board[0]) >= 4:
            action = 1  # clear bottom line
        elif env.powerups[1] == 1 and np.sum(env.player_board == 0) > 50:
            action = 2  # gravity
        elif env.powerups[2] == 1:
            action = 3 + random.randint(0, 9)  # bomb
        elif env.powerups[3] == 1:
            action = 13 + random.randint(0, 9)  # wild
        else:
            action = 0  # do nothing

        X.append(state)
        y.append(action)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def train_supervised(model, env, epochs=5, batch_size=64,timestamp=""):
    X, y = generate_supervised_dataset(env, samples=10000)
    log_dir = f"runs/{timestamp}/supervised_experiment_{timestamp}"
    writer = SummaryWriter(log_dir)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    global_step = 0

    for epoch in range(epochs):
        permutation = np.random.permutation(len(X))
        total_loss = 0 # Track loss per epoch
        num_batches = 0 # Track number of batches per epoch

        for i in range(0, len(X), batch_size):
            idx = permutation[i:i+batch_size]
            x_batch = torch.tensor(X[idx])
            y_batch = torch.tensor(y[idx])

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            writer.add_scalar('Supervised/Loss', loss.item(), global_step) # Log loss per batch
            total_loss += loss.item()
            num_batches += 1
            global_step += 1

        print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        writer.add_scalar('Supervised/Average Epoch Loss', avg_loss, epoch) # Log average epoch loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    writer.close()     

    os.makedirs(f"models/{timestamp}", exist_ok=True)
    save_path = f"models/{timestamp}/powerup_supervised_{timestamp}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Supervised model saved to: {save_path}")    
