import torch
import torch.optim as optim
import numpy as np
import random
import os
from torch.utils.tensorboard import SummaryWriter

def generate_supervised_dataset(env, samples=1000):
    board_data, powerup_data, labels = [], [], []

    for _ in range(samples):
        env.reset()

        board = np.stack([env.player_board, env.opponent_board], axis=0)  # Shape: (2, 20, 10)
        powerups = env.powerups  # Shape: (4,)

        # === Heuristic Labeling ===
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

        board_data.append(board)
        powerup_data.append(powerups)
        labels.append(action)

    return (
        np.array(board_data, dtype=np.float32),     # (N, 2, 20, 10)
        np.array(powerup_data, dtype=np.float32),   # (N, 4)
        np.array(labels, dtype=np.int64)            # (N,)
    )


def train_supervised(model, env, epochs=5, batch_size=64, timestamp=""):
    X_board, X_powerup, y = generate_supervised_dataset(env, samples=10000)

    log_dir = f"runs/{timestamp}/supervised_experiment_{timestamp}"
    writer = SummaryWriter(log_dir)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    global_step = 0

    for epoch in range(epochs):
        permutation = np.random.permutation(len(y))
        total_loss = 0
        num_batches = 0

        for i in range(0, len(y), batch_size):
            idx = permutation[i:i+batch_size]

            board_batch = torch.tensor(X_board[idx])          # [B, 2, 20, 10]
            powerup_batch = torch.tensor(X_powerup[idx])      # [B, 4]
            label_batch = torch.tensor(y[idx])                # [B]

            optimizer.zero_grad()
            preds = model(board_batch, powerup_batch)
            loss = criterion(preds, label_batch)
            loss.backward()
            optimizer.step()

            writer.add_scalar('Supervised/Loss', loss.item(), global_step)
            total_loss += loss.item()
            num_batches += 1
            global_step += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        writer.add_scalar('Supervised/Average Epoch Loss', avg_loss, epoch)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    writer.close()
    os.makedirs(f"models/{timestamp}", exist_ok=True)
    save_path = f"models/{timestamp}/powerup_supervised_{timestamp}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Supervised model saved to: {save_path}")
