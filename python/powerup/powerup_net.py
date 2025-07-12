from powerup_env import PowerupEnv
import torch.nn as nn
import torch;
import numpy as np

class PowerupNet(nn.Module):
    def __init__(self, board_channels=2, powerup_size=4, output_size=23):
        super(PowerupNet, self).__init__()
        
        # CNN for player + opponent boards
        self.cnn = nn.Sequential(
            nn.Conv2d(board_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute flattened CNN output shape for 2x20x10 input
        self._dummy_input = torch.zeros(1, 2, 20, 10)
        with torch.no_grad():
            self.cnn_output_size = self.cnn(self._dummy_input).shape[1]

        # MLP for powerups
        self.mlp_powerups = nn.Sequential(
            nn.Linear(powerup_size, 32),
            nn.ReLU()
        )

        # Final combined MLP
        self.combined = nn.Sequential(
            nn.Linear(self.cnn_output_size + 32, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, board_tensor, powerup_tensor):
        board_features = self.cnn(board_tensor)
        powerup_features = self.mlp_powerups(powerup_tensor)
        combined = torch.cat((board_features, powerup_features), dim=1)
        return self.combined(combined)


# # === Load the model ===
# env = PowerupEnv()
# input_size = env.get_state().shape[0]
# output_size = 23  # 0–22 actions

# model = PowerupNet(input_size,output_size)
# model.load_state_dict(torch.load("./models/20250711_164428/dqn_powerup_20250711_164428.pth"))
# model.eval()

# # === Initialize environment ===
# state = env.reset()
# done = False
# total_reward = 0

# # === Inference loop ===
# while not done:
#     # Prepare input
#     input_vector = np.concatenate([
#         env.player_board.flatten(),
#         env.opponent_board.flatten(),
#         env.powerups
#     ]).astype(np.float32)

#     input_tensor = torch.tensor(input_vector).unsqueeze(0)  # Shape: [1, input_dim]

#     # Predict best action
#     with torch.no_grad():
#         q_values = model(input_tensor)
#         action = torch.argmax(q_values).item()

#     # Apply action in the environment
#     state, reward, done, _ = env.step(action)
#     print(action)
#     total_reward += reward

# print(f"\n✅ Inference episode completed. Total Reward: {total_reward:.2f}")