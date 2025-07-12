from powerup_env import PowerupEnv
from power_up_supervised import train_supervised
from powerup_dqn_train import train_dqn
from powerup_net import PowerupNet
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


env = PowerupEnv()
input_size = env.get_state().shape[0]
output_size = 23  # 0–22 actions

model = PowerupNet(board_channels=2, powerup_size=4, output_size=output_size)

# Train supervised first (optional pretraining)
train_supervised(model, env, epochs=5,timestamp=timestamp)

# Then DQN fine-tuning
train_dqn(model, env, episodes=30000,timestamp=timestamp)
