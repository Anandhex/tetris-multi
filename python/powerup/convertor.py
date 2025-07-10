from powerup_env import PowerupEnv
import torch
from powerup_net import PowerupNet


env = PowerupEnv()
input_size = env.get_state().shape[0]
output_size = 23  # 0–22 actions
# 1. Define your model (use same dimensions!)
model = PowerupNet(input_size, output_size)
model.load_state_dict(torch.load("models/20250710_190517/dqn_powerup_20250710_190517.pth"))
model.eval()

# 2. Dummy input for tracing — must match input shape during inference
dummy_input = torch.randn(1, input_size)  # e.g., (batch_size=1, features=30)

# 3. Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "models/powerup_supervised_20250710_1700.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

print("Model exported to ONNX successfully.")