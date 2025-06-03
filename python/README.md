Tetris AI Training with Deep Reinforcement Learning
A complete deep reinforcement learning system for training AI agents to play Tetris using Unity and Python. Features socket-based communication, curriculum learning, and comprehensive TensorBoard monitoring.
Show Image Show Image Show Image Show Image Show Image
🎮 Overview
This project implements a sophisticated AI training system for Tetris using Deep Q-Networks (DQN). The AI learns to play Tetris through trial and error, gradually improving its strategy over thousands of games. The system uses Unity for game simulation and Python for AI training, connected via socket communication.
Key Features

🧠 Deep Q-Network (DQN) with convolutional neural networks for board analysis
🔌 Socket-based communication between Unity and Python for real-time training
📈 Curriculum learning with progressive difficulty scaling
📊 TensorBoard integration for comprehensive training monitoring
🎯 40 discrete actions (10 columns × 4 rotations) for precise piece placement
📱 Real-time visualization of training progress and game metrics
💾 Automatic model saving with best score and best average tracking
🌐 Cross-platform support (Windows, Linux, Mac)
🔄 Experience replay and target networks for stable learning
🎖️ Custom reward shaping for faster convergence

📋 Table of Contents

Prerequisites
Installation
Unity Setup
Quick Start
Project Structure
Training Modes
Monitoring
Configuration
Troubleshooting
Advanced Usage
Performance Tips
Contributing

🔧 Prerequisites
Software Requirements

Unity 2021.3+ (LTS recommended)
Python 3.8+
CUDA-compatible GPU (optional, for faster training)
8GB+ RAM (16GB recommended for large batch sizes)
5GB free disk space (for models and logs)

Unity Requirements
Your Unity project must implement the following interface:
csharpCopypublic interface IPlayerInputController
{
    bool GetLeft();
    bool GetRight();
    bool GetDown();
    bool GetRotateLeft();
    bool GetRotateRight();
    bool GetHardDrop();
}
Python Dependencies
All dependencies are automatically installed via setup scripts:

PyTorch 1.9+
TensorBoard 2.7+
NumPy 1.21+
Matplotlib 3.3+
Pandas 1.3+
Scikit-learn 1.0+

🚀 Installation
Option 1: Automated Setup (Recommended)
Windows
cmdCopy# Download project files and run setup
setup.bat
Linux/Mac
bashCopy# Make scripts executable and run setup
chmod +x *.sh
./setup.sh
Option 2: Manual Installation
bashCopy# Clone repository
git clone <repository-url>
cd tetris-ai-training

# Create project directories
mkdir -p runs models logs data

# Install Python dependencies
pip install -r requirements.txt

# For CUDA support (optional)
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
🎮 Unity Setup
1. Scene Setup
Create a new Unity scene with the following hierarchy:
CopyTetrisSocketTraining
├── Main Camera
├── SocketManager          (Add SocketManager script)
├── MainThreadDispatcher   (Add UnityMainThreadDispatcher script)
├── GameManager           (Add SocketBoardManager script)
└── Canvas (Optional)
    ├── ScoreText
    └── StatusText
2. Component Configuration
SocketManager Settings
csharpCopyPort: 12345
Host: "127.0.0.1"
SocketBoardManager Settings
csharpCopyBoard Prefab: [Assign your Tetris board prefab]
Board Position: (0, 0, 0)
Use Direct Placement: true
Auto Setup Camera: true
Show Debug Info: true
3. Required Unity Scripts
Copy these scripts to your Unity project:
CopyAssets/Scripts/
├── Socket/
│   ├── SocketManager.cs
│   ├── UnityMainThreadDispatcher.cs
│   ├── SocketTetrisAgent.cs
│   └── SocketBoardManager.cs
├── DataStructures/
│   ├── GameState.cs
│   ├── GameCommand.cs
│   ├── ActionData.cs
│   ├── CurriculumData.cs
│   └── ResetData.cs
└── Board/
    └── [Your existing Tetris scripts]
🎯 Quick Start
Step 1: Start Unity

Open Unity project with Tetris implementation
Load the training scene
Press Play in Unity
Verify console shows: "✓ Unity Socket Server started successfully on port 12345"

Step 2: Start Training
Windows - Interactive Menu
cmdCopytrain.bat
Choose from:

1: New training (fresh start)
2: Continue existing training
3: Evaluate existing model
4: Quick test (100 episodes)

Linux/Mac - Interactive Menu
bashCopy./train.sh
One-Click Full Training
cmdCopy# Windows
launch_training.bat

# Linux/Mac
./launch_training.sh
Step 3: Monitor Progress
cmdCopy# Windows
tensorboard.bat

# Linux/Mac
./tensorboard.sh
Open http://localhost:6006 in browser
Expected Output
Copy✓ Connected to Unity at 127.0.0.1:12345
✓ Action space: 40 actions (10 columns × 4 rotations)
Starting training for 2000 episodes...

Episode 0: Score=120, Lines=3, Steps=45, Reward=15.50, ε=0.995
Episode 10: Score=340, Lines=8, Steps=72, Avg10_Score=180.5, ε=0.950
Episode 100: Score=890, Lines=22, Steps=156, Avg10_Score=445.2, ε=0.605
...
📁 Project Structure
Copytetris-ai-training/
├── 📁 scripts/                    # Python training scripts
│   ├── 🐍 tetris_client.py        # Unity communication client
│   ├── 🧠 dqn_agent.py           # Deep Q-Network implementation
│   ├── 🎯 tetris_trainer.py      # Main training loop
│   └── 🚀 train_tetris.py        # Training entry point
├── 📁 unity_scripts/              # Unity C# scripts
│   ├── 🔌 SocketManager.cs        # Socket server
│   ├── 🤖 SocketTetrisAgent.cs    # AI agent controller
│   ├── 🧵 UnityMainThreadDispatcher.cs
│   └── 📊 data_structures/
│       ├── GameState.cs
│       ├── GameCommand.cs
│       └── ...
├── 📁 models/                      # Saved AI models
│   ├── 💾 tetris_model_YYYYMMDD_HHMMSS.pth
│   ├── 🏆 tetris_model_best_score.pth
│   └── 📈 tetris_model_best_avg.pth
├── 📁 runs/                        # TensorBoard logs
│   ├── 📊 training_YYYYMMDD_HHMMSS/
│   ├── 🔍 eval_YYYYMMDD_HHMMSS/
│   └── ...
├── 📁 logs/                        # Training text logs
├── 📁 data/                        # Exported training data
├── 📁 batch_scripts/               # Windows automation
│   ├── ⚙️ setup.bat
│   ├── 🎯 train.bat
│   ├── 📊 tensorboard.bat
│   ├── 🚀 launch_training.bat
│   └── 📋 evaluate_model.bat
├── 📁 shell_scripts/               # Linux/Mac automation
│   ├── ⚙️ setup.sh
│   ├── 🎯 train.sh
│   ├── 📊 tensorboard.sh
│   ├── 🚀 launch_training.sh
│   ├── 📋 evaluate_model.sh
│   └── 📈 monitor.sh
├── 📄 requirements.txt
└── 📖 README.md
🎮 Training Modes
1. New Training
Starts fresh training with randomly initialized neural network:
bashCopypython train_tetris.py --mode train --episodes 2000
2. Continue Training
Resumes training from existing model checkpoint:
bashCopypython train_tetris.py --mode continue --episodes 1000 --model_path models/tetris_model.pth
3. Evaluation Mode
Tests trained model performance without learning:
bashCopypython train_tetris.py --mode evaluate --eval_episodes 20 --model_path models/tetris_model_best_score.pth
4. Custom Training
Advanced training with custom parameters:
bashCopypython train_tetris.py \
    --mode train \
    --episodes 5000 \
    --model_path models/custom_model.pth \
    --tensorboard_dir runs/custom_training
📊 Monitoring with TensorBoard
TensorBoard provides comprehensive real-time monitoring:
🎯 Training Metrics

Loss: Neural network training loss
Epsilon: Exploration rate (decreases over time)
Learning Rate: Optimizer learning rate
Q-Values: Network output statistics
Gradient Norms: Training stability indicators

🎮 Episode Metrics

Scores: Game scores per episode
Lines Cleared: Tetris lines completed
Episode Length: Steps per game
Rewards: Cumulative rewards earned

🏆 Performance Tracking

Running Averages: 10-episode and 100-episode moving averages
Best Models: Tracking of highest scores achieved
Evaluation Results: Periodic testing without exploration

📈 Game Analysis

Board State: Holes count, stack height
Action Distribution: Which moves the AI prefers
Curriculum Progress: Learning difficulty progression

Accessing TensorBoard
bashCopy# Start TensorBoard
tensorboard --logdir runs/

# Or use provided scripts
./tensorboard.sh  # Linux/Mac
tensorboard.bat   # Windows
Navigate to http://localhost:6006 to view metrics.
⚙️ Configuration
Training Parameters
Edit dqn_agent.py to customize:
pythonCopyclass DQNAgent:
    def __init__(self):
        self.learning_rate = 0.001      # Neural network learning rate
        self.batch_size = 64            # Training batch size
        self.epsilon_decay = 0.9995     # Exploration decay rate
        self.target_update_freq = 1000  # Target network update frequency
        self.memory_size = 50000        # Experience replay buffer size
Curriculum Learning
Modify tetris_trainer.py:
pythonCopyself.curriculum_stages = [
    {'episodes': 200, 'height': 10, 'preset': 1, 'pieces': 3},  # Easy
    {'episodes': 300, 'height': 15, 'preset': 2, 'pieces': 5},  # Medium
    {'episodes': 500, 'height': 20, 'preset': 3, 'pieces': 7},  # Hard
]
Reward Function
Customize in tetris_trainer.py:
pythonCopydef calculate_reward(self, prev_state, current_state, action):
    reward = current_state.get('reward', 0)
    
    # Line clear bonus (exponential)
    lines_cleared = current_state.get('linesCleared', 0) - prev_state.get('linesCleared', 0)
    reward += lines_cleared ** 2 * 10
    
    # Hole penalty
    holes_created = current_state.get('holesCount', 0) - prev_state.get('holesCount', 0)
    reward -= holes_created * 5
    
    return reward
🔧 Troubleshooting
Common Issues
❌ "Connection Refused" Error
Problem: Python can't connect to Unity
Solutions:

Ensure Unity is running and playing the scene
Check Unity console for "Socket Server started" message
Verify port 12345 isn't blocked by firewall
Try changing port in both Unity and Python

❌ "No Module Named torch" Error
Problem: PyTorch not installed
Solution:
bashCopypip install torch torchvision tensorboard
❌ Unity Freezes During Training
Problem: Too many rapid actions
Solutions:

Increase actionCooldown in SocketTetrisAgent
Add delays in action execution coroutine
Reduce training speed in Python

❌ Poor Training Performance
Problem: AI not learning effectively
Solutions:

Check reward function for proper incentives
Verify curriculum learning progression
Adjust exploration rate (epsilon decay)
Increase training episodes

❌ TensorBoard Not Loading
Problem: No data in TensorBoard
Solutions:

Ensure training has started and logged data
Check correct log directory path
Refresh browser page
Restart TensorBoard with correct --logdir

Debug Mode
Enable detailed logging:
pythonCopy# In tetris_trainer.py
logging.basicConfig(level=logging.DEBUG)

# In dqn_agent.py
self.debug_mode = True
Performance Optimization
pythonCopy# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Adjust batch size based on GPU memory
batch_size = 128 if device == 'cuda' else 32

# Enable mixed precision training (advanced)
from torch.cuda.amp import autocast, GradScaler
🚀 Advanced Usage
Custom Neural Network Architecture
pythonCopyclass CustomTetrisDQN(nn.Module):
    def __init__(self):
        super().__init__()
        # Add your custom layers here
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            # ... more layers
        )
Multi-Agent Training
Train multiple agents simultaneously:
pythonCopy# Create multiple Unity instances on different ports
agents = [
    DQNAgent(port=12345),
    DQNAgent(port=12346),
    DQNAgent(port=12347),
]
Hyperparameter Optimization
Use grid search or random search:
pythonCopyhyperparams = {
    'learning_rate': [0.001, 0.0005, 0.0001],
    'batch_size': [32, 64, 128],
    'epsilon_decay': [0.995, 0.9975, 0.999],
}
Export Trained Model
pythonCopy# Save for deployment
torch.jit.save(torch.jit.script(model), 'tetris_model_deployed.pt')

# Load in Unity (with ML-Agents)
# Or convert to ONNX format
torch.onnx.export(model, dummy_input, 'tetris_model.onnx')
📈 Performance Tips
Training Speed

Use GPU: Ensure CUDA is properly installed
Batch Size: Increase batch size for GPU training
Parallel Processing: Train multiple agents simultaneously
Efficient Data: Use efficient data loading and preprocessing

Memory Management
pythonCopy# Clear GPU cache periodically
torch.cuda.empty_cache()

# Use gradient checkpointing for large models
model = torch.utils.checkpoint.checkpoint_sequential(model, segments=2)
Training Stability

Gradient Clipping: Prevent exploding gradients
Learning Rate Scheduling: Decay learning rate over time
Target Network: Use separate target network for stable Q-learning
Experience Replay: Shuffle training data for better convergence

🤝 Contributing
Development Setup
bashCopy# Fork the repository
git clone https://github.com/your-username/tetris-ai-training.git

# Create development branch
git checkout -b feature/your-feature-name

# Install development dependencies
pip install -r requirements-dev.txt
Code Style

Follow PEP 8 for Python code
Use Unity C# conventions for Unity scripts
Add docstrings to all functions
Include type hints where possible

Testing
bashCopy# Run Python tests
python -m pytest tests/

# Test Unity scripts in Unity Test Runner
Pull Request Process

Update documentation for new features
Add tests for new functionality
Ensure all tests pass
Update README.md if needed
Submit pull request with clear description

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

Unity Technologies for the Unity game engine
PyTorch Team for the deep learning framework
OpenAI for reinforcement learning research and inspiration
Tetris Holdings for the classic game concept

📞 Support
Getting Help

Issues: Report bugs on GitHub Issues
Discussions: Ask questions in GitHub Discussions
Documentation: Check this README and code comments
Community: Join our Discord server (link in repository)

Useful Resources

Deep Reinforcement Learning Course
PyTorch Tutorials
Unity ML-Agents
TensorBoard Guide


Happy Training! 🎮🤖
Built with ❤️ for the AI and gaming community