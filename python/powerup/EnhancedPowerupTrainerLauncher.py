#!/usr/bin/env python3
"""
Simple runner script for Enhanced Powerup Training
Just run this file to start the complete training process
"""

import os
import sys
from pathlib import Path

def main():
    """Main function to run enhanced powerup training"""
    
    print("🚀 Enhanced Powerup Training Runner")
    print("="*50)
    
    # Create necessary directories
    directories = ['models', 'checkpoints', 'datasets', 'logs']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    try:
        # Import required components
        print("\n📦 Importing components...")
        from powerup_dqn_network import FinalPowerUpDQNAgent
        from python.powerup.EnhancedPowerupTrainer import ExtendedPowerupTrainer
        print("✅ All components imported successfully")
        
        # Initialize agent
        print("\n🤖 Initializing PowerUp DQN Agent...")
        powerup_agent = FinalPowerUpDQNAgent(
            state_size=7,
            learning_rate=0.001,
            epsilon=0.9,
            epsilon_decay=0.9995,
            epsilon_min=0.02,
            memory_size=50000,
            batch_size=64
        )
        print("✅ Agent initialized")
        
        # Create trainer with configuration
        print("\n⚙️ Setting up trainer...")
        training_config = {
            'total_episodes': 25000,     # Long training
            'batch_episodes': 500,       # Progress tracking
            'visualization_interval': 50, # Frequent updates
            'save_interval': 1000,       # Save checkpoints
            'curriculum_learning': True,
            'adaptive_epsilon': True
        }
        
        trainer = ExtendedPowerupTrainer(powerup_agent, training_config=training_config)
        print("✅ Trainer configured")
        
        # Start training - dataset will be created automatically
        print("\n🎯 Starting training process...")
        print("This will:")
        print("- Generate large realistic dataset (or load existing)")
        print("- Train for 25,000 episodes")
        print("- Show live visualization")
        print("- Keep visualization open after completion")
        print("- Save checkpoints every 1,000 episodes")
        print()
        
        # The dataset path is now optional - it will use a default name
        trainer.extended_training_loop()  # No need to specify path
        
        print("\n🎉 Training completed successfully!")
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\nMake sure you have these files in the same directory:")
        print("- powerup_dqn_network.py")
        print("- enhanced_powerup_trainer.py") 
        print("- enhanced_game_state_generator.py")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n👋 Training session ended")

if __name__ == "__main__":
    main()