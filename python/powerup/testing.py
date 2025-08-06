from powerup_dqn_agent import PowerupDQNAgent
from environments import TrainingEnvironment
from python.powerup.dataset_generator import TetrisDatasetGenerator
from trainer import PowerupTrainer

class ModelTester:
    """Test trained powerup model"""
    
    def __init__(self, model_path: str, dataset_path: str):
        self.agent = PowerupDQNAgent()
        self.agent.load_model(model_path)
        self.agent.epsilon = 0  # No exploration during testing
        
        self.test_env = TrainingEnvironment(dataset_path)
        
    def test_model(self, num_tests: int = 100):
        """Test model performance on dataset"""
        total_reward = 0
        action_counts = {'none': 0, 'bottom_clear': 0, 'gravity': 0, 'bomb': 0}
        
        print(f"Testing model on {num_tests} scenarios...")
        
        for test in range(num_tests):
            self.test_env.reset()
            
            # Test single decision
            action = self.agent.choose_action(self.test_env)
            _, reward = self.test_env.apply_powerup(action)
            
            total_reward += reward
            action_counts[action['type']] += 1
            
            if test % 20 == 0:
                print(f"Test {test}: Action={action['type']}, Reward={reward:.2f}")
        
        avg_reward = total_reward / num_tests
        
        print(f"\nTest Results:")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Action Distribution: {action_counts}")
        
        return avg_reward, action_counts


# Usage examples and main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Tetris Powerup DQN')
    parser.add_argument('--mode', choices=['generate', 'train', 'test', 'infer'], 
                       required=True, help='Mode to run')
    parser.add_argument('--dataset', default='tetris_boards.pkl', 
                       help='Dataset path')
    parser.add_argument('--model', default='powerup_models/powerup_model_final.h5', 
                       help='Model path')
    parser.add_argument('--episodes', type=int, default=1000, 
                       help='Training episodes')
    parser.add_argument('--samples', type=int, default=1000, 
                       help='Dataset samples to generate')
    
    args = parser.parse_args()
    
    if args.mode == 'generate':
        # Generate dataset
        generator = TetrisDatasetGenerator()
        generator.generate_dataset(args.samples, args.dataset)
        
    elif args.mode == 'train':
        # Train model
        trainer = PowerupTrainer(args.dataset)
        trainer.train(episodes=args.episodes)
        
    elif args.mode == 'test':
        # Test model
        tester = ModelTester(args.model, args.dataset)
        tester.test_model()
        
    elif args.mode == 'infer':
        # Run inference with Unity
        # unity_client = UnityTetrisClient()
        # if unity_client.connect():
        #     inference = PowerupInference(args.model)
        #     inference.run_inference_loop(unity_client)
        # else:
            print("Failed to connect to Unity")