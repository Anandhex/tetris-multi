from trainer import PowerupTrainer
import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description='Tetris Powerup DQN - PyTorch Version')
    parser.add_argument('--mode', choices=['generate', 'train', 'test', 'infer'], 
                       required=True, help='Mode to run')
    parser.add_argument('--dataset', default='tetris_board.pkl', 
                       help='Dataset path')
    parser.add_argument('--model', default='powerup_models/powerup_model_final.pth',  # Changed to .pth
                       help='Model path')
    parser.add_argument('--episodes', type=int, default=1000, 
                       help='Training episodes')
    parser.add_argument('--samples', type=int, default=1000, 
                       help='Dataset samples to generate')
    parser.add_argument('--cuda', action='store_true',
                       help='Force CUDA usage (will fail if not available)')
    
    args = parser.parse_args()
    
    # Print PyTorch and CUDA info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Force CUDA check if requested
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available!")
    
    if args.mode == 'generate':
        # Generate dataset
        from dataset_generator import TetrisDatasetGenerator
        generator = TetrisDatasetGenerator()
        generator.generate_dataset(args.samples, args.dataset)
        
    elif args.mode == 'train':
        # Train model
        from trainer import PowerupTrainer
        trainer = PowerupTrainer(args.dataset)
        final_model = trainer.train(episodes=args.episodes)
        print(f"\nTraining completed! Final model: {final_model}")
        
    elif args.mode == 'test':
        # Test model
        from testing import ModelTester
        tester = ModelTester(args.model, args.dataset)
        avg_reward, action_dist = tester.test_model()
        print(f"\nTesting completed! Average reward: {avg_reward:.2f}")
        
    elif args.mode == 'infer':
        # Run inference with Unity
        # from unity_client import UnityTetrisClient
        # from inference import PowerupInference
        
        # unity_client = UnityTetrisClient()
        # if unity_client.connect():
        #     inference = PowerupInference(args.model)
        #     inference.run_inference_loop(unity_client)
        # else:
            print("Failed to connect to Unity")

if __name__ == "__main__":
    main()