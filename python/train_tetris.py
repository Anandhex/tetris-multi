# train_tetris.py
import argparse
import os
import sys
from tetris_trainer import TetrisTrainer

def main():
    parser = argparse.ArgumentParser(description='Train Tetris AI Agent')
    parser.add_argument('--mode', choices=['train', 'continue', 'evaluate'], default='train',
                       help='Training mode')
    parser.add_argument('--episodes', type=int, default=10000000,
                       help='Number of episodes to train')
    parser.add_argument('--model_path', type=str, default='tetris_model.pth',
                       help='Path to save/load model')
    parser.add_argument('--tensorboard_dir', type=str, default=None,
                       help='TensorBoard log directory')
    parser.add_argument('--eval_episodes', type=int, default=10,
                       help='Number of episodes for evaluation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TETRIS AI TRAINING WITH TENSORBOARD")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Episodes: {args.episodes}")
    print(f"Model Path: {args.model_path}")
    print(f"TensorBoard Dir: {args.tensorboard_dir}")
    print("=" * 60)
    
    if args.mode == 'train':
        print("Starting new training session...")
        trainer = TetrisTrainer(
            agent_type='dqn', 
            load_model=False, 
            model_path=args.model_path,
            tensorboard_log_dir=args.tensorboard_dir
        )
        trainer.train(episodes=sys.maxsize)
        
    elif args.mode == 'continue':
        if not os.path.exists(args.model_path):
            print(f"Model file {args.model_path} not found!")
            return
        
        print("Continuing training from existing model...")
        trainer = TetrisTrainer(
            agent_type='dqn', 
            load_model=True, 
            model_path=args.model_path,
            tensorboard_log_dir=args.tensorboard_dir
        )
        trainer.train(episodes=args.episodes)
        
    elif args.mode == 'evaluate':
        if not os.path.exists(args.model_path):
            print(f"Model file {args.model_path} not found!")
            return
        
        print("Evaluating existing model...")
        trainer = TetrisTrainer(
            agent_type='dqn', 
            load_model=True, 
            model_path=args.model_path,
            tensorboard_log_dir=args.tensorboard_dir
        )
        
        if trainer.client.connect():
            trainer.evaluate(episodes=args.eval_episodes)
            trainer.client.disconnect()
        
    print("\nTo view TensorBoard logs, run:")
    if args.tensorboard_dir:
        print(f"tensorboard --logdir {args.tensorboard_dir}")
    else:
        print("tensorboard --logdir runs/")

if __name__ == "__main__":
    main()