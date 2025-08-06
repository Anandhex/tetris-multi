# dataset_analyzer.py - Analyze .pkl dataset for unique states
import pickle
import numpy as np
from collections import defaultdict
import hashlib

def analyze_dataset(dataset_path: str):
    """
    Analyze a .pkl dataset to count unique board configurations and powerup combinations
    """
    print(f"Loading dataset from: {dataset_path}")
    
    # Load dataset
    try:
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        print(f"✅ Dataset loaded successfully: {len(dataset)} total samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Initialize counters
    unique_boards = set()
    unique_powerup_combinations = set()
    unique_states = set()  # board + powerup combination
    board_type_counts = defaultdict(int)
    powerup_scenario_counts = defaultdict(int)
    
    # Track powerup availability statistics
    powerup_stats = {
        'bottom_clear': {'available': 0, 'total': 0},
        'gravity': {'available': 0, 'total': 0},
        'bomb': {'available': 0, 'total': 0}
    }
    
    powerup_combination_counts = defaultdict(int)
    
    print("\n🔍 Analyzing dataset...")
    
    for i, sample in enumerate(dataset):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(dataset)} samples...")
        
        # Extract board and powerups
        if isinstance(sample, dict):
            board = np.array(sample['board'], dtype=np.int32)
            powerups = sample.get('powerups', {})
            board_type = sample.get('board_type', 'unknown')
            powerup_scenario = sample.get('powerup_scenario', 'unknown')
        else:
            # If sample is just a board array
            board = np.array(sample, dtype=np.int32)
            powerups = {'bottom_clear': False, 'gravity': False, 'bomb': False}
            board_type = 'unknown'
            powerup_scenario = 'unknown'
        
        # Create unique identifiers
        board_hash = hashlib.md5(board.tobytes()).hexdigest()
        powerup_tuple = tuple(sorted(powerups.items()))
        state_hash = hashlib.md5((board_hash + str(powerup_tuple)).encode()).hexdigest()
        
        # Add to sets
        unique_boards.add(board_hash)
        unique_powerup_combinations.add(powerup_tuple)
        unique_states.add(state_hash)
        
        # Count board types
        board_type_counts[board_type] += 1
        powerup_scenario_counts[powerup_scenario] += 1
        
        # Track powerup statistics
        for powerup, available in powerups.items():
            if powerup in powerup_stats:
                powerup_stats[powerup]['total'] += 1
                if available:
                    powerup_stats[powerup]['available'] += 1
        
        # Count powerup combinations
        available_powerups = tuple(sorted([p for p, available in powerups.items() if available]))
        powerup_combination_counts[available_powerups] += 1
    
    print(f"✅ Analysis complete!")
    
    # Print results
    print("\n" + "="*60)
    print("📊 DATASET ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"  Total samples: {len(dataset):,}")
    print(f"  Unique board configurations: {len(unique_boards):,}")
    print(f"  Unique powerup combinations: {len(unique_powerup_combinations):,}")
    print(f"  Unique states (board + powerup): {len(unique_states):,}")
    
    # Calculate uniqueness percentages
    board_uniqueness = (len(unique_boards) / len(dataset)) * 100
    state_uniqueness = (len(unique_states) / len(dataset)) * 100
    
    print(f"\n📊 UNIQUENESS RATIOS:")
    print(f"  Board uniqueness: {board_uniqueness:.1f}%")
    print(f"  State uniqueness: {state_uniqueness:.1f}%")
    
    if board_uniqueness < 50:
        print("  ⚠️  WARNING: Low board uniqueness - many duplicate boards!")
    if state_uniqueness < 70:
        print("  ⚠️  WARNING: Low state uniqueness - many duplicate states!")
    
    print(f"\n🎮 BOARD TYPE DISTRIBUTION:")
    for board_type, count in sorted(board_type_counts.items()):
        percentage = (count / len(dataset)) * 100
        print(f"  {board_type}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n⚡ POWERUP SCENARIO DISTRIBUTION:")
    for scenario, count in sorted(powerup_scenario_counts.items()):
        percentage = (count / len(dataset)) * 100
        print(f"  {scenario}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n💪 POWERUP AVAILABILITY STATISTICS:")
    for powerup, stats in powerup_stats.items():
        if stats['total'] > 0:
            availability_pct = (stats['available'] / stats['total']) * 100
            print(f"  {powerup}: {stats['available']:,}/{stats['total']:,} ({availability_pct:.1f}% available)")
    
    print(f"\n🔧 POWERUP COMBINATION FREQUENCIES:")
    sorted_combinations = sorted(powerup_combination_counts.items(), key=lambda x: x[1], reverse=True)
    for combination, count in sorted_combinations:
        percentage = (count / len(dataset)) * 100
        if not combination:
            combo_name = "No powerups"
        else:
            combo_name = " + ".join(combination)
        print(f"  {combo_name}: {count:,} ({percentage:.1f}%)")
    
    # Detect potential issues
    print(f"\n🚨 POTENTIAL ISSUES:")
    issues_found = False
    
    # Check for no-powerup scenarios
    no_powerup_count = powerup_combination_counts.get((), 0)
    if no_powerup_count > 0:
        no_powerup_pct = (no_powerup_count / len(dataset)) * 100
        print(f"  ⚠️  {no_powerup_count:,} samples ({no_powerup_pct:.1f}%) have NO powerups available")
        issues_found = True
    
    # Check for very low uniqueness
    if len(unique_boards) < len(dataset) * 0.3:
        duplicate_pct = ((len(dataset) - len(unique_boards)) / len(dataset)) * 100
        print(f"  ⚠️  {duplicate_pct:.1f}% of boards are duplicates")
        issues_found = True
    
    # Check for powerup imbalance
    powerup_availabilities = []
    for powerup, stats in powerup_stats.items():
        if stats['total'] > 0:
            powerup_availabilities.append(stats['available'] / stats['total'])
    
    if powerup_availabilities:
        min_availability = min(powerup_availabilities)
        max_availability = max(powerup_availabilities)
        if max_availability - min_availability > 0.3:  # 30% difference
            print(f"  ⚠️  Powerup availability imbalanced: {min_availability*100:.1f}% to {max_availability*100:.1f}%")
            issues_found = True
    
    if not issues_found:
        print("  ✅ No major issues detected!")
    
    # Provide recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if len(unique_states) < len(dataset) * 0.8:
        print(f"  📈 Consider generating more diverse board configurations")
    if no_powerup_count > len(dataset) * 0.1:
        print(f"  ⚡ Reduce scenarios with no powerups for better training")
    if len(unique_boards) == len(dataset):
        print(f"  ✅ Excellent board diversity!")
    if all(abs(avail - 0.5) < 0.1 for avail in powerup_availabilities):
        print(f"  ✅ Good powerup balance!")
    
    return {
        'total_samples': len(dataset),
        'unique_boards': len(unique_boards),
        'unique_powerup_combinations': len(unique_powerup_combinations),
        'unique_states': len(unique_states),
        'board_uniqueness_pct': board_uniqueness,
        'state_uniqueness_pct': state_uniqueness,
        'powerup_stats': powerup_stats,
        'powerup_combinations': dict(powerup_combination_counts)
    }

def compare_datasets(dataset_paths: list):
    """Compare multiple datasets"""
    print("🔍 Comparing multiple datasets...")
    
    results = {}
    for path in dataset_paths:
        print(f"\n{'='*60}")
        print(f"Analyzing: {path}")
        print(f"{'='*60}")
        results[path] = analyze_dataset(path)
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("📊 DATASET COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for path, result in results.items():
        print(f"\n{path}:")
        print(f"  Samples: {result['total_samples']:,}")
        print(f"  Unique boards: {result['unique_boards']:,} ({result['board_uniqueness_pct']:.1f}%)")
        print(f"  Unique states: {result['unique_states']:,} ({result['state_uniqueness_pct']:.1f}%)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset_analyzer.py <dataset.pkl> [dataset2.pkl ...]")
        print("Example: python dataset_analyzer.py tetris_boards.pkl")
        sys.exit(1)
    
    dataset_paths = sys.argv[1:]
    
    if len(dataset_paths) == 1:
        # Analyze single dataset
        analyze_dataset(dataset_paths[0])
    else:
        # Compare multiple datasets
        compare_datasets(dataset_paths)