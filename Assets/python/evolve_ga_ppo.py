import os
import subprocess
import yaml
import random
import time
import json
import glob
from deap import base, creator, tools, algorithms
from tensorboard.backend.event_processing import event_accumulator
from ruamel.yaml import YAML

# Constants
CONFIG_TEMPLATE_PATH = "base_config.yaml"
CONFIG_DIR = "configs2"
RESULTS_DIR = "results2"
ENV_PATH = "./../../Build/tetris.app"
MAX_STEPS = 50000
POP_SIZE = 20
N_GEN = 15

# Search space for hyperparams + reward weights
PARAM_BOUNDS = {
    # PPO Hyperparameters
    "learning_rate": (1e-5, 5e-4),
    "batch_size": (256, 512),        # integer
    "beta": (1e-4, 0.05),
    "epsilon": (0.1, 0.5),            # PPO clip param
    "lambd": (0.9, 1.0),
    "num_epoch": (3, 15),             # integer epochs

    # Reward Weights from RewardWeights class
    "clearReward": (0.5, 2.0),
    "comboMultiplier": (0.5, 3.0),
    "perfectClearBonus": (0.0, 5.0),
    "stagnationPenaltyFactor": (0.0, 1.0),
    "roughnessRewardMultiplier": (0.0, 2.0),
    "roughnessPenaltyMultiplier": (0.0, 2.0),
    "holeFillReward": (0.0, 3.0),
    "holeCreationPenalty": (0.0, 3.0),
    "wellRewardMultiplier": (0.0, 2.0),
    "iPieceInWellBonus": (0.0, 2.0),
    "stackHeightPenalty": (0.0, 1.0),
    "uselessRotationPenalty": (0.0, 1.0),
    "tSpinReward": (0.0, 3.0),
    "iPieceGapFillBonus": (0.0, 2.0),
    "accessibilityRewardMultiplier": (0.0, 2.0),
    "accessibilityPenaltyMultiplier": (0.0, 2.0),
    "deathPenalty": (-5.0, 0.0),  # negative penalty

    "idleActionPenalty": (-1.0, 0.0),       # Penalty (negative or zero)
    "moveDownActionReward": (0.0, 1.0),
    "hardDropActionReward": (0.0, 2.0),

    "doubleLineClearRewardMultiplier": (0.5, 3.0),
    "tripleLineClearRewardMultiplier": (1.0, 4.0),
    "tetrisClearRewardMultiplier": (2.0, 5.0),
    "maxWellRewardCap": (1.0, 10.0),
}

# DEAP setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", dict, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Individual generator
def generate_individual():
    individual = {}
    for key, bounds in PARAM_BOUNDS.items():
        if key in ["batch_size", "num_epoch"]:
            # Integer parameters
            individual[key] = random.randint(bounds[0], bounds[1])
        else:
            individual[key] = random.uniform(bounds[0], bounds[1])
    return creator.Individual(individual)

toolbox.register("individual", generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Crossover for dict individuals (uniform crossover)
def cx_dict(ind1, ind2):
    for key in ind1.keys():
        if random.random() < 0.5:
            ind1[key], ind2[key] = ind2[key], ind1[key]
    return ind1, ind2

toolbox.register("mate", cx_dict)

# Mutation for dict individuals
def mut_dict(individual, indpb=0.2):
    for key, bounds in PARAM_BOUNDS.items():
        if random.random() < indpb:
            if key in ["batch_size", "num_epoch"]:
                # Mutate integer param by adding small int delta
                delta = random.randint(-50, 50) if key == "batch_size" else random.randint(-2, 2)
                new_val = individual[key] + delta
                # Clamp to bounds
                individual[key] = max(bounds[0], min(bounds[1], new_val))
            else:
                # Mutate continuous param with gaussian noise
                mu = 0
                sigma = (bounds[1] - bounds[0]) * 0.1
                new_val = individual[key] + random.gauss(mu, sigma)
                # Clamp to bounds
                individual[key] = max(bounds[0], min(bounds[1], new_val))
    return individual,

toolbox.register("mutate", mut_dict, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=2)

# Write modified config for Unity ML-Agents
import os
import yaml

def write_config(ind, run_id):
    with open(CONFIG_TEMPLATE_PATH) as f:
        yaml_loader = YAML()
        config = yaml_loader.load(f)

    # PPO hyperparameters
    hparams = config["behaviors"]["TetrisAgent"]["hyperparameters"]
    hparams["learning_rate"] = float(ind["learning_rate"])
    hparams["batch_size"] = int(ind["batch_size"])
    hparams["beta"] = float(ind["beta"])
    hparams["epsilon"] = float(ind.get("epsilon", 0.3))
    hparams["lambd"] = float(ind.get("lambd", 0.98))
    hparams["num_epoch"] = int(ind.get("num_epoch", 3))
    config["behaviors"]["TetrisAgent"]["max_steps"] = MAX_STEPS

    # Optional: Add dynamic environment parameters
    # Uncomment if needed and `ind` contains them
    if "environment_parameters" not in config:
        config["environment_parameters"] = {}

    reward_params = [
        "clearReward", "comboMultiplier", "perfectClearBonus", "stagnationPenaltyFactor",
        "roughnessRewardMultiplier", "roughnessPenaltyMultiplier", "holeFillReward",
        "holeCreationPenalty", "wellRewardMultiplier", "iPieceInWellBonus",
        "stackHeightPenalty", "uselessRotationPenalty", "tSpinReward",
        "iPieceGapFillBonus", "accessibilityRewardMultiplier", "accessibilityPenaltyMultiplier",
        "deathPenalty", "idleActionPenalty", "moveDownActionReward", "hardDropActionReward",
        "doubleLineClearRewardMultiplier", "tripleLineClearRewardMultiplier",
        "tetrisClearRewardMultiplier", "maxWellRewardCap"
    ]

    for param in reward_params:
        if param in ind:
            config["environment_parameters"][param] = float(ind[param])

    os.makedirs(CONFIG_DIR, exist_ok=True)
    path = os.path.join(CONFIG_DIR, f"run_{run_id}.yaml")

    yaml_dumper = YAML()
    yaml_dumper.indent(mapping=2, sequence=4, offset=2)  # Important!
    with open(path, "w") as f:
        f.write('---\n')  # Add YAML document start
        yaml_dumper.dump(config, f)

    return path


# def write_config(ind, run_id):
#     with open(CONFIG_TEMPLATE_PATH) as f:
#         yaml_loader = YAML()
#         config = yaml_loader.load(f)

#     hparams = config["behaviors"]["TetrisAgent"]["hyperparameters"]
#     hparams["learning_rate"] = float(ind["learning_rate"])
#     hparams["batch_size"] = int(ind["batch_size"])
#     hparams["beta"] = float(ind["beta"])
#     hparams["epsilon"] = float(ind.get("epsilon", 0.3))
#     hparams["lambd"] = float(ind.get("lambd", 0.98))
#     hparams["num_epoch"] = int(ind.get("num_epoch", 3))
#     config["behaviors"]["TetrisAgent"]["max_steps"] = MAX_STEPS

#     os.makedirs(CONFIG_DIR, exist_ok=True)
#     path = os.path.join(CONFIG_DIR, f"run_{run_id}.yaml")

#     yaml_dumper = YAML()
#     yaml_dumper.indent(mapping=2, sequence=4, offset=2)  # Important!
#     with open(path, "w") as f:
#         f.write('---\n')  # Add YAML document start
#         yaml_dumper.dump(config, f)

#     return path


def train_and_evaluate(ind, run_id):
    config_path = write_config(ind, run_id)

    print(f"🔧 Starting training for Run {run_id}")

    # Run mlagents-learn with config, environment, no graphics, train mode
    cmd = [
        "mlagents-learn", config_path,
        "--run-id", run_id,
        "--env", ENV_PATH,
        "--no-graphics",
        "--train"
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Print output for debugging if needed
    if result.returncode != 0:
        print(f"❌ Training failed for {run_id}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return 0.0

    return read_fitness_from_summary(run_id)

def read_fitness_from_summary(run_id):
    # Try multiple possible log directory structures
    possible_log_dirs = [
        os.path.join("results", run_id, "TetrisAgent"),
        os.path.join("results", run_id),
        os.path.join("result", run_id, "TetrisAgent"),  # Note: singular "result"
        os.path.join("result", run_id)
    ]
    
    log_dir = None
    for dir_path in possible_log_dirs:
        if os.path.exists(dir_path):
            # Check if it contains event files
            event_files = glob.glob(os.path.join(dir_path, "events.out.tfevents.*"))
            if event_files:
                log_dir = dir_path
                break
    
    if not log_dir:
        print(f"⚠️ Log directory not found for run {run_id}")
        print("Searched in:")
        for dir_path in possible_log_dirs:
            print(f"  - {dir_path} (exists: {os.path.exists(dir_path)})")
        
        # List contents of results directory for debugging
        if os.path.exists("results"):
            print("\nContents of results directory:")
            for item in os.listdir("results"):
                print(f"  - {item}")
                if os.path.isdir(os.path.join("results", item)):
                    subdir_path = os.path.join("results", item)
                    print(f"    Contents of {item}:")
                    for subitem in os.listdir(subdir_path):
                        print(f"      - {subitem}")
        
        return 0.0

    print(f"📊 Found logs in {log_dir}")

    ea = event_accumulator.EventAccumulator(log_dir)
    try:
        ea.Reload()
    except Exception as e:
        print(f"⚠️ Failed to load TensorBoard logs: {e}")
        return 0.0

    scalar_tags = ea.Tags().get('scalars', [])
    if not scalar_tags:
        print("⚠️ No scalar tags found in TensorBoard logs.")
        return 0.0

    print("Available scalar tags:")
    for tag in scalar_tags:
        print(f"  - {tag}")

    # Look for cumulative reward - try different possible tag names
    reward_tag = None
    possible_reward_tags = [
        "Environment/Cumulative Reward",
        "TetrisAgent/Environment/Cumulative Reward", 
        "Cumulative Reward",
        "Environment/Episode Length",
        "Policy/Extrinsic Reward"
    ]
    
    for tag in possible_reward_tags:
        if tag in scalar_tags:
            reward_tag = tag
            break
    
    # Fallback: look for any tag containing "reward" and "cumulative"
    if not reward_tag:
        for tag in scalar_tags:
            if "cumulative" in tag.lower() and "reward" in tag.lower():
                reward_tag = tag
                break

    if not reward_tag:
        print("⚠️ Reward tag not found in TensorBoard logs.")
        print("Available tags:", scalar_tags)
        return 0.0

    print(f"Using reward tag: {reward_tag}")
    events = ea.Scalars(reward_tag)
    if not events:
        print("⚠️ No scalar events found for reward tag.")
        return 0.0

    # Get the mean of the last 10% of values for more stable fitness
    num_values = len(events)
    if num_values >= 10:
        last_values = [events[i].value for i in range(int(num_values * 0.9), num_values)]
        fitness = sum(last_values) / len(last_values)
    else:
        fitness = events[-1].value
    
    print(f"📈 Computed fitness for {run_id}: {fitness}")
    return fitness

# Fitness function for DEAP
def evaluate(ind):
    run_id = f"GA_Run_{int(time.time())}_{random.randint(1000, 9999)}"
    fitness = train_and_evaluate(ind, run_id)
    print(f"🏅 Run {run_id} fitness: {fitness}")
    return (fitness,)

toolbox.register("evaluate", evaluate)

# Save best individuals to file
def save_best_individuals(hof, generation):
    os.makedirs("ga_results", exist_ok=True)
    with open(f"ga_results/best_individuals_gen_{generation}.json", "w") as f:
        best_data = []
        for i, ind in enumerate(hof):
            best_data.append({
                "rank": i + 1,
                "fitness": ind.fitness.values[0],
                "parameters": dict(ind)
            })
        json.dump(best_data, f, indent=2)

# Main GA loop
if __name__ == "__main__":
    print(f"🚀 Starting Genetic Algorithm with {POP_SIZE} individuals for {N_GEN} generations")
    
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(5)  # Keep top 5 individuals

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: round(sum(v[0] for v in x) / len(x), 2))
    stats.register("max", lambda x: round(max(v[0] for v in x), 2))
    stats.register("min", lambda x: round(min(v[0] for v in x), 2))

    # Custom algorithm with checkpointing
    for gen in range(N_GEN):
        print(f"\n🧬 Generation {gen + 1}/{N_GEN}")
        
        # Evaluate population
        fitnesses = map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Update hall of fame
        hof.update(pop)
        
        # Record statistics
        record = stats.compile(pop)
        print(f"📊 Gen {gen + 1} - Max: {record['max']}, Avg: {record['avg']}, Min: {record['min']}")
        
        # Save best individuals every few generations
        if (gen + 1) % 3 == 0:
            save_best_individuals(hof, gen + 1)
        
        # Selection and reproduction (skip on last generation)
        if gen < N_GEN - 1:
            # Select parents
            parents = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, parents))
            
            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:  # crossover probability
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < 0.3:  # mutation probability
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            pop[:] = offspring

    print("\n🏆 Final Results:")
    print("="*50)
    for i, ind in enumerate(hof):
        print(f"Rank {i+1}: Fitness = {ind.fitness.values[0]:.2f}")
        print(f"Parameters: {dict(ind)}")
        print("-" * 30)
    
    # Save final results
    save_best_individuals(hof, N_GEN)
    print(f"\n💾 Results saved to ga_results/best_individuals_gen_{N_GEN}.json")