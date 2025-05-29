import os
import subprocess
import yaml
import random
import time
import json
from deap import base, creator, tools, algorithms
from tensorboard.backend.event_processing import event_accumulator

# Constants
CONFIG_TEMPLATE_PATH = "base_config.yaml"
CONFIG_DIR = "configs"
RESULTS_DIR = "results"
ENV_PATH = "./../../Build/tetris.app"
MAX_STEPS = 500_000
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
        if key == "batch_size":
            # batch_size as integer
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
            if key == "batch_size":
                # Mutate integer param by adding small int delta
                delta = random.randint(-50, 50)
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
def write_config(ind, run_id):
    with open(CONFIG_TEMPLATE_PATH) as f:
        config = yaml.safe_load(f)

    # PPO hyperparameters
    hparams = config["behaviors"]["TetrisAgent"]["hyperparameters"]
    hparams["learning_rate"] = float(ind["learning_rate"])
    hparams["batch_size"] = int(ind["batch_size"])
    hparams["beta"] = float(ind["beta"])
    hparams["epsilon"] = float(ind.get("epsilon", 0.3))  # fallback if missing
    hparams["lambd"] = float(ind.get("lambd", 0.98))
    hparams["num_epoch"] = int(ind.get("num_epoch", 3))
    config["behaviors"]["TetrisAgent"]["max_steps"] = MAX_STEPS


    
    os.makedirs(CONFIG_DIR, exist_ok=True)
    path = os.path.join(CONFIG_DIR, f"run_{run_id}.yaml")
    with open(path, "w") as f:
        yaml.dump(config, f)

    return path
# Launch Unity training & wait for completion

def train_and_evaluate(ind, run_id):
    config_path = write_config(ind, run_id)

    # List of reward parameters to include in env parameters
    reward_params = [
        "clearReward",
        "comboMultiplier",
        "perfectClearBonus",
        "stagnationPenaltyFactor",
        "roughnessRewardMultiplier",
        "roughnessPenaltyMultiplier",
        "holeFillReward",
        "holeCreationPenalty",
        "wellRewardMultiplier",
        "iPieceInWellBonus",
        "stackHeightPenalty",
        "uselessRotationPenalty",
        "tSpinReward",
        "iPieceGapFillBonus",
        "accessibilityRewardMultiplier",
        "accessibilityPenaltyMultiplier",
        "deathPenalty",
        "idleActionPenalty",
        "moveDownActionReward",
        "hardDropActionReward",
        "doubleLineClearRewardMultiplier",
        "tripleLineClearRewardMultiplier",
        "tetrisClearRewardMultiplier",
        "maxWellRewardCap"
    ]

    # Build dictionary of env parameters from individual for the reward params
    env_params_dict = {
        param: float(ind[param])
        for param in reward_params if param in ind
    }

    # Convert to JSON string
    env_params_json = json.dumps(env_params_dict)

    # Prepare CLI args - single --env-parameters followed by JSON string
    # env_params_args = ["--env-parameters", env_params_json]

    # print(f"🔧 Starting training for Run {run_id} with env parameters: {env_params_json}")

    # Run mlagents-learn with config, environment, no graphics, train mode, and env parameters
    cmd = [
        "mlagents-learn", config_path,
        "--run-id", run_id,
        "--env", ENV_PATH,
        "--no-graphics",
        "--train"
    ] 
    # + env_params_args

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Uncomment for debugging logs
    # print(result.stdout)
    # print(result.stderr)

    return read_fitness_from_summary(run_id)


# Parse metrics.json summary for cumulative reward
def read_fitness_from_summary(run_id):
    log_dir = os.path.join("results", run_id, "TetrisAgent")
    print(f"Looking for logs in {log_dir} ...")
    if not os.path.exists(log_dir):
        print(f"⚠️ Log directory not found: {log_dir}")
        return 0.0

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

    # Look for cumulative reward
    reward_tag = None
    for tag in scalar_tags:
        if "cumulative" in tag.lower() and "reward" in tag.lower():
            reward_tag = tag
            break

    if not reward_tag:
        print("⚠️ Reward tag not found in TensorBoard logs.")
        return 0.0

    print(f"Using reward tag: {reward_tag}")
    events = ea.Scalars(reward_tag)
    if not events:
        print("⚠️ No scalar events found for reward tag.")
        return 0.0

    last_value = events[-1].value
    print(f"Last recorded reward value: {last_value}")
    return last_value

# Fitness function for DEAP
def evaluate(ind):
    run_id = f"GA_Run_{int(time.time())}_{random.randint(1000, 9999)}"
    fitness = train_and_evaluate(ind, run_id)
    print(f"🏅 Run {run_id} fitness: {fitness}")
    return (fitness,)

toolbox.register("evaluate", evaluate)

# Main GA loop
if __name__ == "__main__":
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: round(sum(v[0] for v in x) / len(x), 2))
    stats.register("max", lambda x: round(max(v[0] for v in x), 2))
    stats.register("min", lambda x: round(min(v[0] for v in x), 2))

    pop, log = algorithms.eaSimple(pop, toolbox,
                                   cxpb=0.5,
                                   mutpb=0.3,
                                   ngen=N_GEN,
                                   stats=stats,
                                   halloffame=hof,
                                   verbose=True)

    print("\n🏆 Best individual:")
    print(hof[0])
