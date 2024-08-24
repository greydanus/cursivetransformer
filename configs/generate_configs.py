import itertools
import json
import os

# Ensure we're in the project root directory
if not os.path.exists('configs'):
    raise FileNotFoundError("The 'configs' directory does not exist in the current directory. Please run this script from the project root.")

base_config = {
    "work_dir": "out",
    "resume": False,
    "sample_only": False,
    "num_workers": 1,
    "max_steps": 150000,
    "lr_decay": 1.0,
    "device": "cuda",
    "seed": 42069,
    "top_k": -1,
    "ablate_cross_attention": False,
    "augment": True,
    "max_seq_length": 1250,
    "learning_rate": 1e-2,
    "weight_decay": 1e-4,
    "wandb_project": "cursivetransformer",
    "wandb_entity": "zwimpee",
    "wandb_api_key": "e56bbe426145e5983e72a933299daca195ebb6a7"
}

# Define the parameter ranges to test
param_ranges = {
    "n_layer": [4, 8, 16],
    "n_head": [4, 8, 16],
    "n_embd": [64, 128, 256],
    "batch_size": [256, 512]  # Based on your findings about batch size
}

# Generate all combinations
configs = []
for n_layer, n_head, n_embd, batch_size in itertools.product(
    param_ranges["n_layer"],
    param_ranges["n_head"],
    param_ranges["n_embd"],
    param_ranges["batch_size"]
):
    config = base_config.copy()
    config.update({
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "n_embd2": n_embd,  # Keeping n_embd2 the same as n_embd
        "batch_size": batch_size,
        "wandb_run_name": f"exp_l{n_layer}_h{n_head}_e{n_embd}_b{batch_size}"
    })
    configs.append(config)

# Print summary and save configurations
print(f"Total number of configurations: {len(configs)}")

for config in configs:
    filename = f"config_l{config['n_layer']}_h{config['n_head']}_e{config['n_embd']}_b{config['batch_size']}.json"
    filepath = os.path.join('configs', filename)
    with open(filepath, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved configuration to {filepath}")

# Print a sample configuration
print("\nSample configuration:")
print(json.dumps(configs[0], indent=2))

print("\nAll configurations have been saved in the 'configs' directory.")