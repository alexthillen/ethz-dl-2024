import os
from itertools import product
import argparse

# Add at the beginning of the file
parser = argparse.ArgumentParser()
parser.add_argument(
    "--loss_type",
    choices=["activation_preserving", "saliency_preserving", "both"],
    default="both",
    help="Which loss type to run experiments for",
)
args = parser.parse_args()

# Define parameter spaces for each loss type
parameter_space = {
    "activation_preserving": {"saliency_lambda": [0.2, 0.4, 0.5], "num_samples_per_class": [5]},
    "saliency_preserving": {
        "saliency_lambda": [70, 65],
        "num_samples_per_class": [3, 6],
        "noise_std": [0.075, 0.1],
        "saliency_threshold": [0.5, 0.6],
    },
}


def run_experiments(loss_type):
    params = parameter_space[loss_type]
    param_names = list(params.keys())
    param_values = list(params.values())

    for values in product(*param_values):
        param_dict = dict(zip(param_names, values))

        command_parts = [f"python ./train.py --loss_type {loss_type} --num_epochs 8"]
        command_parts.extend([f"--{name} {value}" for name, value in param_dict.items()])

        command = " ".join(command_parts)
        print(f"Running: {command}")
        os.system(command)


if args.loss_type == "both":
    loss_types = parameter_space.keys()
else:
    loss_types = [args.loss_type]

for loss_type in loss_types:
    print(f"\nRunning experiments for {loss_type}")
    run_experiments(loss_type)
