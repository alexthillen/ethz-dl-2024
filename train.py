import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader, Subset, ConcatDataset
import argparse
import torch.nn.functional as F
from activation_preserving_loss import ActivationPreservingLoss
from saliency_preserving_loss import SaliencyPreservingLoss

# Parse command line arguments
parser = argparse.ArgumentParser(description="Training parameters")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--num_epochs", type=int, default=8, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--buffer_size", type=int, default=3000, help="Replay buffer size")
parser.add_argument("--loss_type", type=str, default="activation_preserving", help="Loss type")
parser.add_argument(
    "--saliency_lambda", type=float, default=0, help="Weight for activation preservation loss"
)
parser.add_argument(
    "--num_samples_per_class",
    type=int,
    default=5,
    help="Number of samples per class for activation preservation",
)
parser.add_argument(
    "--noise_std",
    type=float,
    default=0.1,
    help="Standard deviation of noise for saliency preservation",
)
parser.add_argument(
    "--saliency_threshold", type=float, default=0.5, help="Threshold for saliency preservation"
)
args = parser.parse_args()
# Hyperparameters and Configuration
HYPERPARAMETERS = {
    # Model training
    "learning_rate": args.learning_rate,
    "num_epochs": args.num_epochs,
    "batch_size": args.batch_size,
    "buffer_size": args.buffer_size,
    "saliency_lambda": args.saliency_lambda,
    "num_samples_per_class": args.num_samples_per_class,
    "loss_type": args.loss_type,
    # Dataset and tasks
    "num_classes": 10,
    "tasks": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],  # Task 1  # Task 2  # Task 3
    # Activation preservation
    "preserved_layers": [
        "features.6",
        "features.7",
        "features.8",
        "classifier.0",
        "classifier.1",
        "classifier.2",
    ],
    # Reproducibility
    "seed": 42,
    # Data normalization
    "cifar_mean": (0.4914, 0.4822, 0.4465),
    "cifar_std": (0.2470, 0.2435, 0.2616),
    "noise_std": args.noise_std,
    "saliency_threshold": args.saliency_threshold,
}


# Set random seeds
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seeds(HYPERPARAMETERS["seed"])

# Dataset preparation
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(HYPERPARAMETERS["cifar_mean"], HYPERPARAMETERS["cifar_std"]),
    ]
)

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)


def filter_dataset_by_labels(dataset, labels):
    """Return indices of dataset samples having the specified labels."""
    indices = [i for i, (_, label) in enumerate(dataset) if label in labels]
    return Subset(dataset, indices)


# Create subsets for each task
train_subsets = [filter_dataset_by_labels(train_dataset, task) for task in HYPERPARAMETERS["tasks"]]
test_subsets = [filter_dataset_by_labels(test_dataset, task) for task in HYPERPARAMETERS["tasks"]]


class SimpleVGG(nn.Module):
    def __init__(self, num_classes=HYPERPARAMETERS["num_classes"]):
        super(SimpleVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(8 * 8 * 128, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)


class ReplayBuffer:
    def __init__(self, buffer_size=HYPERPARAMETERS["buffer_size"]):
        self.buffer_size = buffer_size
        self.buffer_data = []
        self.buffer_labels = []

    def add_samples(self, data, labels):
        if len(self.buffer_data) >= self.buffer_size:
            to_remove = len(self.buffer_data) + len(data) - self.buffer_size
            indices_to_remove = random.sample(range(len(self.buffer_data)), to_remove)
            for idx in sorted(indices_to_remove, reverse=True):
                del self.buffer_data[idx]
                del self.buffer_labels[idx]

        self.buffer_data.extend(data)
        self.buffer_labels.extend(labels)

    def get_samples(self, batch_size):
        if len(self.buffer_data) == 0:
            return None, None
        indices = random.sample(
            range(len(self.buffer_data)), min(batch_size, len(self.buffer_data))
        )
        replay_data = [self.buffer_data[i] for i in indices]
        replay_labels = [self.buffer_labels[i] for i in indices]
        return torch.stack(replay_data), torch.tensor(replay_labels)

    def __len__(self):
        return len(self.buffer_data)


def store_in_replay_buffer(dataset_subset, replay_buffer, store_size=None):
    if store_size is None:
        store_size = HYPERPARAMETERS["buffer_size"] // 3
    indices = np.random.choice(len(dataset_subset), size=store_size, replace=False)
    data = []
    labels = []
    for idx in indices:
        x, y = dataset_subset[idx]
        data.append(x)
        labels.append(y)
    replay_buffer.add_samples(data, labels)


def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total, correct, total


def train_sequential(model, train_subsets, test_subsets, tasks):
    print(f"Training with hyperparameters:")
    print(f"saliency_lambda: {HYPERPARAMETERS['saliency_lambda']}")
    print(f"samples_per_class: {HYPERPARAMETERS['num_samples_per_class']}")
    print(f"buffer_size: {HYPERPARAMETERS['buffer_size']}")

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    replay_buffer = ReplayBuffer()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMETERS["learning_rate"])

    train_acc_history = []
    test_acc_history = []
    test_acc_per_model = []

    for task_idx, (train_subset, test_subset, labels) in enumerate(
        zip(train_subsets, test_subsets, tasks)
    ):
        print(f"=== Training on Task {task_idx+1} with labels {labels} ===")

        train_loader = DataLoader(
            train_subset, batch_size=HYPERPARAMETERS["batch_size"], shuffle=True
        )
        test_loader = DataLoader(
            test_subset, batch_size=HYPERPARAMETERS["batch_size"], shuffle=False
        )

        saliency_loss = None
        if task_idx > 0 and HYPERPARAMETERS["saliency_lambda"] > 0:
            prev_datasets = ConcatDataset([train_subsets[prev_idx] for prev_idx in range(task_idx)])
            if HYPERPARAMETERS["loss_type"] == "activation_preserving":
                saliency_loss = ActivationPreservingLoss(
                    model,
                    dataset=prev_datasets,
                    num_samples_per_class=HYPERPARAMETERS["num_samples_per_class"],
                    layers=HYPERPARAMETERS["preserved_layers"],
                )
            elif HYPERPARAMETERS["loss_type"] == "saliency_preserving":
                saliency_loss = SaliencyPreservingLoss(
                    model,
                    dataset=prev_datasets,
                    num_samples_per_class=HYPERPARAMETERS["num_samples_per_class"],
                    saliency_threshold=HYPERPARAMETERS["saliency_threshold"],
                    noise_std=HYPERPARAMETERS["noise_std"],
                )

        train = True
        model_path = f"task_{'-'.join(map(str, range(1, task_idx+2)))}_nr_epochs_{HYPERPARAMETERS['num_epochs']}_buffersize_{HYPERPARAMETERS['buffer_size']}"
        if task_idx > 0:
            model_path += f"_saliency_lambda_{HYPERPARAMETERS['saliency_lambda']}_samples_per_class_{HYPERPARAMETERS['num_samples_per_class']}_noise_std_{HYPERPARAMETERS['noise_std']}_saliency_threshold_{HYPERPARAMETERS['saliency_threshold']}"
        model_path += ".pth"

        try:
            print(f"Trying to load model from {model_path}")
            model.load_state_dict(torch.load(model_path))
            train = False
            print("Model loaded from previous training instead of training from scratch")
        except FileNotFoundError:
            print("No model found, training from scratch")

        if train:
            for epoch in range(HYPERPARAMETERS["num_epochs"]):
                start_time = time.time()
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                total_sal_loss = 0.0

                for images, lbls in train_loader:
                    images, lbls = images.to(device), lbls.to(device)

                    replay_images, replay_labels = replay_buffer.get_samples(
                        HYPERPARAMETERS["batch_size"] // 2
                    )
                    if replay_images is not None:
                        replay_images = replay_images.to(device)
                        replay_labels = replay_labels.to(device)
                        images = torch.cat((images, replay_images), dim=0)
                        lbls = torch.cat((lbls, replay_labels), dim=0)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, lbls)

                    if saliency_loss is not None:
                        sal_loss = HYPERPARAMETERS["saliency_lambda"] * saliency_loss(model)
                        total_sal_loss += sal_loss.item()
                        loss += sal_loss

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(lbls).sum().item()
                    total += lbls.size(0)

                epoch_time = time.time() - start_time
                epoch_loss = running_loss / len(train_loader)
                epoch_acc = 100.0 * correct / total
                print(
                    f"Epoch [{epoch+1}/{HYPERPARAMETERS['num_epochs']}], "
                    f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, Time: {epoch_time:.2f}s"
                )
                print(f"Total sal_loss: {total_sal_loss / len(train_loader):.4f}")

            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

        task_train_acc, train_correct, train_total = evaluate_accuracy(model, train_loader, device)
        task_test_acc, test_correct, test_total = evaluate_accuracy(model, test_loader, device)

        train_acc_history.append(task_train_acc)
        test_acc_history.append(task_test_acc)

        print(
            f"Task {task_idx+1} Train Accuracy: {task_train_acc:.2f}% ({train_correct}/{train_total})"
        )
        print(
            f"Task {task_idx+1} Test Accuracy:  {task_test_acc:.2f}% ({test_correct}/{test_total})\n"
        )
        test_acc_model = []

        for task_idx_test, test_subset in enumerate(test_subsets):
            test_loader = DataLoader(
                test_subset, batch_size=HYPERPARAMETERS["batch_size"], shuffle=False
            )
            test_acc, test_correct, test_total = evaluate_accuracy(model, test_loader, device)
            nr_incorrect = test_total - test_correct
            print(
                f"Task {task_idx+1} -> Test Accuracy on {task_idx_test+1} : {test_acc:.2f}% "
                f"({test_correct}/{test_total}), Incorrect: {nr_incorrect}"
            )
            test_acc_model.append(test_acc)

        if task_idx == 2:
            print("Average accuracy across tasks: ", sum(test_acc_model) / len(test_acc_model))
        test_acc_per_model.append(test_acc_model)

        store_in_replay_buffer(train_subset, replay_buffer)

    # Plot final accuracies
    plot_final_accuracies(test_acc_per_model)

    return model, train_acc_history, test_acc_history


def plot_final_accuracies(test_acc_per_model, title="Test Accuracies"):
    plt.figure(figsize=(10, 6))

    # Use a larger font size for better legibility
    plt.rcParams.update({"font.size": 14})

    for task_idx in range(len(test_acc_per_model[0])):
        task_accuracies = [acc[task_idx] for acc in test_acc_per_model]
        plt.plot(
            range(1, len(test_acc_per_model) + 1),
            task_accuracies,
            marker="o",
            label=f"Task {task_idx + 1}",
        )

    # Calculate the average accuracy at the last time step
    avg_accuracy_last_step = np.mean([acc[-1] for acc in zip(*test_acc_per_model)])
    plt.axhline(
        y=avg_accuracy_last_step,
        color="r",
        linestyle="--",
        label=f"Avg at Last Step: {avg_accuracy_last_step:.2f}%",
    )

    plt.xlabel("Training Task")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True)

    plt.xticks(ticks=range(1, len(test_acc_per_model) + 1))

    # Save the plot
    plot_path = f"final_accuracies_nr_epochs_{HYPERPARAMETERS['num_epochs']}_buffersize_{HYPERPARAMETERS['buffer_size']}_saliency_lambda_{HYPERPARAMETERS['saliency_lambda']}_samples_per_class_{HYPERPARAMETERS['num_samples_per_class']}_{HYPERPARAMETERS['loss_type']}_noise_std_{HYPERPARAMETERS['noise_std']}_saliency_threshold_{HYPERPARAMETERS['saliency_threshold']}.png"
    plt.savefig(plot_path, format="png", dpi=300)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    # Initialize model
    model = SimpleVGG()

    # Train the model sequentially on tasks
    model, train_acc, test_acc = train_sequential(
        model, train_subsets, test_subsets, HYPERPARAMETERS["tasks"]
    )

    # Print final accuracies
    print("\nFinal Results:")
    print("Train accuracies across tasks:", train_acc)
    print("Test accuracies across tasks:", test_acc)
