import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import copy

# Define which classes belong to each task
tasks_classes = [
    [0, 1, 2],  # Task 1
    [3, 4, 5],  # Task 2
    [6, 7, 8],  # Task 3
]

EWC_LAMBDA = 10**6
print("version", EWC_LAMBDA)

def get_task_dataset(task_classes, train=True):
    """
    Returns a subset of the CIFAR10 dataset that contains only the specified classes.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),  # mean
                             (0.2470, 0.2435, 0.2616))  # std
    ])

    cifar10 = torchvision.datasets.CIFAR10(root='./data', train=train,
                                           download=True, transform=transform)

    # Filter to keep only the specified classes
    class_indices = [i for i, (_, label) in enumerate(cifar10) if label in task_classes]

    # Subset
    subset = torch.utils.data.Subset(cifar10, class_indices)
    return subset


def get_dataloader(task_classes, batch_size=64, train=True):
    dataset = get_task_dataset(task_classes, train=train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


class SimpleVGG(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleVGG, self).__init__()

        # A simplified VGG-like feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(8 * 8 * 128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)  # this layer will be replaced/extended for new tasks
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class EWC:
    def __init__(self, model, dataloader, device='cpu'):
        self.model = model
        self.dataloader = dataloader
        self.device = device

        # Store old parameters
        self.params = {}
        for name, param in self.model.named_parameters():
            self.params[name] = param.detach().clone().to(self.device)

        # Compute Fisher
        self.fisher = self.compute_fisher()

    def compute_fisher(self):
        # Set model in evaluation mode
        self.model.eval()

        # We accumulate the diagonal Fisher information in a dict
        fisher_dict = {}
        for name, param in self.model.named_parameters():
            fisher_dict[name] = torch.zeros_like(param).to(self.device)

        # In practice, use a subset or the entire dataset
        # Here we do a single pass for simplicity
        criterion = nn.CrossEntropyLoss()

        for inputs, targets in self.dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()

            # Accumulate squared gradient for each parameter
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher_dict[name] += param.grad.data.pow(2)

        # Normalize by the number of samples in the dataset
        num_samples = len(self.dataloader.dataset)
        for name in fisher_dict:
            fisher_dict[name] = fisher_dict[name] / num_samples

        return fisher_dict

    def ewc_loss(self, model):

        loss = 0.0
        for name, param in model.named_parameters():
            # (theta_i - theta_i^old)^2
            diff = param - self.params[name]
            # F_i * diff^2
            loss += torch.sum(self.fisher[name] * diff.pow(2)) / 2
        return loss


def train_model(model, dataloader, optimizer, device='cpu', ewc_object=None, ewc_lambda=0.4, epochs=5):
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            ce_loss = criterion(outputs, targets)

            # Add EWC loss if ewc_object is provided
            if ewc_object is not None:
                penalty = ewc_object.ewc_loss(model)
                loss = ce_loss + ewc_lambda * penalty
            else:
                loss = ce_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[Epoch {epoch + 1}/{epochs}] Loss: {running_loss / len(dataloader):.4f}")


def evaluate_model(model, dataloader, device='cpu'):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    return correct / total


# def extend_classifier(model, new_num_classes):
#     """
#     Extends the final layer of the model to accommodate new_num_classes outputs.
#     Initializes the new weights randomly, and keeps old weights for the old classes.
#     """
#     old_classifier = model.classifier[-1]  # last Linear layer
#     old_out_features = old_classifier.out_features
#
#     # If new_num_classes <= old_out_features, no extension needed
#     if new_num_classes <= old_out_features:
#         return model
#
#     in_features = old_classifier.in_features
#
#     # Create new layer
#     new_classifier = nn.Linear(in_features, new_num_classes)
#     # Copy old weights
#     with torch.no_grad():
#         new_classifier.weight[:old_out_features] = old_classifier.weight
#         new_classifier.bias[:old_out_features] = old_classifier.bias
#
#     # Replace the last layer
#     model.classifier[-1] = new_classifier
#     return model


def run_sequential_learning(tasks_classes, device='cpu'):
    # Initialize a model with output size = number of classes in Task 1
    model = SimpleVGG(num_classes=9)
    model.to(device)

    ewc_objects = []  # keep EWC info for each task
    all_accuracies = []
    all_forgetting = []  # store “forgetting measure”

    # Keep track of best accuracies per task to measure forgetting
    best_accuracies_so_far = [0.0 for _ in tasks_classes]

    for task_idx, classes in enumerate(tasks_classes):
        print(f"\n=== Training on Task {task_idx + 1}, classes {classes} ===")

        # Extend classifier if needed
        #current_total_classes = sum(len(tc) for tc in tasks_classes[:task_idx + 1])
        #model = extend_classifier(model, current_total_classes)

        # Prepare dataloader for the current task
        train_loader = get_dataloader(classes, batch_size=64, train=True)

        # Create optimizer
        #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # EWC penalty uses info from ALL previous tasks
        # Combine penalty from each old task’s EWC
        def combined_ewc_loss(model):
            if len(ewc_objects) == 0:
                return 0.0
            ewc_sum = 0.0
            for ewc_obj in ewc_objects:
                ewc_sum += ewc_obj.ewc_loss(model)
            return ewc_sum

        # Modified train function to handle multiple EWC objects
        def train_with_ewc(model, dataloader, optimizer, device='cpu', ewc_lambda=0.4, epochs=5):
            criterion = nn.CrossEntropyLoss()
            model.to(device)

            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                for inputs, targets in dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    ce_loss = criterion(outputs, targets)

                    # Combined penalty
                    penalty = combined_ewc_loss(model)

                    print("CE loss", ce_loss.item(), "EWC penalty", ewc_lambda * penalty)

                    loss = ce_loss + ewc_lambda * penalty
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                print(f"[Epoch {epoch + 1}/{epochs}] Loss: {running_loss / len(dataloader):.4f}")

        # Train on current task
        if task_idx == 0:
            # No EWC for first task
            train_model(model, train_loader, optimizer, device=device, ewc_object=None, epochs=5)
        else:
            # Use combined EWC from previous tasks
            train_with_ewc(model, train_loader, optimizer, device=device, ewc_lambda=EWC_LAMBDA, epochs=5)

        # After training on this task, create EWC object for this task
        # so that future tasks will regularize to these parameters
        ewc_object = EWC(copy.deepcopy(model), get_dataloader(classes, train=True), device=device)
        ewc_objects.append(ewc_object)

        # Evaluate on all tasks seen so far
        task_accuracies = []
        for eval_idx in range(task_idx + 1):
            eval_classes = tasks_classes[eval_idx]
            eval_loader = get_dataloader(eval_classes, batch_size=64, train=False)
            acc = evaluate_model(model, eval_loader, device=device)
            task_accuracies.append(acc)

        all_accuracies.append(task_accuracies)

        # Compute forgetting measure for each previous task
        # We track how much the accuracy dropped relative to the best performance so far
        for i in range(len(task_accuracies)):
            best_accuracies_so_far[i] = max(best_accuracies_so_far[i], task_accuracies[i])

        forgetting = []
        for i in range(len(task_accuracies)):
            drop = best_accuracies_so_far[i] - task_accuracies[i]
            forgetting.append(drop)

        all_forgetting.append(forgetting)
        print(f"Task accuracies so far: {task_accuracies}")
        print(f"Forgetting so far: {forgetting}")

    return model, ewc_objects, all_accuracies, all_forgetting


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    final_model, ewc_objs, accuracies, forgetting = run_sequential_learning(tasks_classes, device=device)

    print("\nFinal Accuracies per Task at the End:")
    for i, accs in enumerate(accuracies):
        print(f"After task {i + 1}, accuracies: {accs}")

    print("\nForgetting measures:")
    for i, f in enumerate(forgetting):
        print(f"After task {i + 1}, forgetting: {f}")
