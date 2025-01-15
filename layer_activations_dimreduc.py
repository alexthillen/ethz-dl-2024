import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
import torchvision.models as models


# For dimensionality reduction
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.decomposition import PCA
from collections import Counter


import os

###############################################################################
# 1. DATA PREPARATION
###############################################################################
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # standard CIFAR-10 normalization
])

# Load CIFAR-10
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
test_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Task label splits
tasks = [
    [0, 1, 2],   # Task 1
    [3, 4, 5],   # Task 2
    [6, 7, 8]    # Task 3
]

num_classes=10

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100 normalization
# ])
#
# # Load CIFAR-100
# train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
# test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
#
# # Task label splits (update based on CIFAR-100 class indices)
# tasks = [
#     list(range(0, 40)),  # Task 1: Classes 0-40
#     list(range(41, 60)), # Task 2:
#     list(range(61, 80))  # Task 3:
# ]



def filter_dataset_by_labels(dataset, labels):
    """Return a Subset of dataset for samples having the specified labels."""
    indices = [i for i, (_, label) in enumerate(dataset) if label in labels]
    return Subset(dataset, indices)

# Create subsets for each task
train_subsets = [filter_dataset_by_labels(train_dataset, task) for task in tasks]
test_subsets  = [filter_dataset_by_labels(test_dataset,  task) for task in tasks]


###############################################################################
# 2. MODEL DEFINITION
###############################################################################
class SimpleVGG(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(8 * 8 * 128, 256),  # Linear layer #1
            nn.ReLU(True),
            nn.Linear(256, num_classes)   # Linear layer #2
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



# Replace SimpleVGG with ResNet-18
class ResNet18Model(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18Model, self).__init__()
        # Load pre-defined ResNet-18
        self.resnet = models.resnet18(pretrained=False)  # Use pretrained=True for ImageNet weights

        # Modify the final fully connected layer to match the number of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


###############################################################################
# 3. REPLAY BUFFER
###############################################################################
class ReplayBuffer:
    def __init__(self, buffer_size=200):
        self.buffer_size = buffer_size
        self.buffer_data = []
        self.buffer_labels = []

    def add_samples(self, data, labels):
        # If buffer is full, remove random samples to make space
        if len(self.buffer_data) + len(data) > self.buffer_size:
            to_remove = (len(self.buffer_data) + len(data)) - self.buffer_size
            import random
            indices_to_remove = random.sample(range(len(self.buffer_data)), to_remove)
            for idx in sorted(indices_to_remove, reverse=True):
                del self.buffer_data[idx]
                del self.buffer_labels[idx]

        self.buffer_data.extend(data)
        self.buffer_labels.extend(labels)

    def get_samples(self, batch_size):
        """Return a random batch from the replay buffer."""
        if len(self.buffer_data) == 0:
            return None, None
        import random
        indices = random.sample(range(len(self.buffer_data)), min(batch_size, len(self.buffer_data)))
        replay_data   = [self.buffer_data[i]   for i in indices]
        replay_labels = [self.buffer_labels[i] for i in indices]
        return torch.stack(replay_data), torch.tensor(replay_labels)

    def __len__(self):
        return len(self.buffer_data)


###############################################################################
# 4. TRAINING & EVALUATION FUNCTIONS
###############################################################################
def store_in_replay_buffer(model, dataset_subset, replay_buffer, device, store_size=250):
    """
    Randomly select store_size samples from the dataset_subset and add to the replay buffer.
    """
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
    return 100.0 * correct / total

def train_sequential(model, train_subsets, test_subsets, tasks,
                     buffer_size=200, num_epochs=5, batch_size=64, lr=0.01):
    """
    Train model sequentially on each task with a replay buffer, combining the training batch and replay buffer in one batch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    replay_buffer = ReplayBuffer(buffer_size=buffer_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # For recording
    train_acc_history = []
    test_acc_history  = []
    # We'll store a copy of the model after each task
    saved_models = []

    for task_idx, (train_subset, test_subset, labels) in enumerate(zip(train_subsets, test_subsets, tasks)):
        print(f"=== Training on Task {task_idx+1} with labels {labels} ===")

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(test_subset,  batch_size=batch_size, shuffle=False)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, lbls in train_loader:
                images, lbls = images.to(device), lbls.to(device)

                # Get replay samples
                replay_images, replay_labels = replay_buffer.get_samples(batch_size // 2)
                if replay_images is not None:
                    #print("Using replay images")
                    replay_images = replay_images.to(device)
                    replay_labels = replay_labels.to(device)

                    # Combine current task samples and replay samples
                    images = torch.cat((images, replay_images), dim=0)
                    lbls = torch.cat((lbls, replay_labels), dim=0)


                label_distribution = Counter(lbls.cpu().tolist())
                #print(f"Label distribution in batch (Epoch {epoch + 1}): {label_distribution}")

                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, lbls)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(lbls).sum().item()
                total += lbls.size(0)

            epoch_loss = running_loss / len(train_loader)
            epoch_acc  = 100. * correct / total
            print(f"  Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

        # Evaluate on the current task
        task_train_acc = evaluate_accuracy(model, train_loader, device)
        task_test_acc  = evaluate_accuracy(model, test_loader,  device)
        train_acc_history.append(task_train_acc)
        test_acc_history.append(task_test_acc)

        print(f"Task {task_idx+1} Train Accuracy: {task_train_acc:.2f}%")
        print(f"Task {task_idx+1} Test Accuracy:  {task_test_acc:.2f}%\n")

        # Add random samples from the current task to the replay buffer
        store_in_replay_buffer(model, train_subset, replay_buffer, device, store_size=buffer_size // 3)

        # Save a copy of the model (so we can later visualize its activations)
        saved_models.append(copy.deepcopy(model))

    return saved_models, train_acc_history, test_acc_history


###############################################################################
# 5. EXTRACTION OF LAYER ACTIVATIONS
###############################################################################
def extract_activations(model, dataset, device):
    """
    Given a trained model and a dataset (e.g. Task 1 test samples),
    return two arrays of activations:
      - activations of the first linear layer (size = 256-dim)
      - activations of the second linear layer (size = num_classes)
    Also return the labels for coloring.
    """
    model.eval()
    activations_layer1 = []
    activations_layer2 = []
    labels = []

    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        for images, lbls in loader:
            images = images.to(device)
            # Forward up to the features
            feats = model.features(images)
            feats = feats.view(feats.size(0), -1)

            # The classifier is [Linear -> ReLU -> Linear]
            # We'll manually compute the intermediate and final outputs
            linear1_out = model.classifier[0](feats)  # first Linear layer
            relu_out    = model.classifier[1](linear1_out)
            linear2_out = model.classifier[2](relu_out)  # second Linear layer

            # Store them
            activations_layer1.append(linear1_out.cpu().numpy())
            activations_layer2.append(linear2_out.cpu().numpy())
            labels.extend(lbls.numpy())

    # Concatenate all batches into single arrays
    activations_layer1 = np.concatenate(activations_layer1, axis=0)
    activations_layer2 = np.concatenate(activations_layer2, axis=0)
    labels = np.array(labels)

    return activations_layer1, activations_layer2, labels


###############################################################################
# 6. DIMENSIONALITY REDUCTION & PLOTTING
###############################################################################
def plot_and_save(embedding, labels, method_name, title, save_path):
    """
    Helper function to scatter-plot the 2D 'embedding', color by 'labels',
    put 'title' on top, and save to 'save_path'.
    """
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(embedding[:, 0],
                          embedding[:, 1],
                          c=labels,
                          cmap="tab10",
                          alpha=0.7,
                          edgecolor='none')

    handles, legend_labels = scatter.legend_elements(prop="colors", alpha=0.7)
    plt.legend(handles, legend_labels, title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')

    #plt.colorbar(scatter, ticks=np.unique(labels))
    plt.title(title + f" ({method_name})")
    #plt.xlabel('Dim 1')
    #plt.ylabel('Dim 2')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()  # Close the figure so it doesn't show interactively


def apply_and_plot_all_dr_methods(activations, labels, task_id, layer_id, output_dir):
    """
    For the given 'activations' (N x D) and 'labels' (N, ) array,
    run 4 different dimensionality reduction methods:
      - PCA
      - t-SNE
      - Isomap
      - MDS
    Then create and save the plots.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define dimensionality reduction methods
    dr_methods = {
        "PCA":    PCA(n_components=2, random_state=42),
        "t-SNE":  TSNE(n_components=2, random_state=42),
        #"Isomap": Isomap(n_components=2),  # less insightful
        #"MDS":    MDS(n_components=2, random_state=42)
    }

    for method_name, dr_obj in dr_methods.items():
        # Apply DR
        embedding_2d = dr_obj.fit_transform(activations)

        # Title and path
        title     = f"Model of Task {task_id} - Layer {layer_id} activations"
        filename  = f"Task{task_id}_Layer{layer_id}_{method_name}.png"
        save_path = os.path.join(output_dir, filename)

        # Plot & save
        plot_and_save(embedding_2d, labels, method_name, title, save_path)
        print(f"Saved: {save_path}")


###############################################################################
# 7. MAIN SCRIPT
###############################################################################
def main():
    # Fix seeds for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Initialize model
    model = SimpleVGG(num_classes=10)
    #model = ResNet18Model(num_classes=num_classes)

    ###########################################################################
    # A) Train the model sequentially on tasks
    ###########################################################################
    saved_models, train_acc, test_acc = train_sequential(
        model,
        train_subsets,
        test_subsets,
        tasks,
        buffer_size=3000,
        num_epochs=8,
        batch_size=64,
        lr=0.001
    )

    # Print final accuracies
    print("Train accuracies across tasks:", train_acc)
    print("Test accuracies across tasks: ", test_acc)

    ###########################################################################
    # B) Visualize how activations for Task 1 test samples change
    #    after each task. We'll use 4 DR methods: PCA, t-SNE, Isomap, MDS.
    ###########################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # test_subsets[0] = test set containing labels [0,1,2] (Task 1)
    test_task1_subset = test_subsets[0]
    test_task2_subset = test_subsets[1]
    test_task3_subset = test_subsets[2]

    from torch.utils.data import ConcatDataset
    combined_test_subset = ConcatDataset([test_subsets[0], test_subsets[1], test_subsets[2]])


    test_loader1 = DataLoader(test_subsets[0], batch_size=64, shuffle=False)
    test_loader2 = DataLoader(test_subsets[1], batch_size=64, shuffle=False)
    test_loader3 = DataLoader(test_subsets[2], batch_size=64, shuffle=False)


    task_test1_model2_acc = evaluate_accuracy(saved_models[1], test_loader1, device)
    task_test1_model3_acc = evaluate_accuracy(saved_models[2], test_loader1, device)

    print(f"Task 1 Test Accuracy on model 2:  {task_test1_model2_acc:.2f}%\n")
    print(f"Task 1 Test Accuracy on model 3:  {task_test1_model3_acc:.2f}%\n")


    # Directory to save output plots
    output_dir = "activations_plots"

    for i, trained_model in enumerate(saved_models):
        # Extract activations for test samples of Task 1
        act_layer1, act_layer2, labels = extract_activations(trained_model, test_task1_subset, device)

        # DR plots for layer 1
        apply_and_plot_all_dr_methods(
            activations=act_layer1,
            labels=labels,
            task_id=(i+1),
            layer_id="Fully connected",
            output_dir=output_dir
        )

        # DR plots for layer 2
        apply_and_plot_all_dr_methods(
            activations=act_layer2,
            labels=labels,
            task_id=(i+1),
            layer_id="Output",
            output_dir=output_dir
        )

    # plot model 3 activation clusters on complete test set
    act_layer1, act_layer2, labels = extract_activations(saved_models[2], combined_test_subset, device)


    apply_and_plot_all_dr_methods(
        activations=act_layer1,
        labels=labels,
        task_id=(1),
        layer_id=1,
        output_dir="activations_plots_full"
    )

    # DR plots for layer 2
    apply_and_plot_all_dr_methods(
        activations=act_layer2,
        labels=labels,
        task_id=(1),
        layer_id=2,
        output_dir="activations_plots_full"
    )

if __name__ == "__main__":
    main()
