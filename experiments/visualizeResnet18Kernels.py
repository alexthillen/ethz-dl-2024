import torch
import torchvision.models as models
import matplotlib.pyplot as plt

# Load a pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)


# Function to normalize the kernels
def normalize_kernels(kernels):
    return (kernels - kernels.min()) / (kernels.max() - kernels.min())


# Function to visualize kernels
def visualize_kernels(kernels, num_kernels_to_show=64, layer_name="Layer"):
    num_kernels = min(kernels.shape[0], num_kernels_to_show)
    rows = int(num_kernels ** 0.5)
    cols = (num_kernels + rows - 1) // rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i in range(num_kernels):
        kernel = kernels[i].permute(1, 2, 0).numpy()  # Convert to (H, W, C)
        axes[i].imshow(kernel)
        axes[i].axis('off')
    for j in range(num_kernels, len(axes)):  # Hide any extra axes
        axes[j].axis('off')

    plt.suptitle(f"Kernels in {layer_name}", fontsize=16)
    plt.show()






def visualize_kernels_single_output(kernels, output_channel=0):
    """Visualizes the kernels for a single output channel."""
    kernels_for_output = kernels[output_channel]  # Shape: [64, 3, 3]

    # Normalize kernels for visualization
    kernels_normalized = (kernels_for_output - kernels_for_output.min()) / (kernels_for_output.max() - kernels_for_output.min())

    # Plot kernels as a grid
    num_kernels = kernels_normalized.shape[0]
    rows = int(num_kernels ** 0.5)
    cols = (num_kernels + rows - 1) // rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for i in range(num_kernels):
        kernel = kernels_normalized[i].numpy()  # Convert to NumPy array
        axes[i].imshow(kernel, cmap='viridis')  # Plot as heatmap
        axes[i].axis('off')
    for j in range(num_kernels, len(axes)):  # Hide extra subplots
        axes[j].axis('off')

    plt.suptitle(f"Kernels for Output Channel {output_channel}", fontsize=16)
    plt.show()





# Iterate through all layers to find convolutional layers
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):  # Check if it's a convolutional layer
        kernels = module.weight.data

        visualize_kernels_single_output(kernels, output_channel=0)

        # kernels_normalized = normalize_kernels(kernels)
        #
        # print(kernels_normalized.shape)
        # # If the kernels are single-channel, add a channel dimension for visualization
        # if kernels_normalized.shape[1] == 1:  # Single channel (grayscale-like)
        #     kernels_normalized = kernels_normalized.repeat(1, 3, 1, 1)
        #
        # # Visualize kernels (show up to 64 for clarity)
        # visualize_kernels(kernels_normalized, num_kernels_to_show=64, layer_name=name)
