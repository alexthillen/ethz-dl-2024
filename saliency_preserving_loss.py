from gradcam import GradCAM
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F


def unnormalize_and_convert_to_numpy(
    tensor, mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
):
    # Convert from normalized tensor [C,H,W] to numpy array [H,W,C] for plotting
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    img = img * np.array(std) + np.array(mean)
    img = np.clip(img, 0, 1)
    return img


class SaliencyPreservingLoss:
    def __init__(
        self, model, dataset, num_samples_per_class=3, saliency_threshold=0.5, noise_std=0.1
    ):
        """
        Initialize the loss with stored examples and their noisy counterparts.

        Args:
            model: The model to compute saliency maps
            dataset: Dataset containing samples from previous tasks
            num_samples_per_class: Number of samples to store per class
            saliency_threshold: Threshold for considering regions as salient (0-1)
            noise_std: Standard deviation of the noise to add
        """

        print(
            f"Initializing SaliencyPreservingLoss with #samples/class = {num_samples_per_class}, saliency_threshold = {saliency_threshold}, noise_std = {noise_std}"
        )
        self.device = next(model.parameters()).device
        self.stored_samples = []
        self.noisy_samples = []

        # Create GradCAM instance
        self.gradcam = GradCAM(model, target_layer_name="features.7")

        # Group samples by class
        samples_by_class = {label: [] for label in set(y for _, y in dataset)}
        model.eval()
        with torch.no_grad():
            for i in range(len(dataset)):
                x, y_true = dataset[i]
                x = x.unsqueeze(0).to(self.device)
                y_pred = model(x).argmax(dim=1).item()

                if y_pred == y_true:
                    if y_true not in samples_by_class:
                        samples_by_class[y_true] = []
                    samples_by_class[y_true].append(i)

                # Early stopping if we have enough samples for each class
                if all(
                    len(indices) >= num_samples_per_class for indices in samples_by_class.values()
                ):
                    break

        # Select equal number of samples from each class
        selected_indices = []
        for class_indices in samples_by_class.values():
            selected_indices.extend(
                random.sample(class_indices, min(num_samples_per_class, len(class_indices)))
            )

        # For each selected sample, compute saliency and create noisy version
        for idx in selected_indices:
            x, y = dataset[idx]
            x = x.to(self.device)

            # Store original sample
            self.stored_samples.append(x.clone())

            # Compute saliency map
            x_input = x.unsqueeze(0)
            saliency_map = self.gradcam.generate_cam(x_input, target_class=[y])[0]

            # Create binary mask from saliency map
            mask = torch.tensor(saliency_map > saliency_threshold).float().to(self.device)
            # Upsample to match input size
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0), size=(32, 32), mode="bilinear", align_corners=False
            )[
                0
            ]  # shape: [1, 32, 32]
            mask = mask.repeat(3, 1, 1)  # shape: [3, 32, 32]

            # Create noisy version: original * mask + noise * (1-mask)
            noise = torch.randn_like(x) * noise_std
            noisy_x = x * mask + (x + noise) * (1 - mask)
            self.noisy_samples.append(noisy_x)

        for i in range(len(self.noisy_samples)):
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))

            # Original image
            original_image = unnormalize_and_convert_to_numpy(self.stored_samples[i])
            axes[0].imshow(original_image)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            # Noisy version
            noisy_image = unnormalize_and_convert_to_numpy(self.noisy_samples[i])
            axes[1].imshow(noisy_image)
            axes[1].set_title("Noisy Version")
            axes[1].axis("off")

            # Save the figure
            plt.savefig(f"deliverables/saliency_loss/noisy_samples_comparison{i}.png")
            plt.close()

        print(
            f"Stored {len(self.stored_samples)} samples total ({len(samples_by_class)} classes, "
            f"~{len(self.stored_samples)/len(samples_by_class):.1f} samples per class)"
        )

        # Convert to tensors
        self.stored_samples = torch.stack(self.stored_samples)
        self.noisy_samples = torch.stack(self.noisy_samples)

    def __call__(self, model, layer_name="features.7"):
        """
        Compute the loss between activations of original and noisy samples.

        Args:
            model: Current model state
            layer_name: Name of the layer to compare activations

        Returns:
            loss: Mean squared error between activations
        """
        # Register temporary forward hook to get activations
        activations = {}

        def hook(module, input, output):
            activations["value"] = output

        # Get the target layer
        target_layer = dict([*model.named_modules()])[layer_name]
        handle = target_layer.register_forward_hook(hook)

        # Get activations for original samples
        _ = model(self.stored_samples)
        original_activations = activations["value"]

        # Get activations for noisy samples
        _ = model(self.noisy_samples)
        noisy_activations = activations["value"]

        # Remove the hook
        handle.remove()

        # Compute MSE loss between activations
        return F.mse_loss(original_activations, noisy_activations) * 3.0 / len(self.stored_samples)
