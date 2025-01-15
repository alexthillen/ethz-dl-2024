import torch
import torch.nn.functional as F


class GradCAM:
    """
    Minimal Grad-CAM implementation for a SimpleVGG model.
    target_layer_name = 'features.7' by default (the last conv layer).
    """

    def __init__(self, model, target_layer_name="features.7"):
        self.model = model
        self.target_layer_name = target_layer_name

        # Will be set by hooks
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _save_gradient(self, grad):
        self.gradients = grad.clone()

    def _register_hooks(self):
        """
        This finds the target layer by name and registers the forward/backward hooks
        to capture the activations and gradients.
        """
        # Build a dictionary of all modules by name
        modules_dict = dict([*self.model.named_modules()])

        # Confirm your layer name is in the dictionary (e.g., 'features.7')
        if self.target_layer_name not in modules_dict:
            raise ValueError(
                f"Layer {self.target_layer_name} not found in model. "
                f"Available layers: {list(modules_dict.keys())}"
            )

        target_layer = modules_dict[self.target_layer_name]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].clone()

        # Register forward and backward hooks
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        """
        input_tensor: shape [B, 3, H, W]
        target_class: integer index of class to target. If None, uses predicted class.
        Returns: A list of CAM heatmaps (one per sample in the batch).
        """
        # Forward pass
        self.model.zero_grad()  # Clear any existing gradients
        output = self.model(input_tensor)

        # If target_class is None, use the top predicted class for each sample
        if target_class is None:
            target_class = output.argmax(dim=1)

        # Convert target_class to a list if it's a single tensor
        if isinstance(target_class, torch.Tensor):
            target_class = target_class.cpu().tolist()  # e.g., [class_idx for each sample]

        cams = []
        batch_size = input_tensor.size(0)

        for i in range(batch_size):
            # Backprop for sample i, class target_class[i]
            self.model.zero_grad()  # zero grads for each sample
            class_idx = target_class[i]
            score = output[i, class_idx]  # scalar
            score.backward(retain_graph=(i < batch_size - 1))

            # Get gradients & activations for sample i
            gradients = self.gradients[i]  # shape: [128, 8, 8]
            activations = self.activations[i]  # shape: [128, 8, 8]

            # Compute channel-wise mean of gradients
            alpha = gradients.mean(dim=(1, 2), keepdim=True)  # shape: [128, 1, 1]

            # Linear combination of activations and alpha
            cam = (activations * alpha).sum(dim=0)  # shape: [8, 8]
            cam = F.relu(cam)  # ReLU to keep only positive activations

            # Normalize to [0, 1]
            cam -= cam.min()
            if cam.max() != 0:
                cam /= cam.max()

            cams.append(cam.detach().cpu().numpy())

        return cams
