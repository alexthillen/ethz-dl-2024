import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class ActivationPreservingLoss:
    def __init__(
        self,
        original_model,
        dataset,
        num_samples_per_class,
        layers
    ):
        self.device = next(original_model.parameters()).device
        self.stored_samples = []
        self.stored_activations = {layer: [] for layer in layers}
        self.stored_logits = []
        self.layers = layers

        samples_by_class = {label: [] for label in set(y for _, y in dataset)}

        original_model.eval()
        with torch.no_grad():
            for i in range(len(dataset)):
                x, y_true = dataset[i]
                x = x.unsqueeze(0).to(self.device)
                y_pred = original_model(x).argmax(dim=1).item()

                if y_pred == y_true:
                    if len(samples_by_class.get(y_true, [])) < num_samples_per_class:
                        samples_by_class.setdefault(y_true, []).append(i)

                if all(
                    len(indices) >= num_samples_per_class for indices in samples_by_class.values()
                ):
                    break

        selected_indices = []
        for class_indices in samples_by_class.values():
            selected_indices.extend(
                random.sample(class_indices, min(num_samples_per_class, len(class_indices)))
            )

        activations = {layer: None for layer in layers}
        handles = []

        def get_hook(layer_name):
            def hook(module, input, output):
                activations[layer_name] = output.detach()

            return hook

        for layer in layers:
            target_layer = dict([*original_model.named_modules()])[layer]
            handle = target_layer.register_forward_hook(get_hook(layer))
            handles.append(handle)

        for idx in selected_indices:
            x, y = dataset[idx]
            x = x.to(self.device)
            self.stored_samples.append(x.clone())

            logits = original_model(x.unsqueeze(0))
            self.stored_logits.append(logits.detach().squeeze(0))

            for layer in layers:
                self.stored_activations[layer].append(activations[layer].squeeze(0))

        for handle in handles:
            handle.remove()

        print(
            f"Stored {len(self.stored_samples)} samples total ({len(samples_by_class)} classes, "
            f"~{len(self.stored_samples)/len(samples_by_class):.1f} samples per class)"
        )

        self.stored_samples = torch.stack(self.stored_samples)
        self.stored_logits = torch.stack(self.stored_logits)
        for layer in layers:
            self.stored_activations[layer] = torch.stack(self.stored_activations[layer])

    def __call__(self, model):
        activations = {layer: None for layer in self.layers}
        handles = []

        def get_hook(layer_name):
            def hook(module, input, output):
                activations[layer_name] = output

            return hook

        for layer in self.layers:
            target_layer = dict([*model.named_modules()])[layer]
            handle = target_layer.register_forward_hook(get_hook(layer))
            handles.append(handle)

        logits = model(self.stored_samples)
        loss = F.mse_loss(logits, self.stored_logits)

        for layer in self.layers:
            loss += F.mse_loss(activations[layer], self.stored_activations[layer])

        for handle in handles:
            handle.remove()

        return loss
