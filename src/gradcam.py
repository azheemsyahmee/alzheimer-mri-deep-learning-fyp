"""
Grad-CAM Explainable AI Utilities & Sliding-Window Patch Extraction.
"""

import cv2
import numpy as np
import torch


class GradCAM:
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM)
    on PyTorch convolutional layers.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.clone()
        for i in range(activations.shape[1]):
            activations[:, i] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze().detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap /= max_val

        return heatmap, class_idx


def compute_brain_mask(img_np):
    """
    Applies binary thresholding to segment brain tissue from background empty space.
    """
    gray = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    return mask / 255.0


def extract_top_patches(img_tensor, heatmap, patch_size=64, step_size=32, top_k=3):
    """
    Extracts top-k highest activation 64x64 patches guided by Grad-CAM heatmap.
    """
    # De-normalize tensor to [0, 1] RGB
    img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
    img_255 = (img_np * 255).astype(np.uint8)

    mask = compute_brain_mask(img_255)
    heat_resized = cv2.resize(heatmap, (224, 224)) * mask
    max_h = np.max(heat_resized)
    if max_h > 0:
        heat_resized /= max_h

    patches = []
    scores = []
    h, w = 224, 224

    for y in range(0, h - patch_size + 1, step_size):
        for x in range(0, w - patch_size + 1, step_size):
            patch_heat = heat_resized[y:y + patch_size, x:x + patch_size]
            score = np.mean(patch_heat)
            patch = img_np[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
            scores.append(score)

    if not patches:
        return [], []

    top_indices = np.argsort(scores)[::-1][:top_k]
    selected_patches = [patches[i] for i in top_indices]
    selected_scores = [scores[i] for i in top_indices]

    return selected_patches, selected_scores
