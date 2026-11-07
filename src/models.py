"""
Neural Network Architectures for Alzheimer's Disease MRI Classification.
Includes:
- ResNet-18 Global Feature Extractor
- PatchCNN Local Lesion Classifier
- HybridEnsemble Weighted Fusion Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PatchCNN(nn.Module):
    """
    Lightweight Convolutional Neural Network trained specifically on localized,
    high-attention patches extracted via Grad-CAM saliency maps.
    Input shape: (B, 3, 64, 64)
    """
    def __init__(self, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def build_resnet18(num_classes=4, pretrained=True, freeze_early=True):
    """
    Constructs ResNet-18 with ImageNet initialization and fine-tunes layer4 + fc head.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_early:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class HybridEnsemble(nn.Module):
    """
    Weighted ensemble fusing global ResNet-18 representations
    with local PatchCNN lesion representations.
    alpha: weighting factor for global ResNet (default: 0.6)
    (1 - alpha): weighting factor for mean patch probabilities (default: 0.4)
    """
    def __init__(self, resnet_model, patch_model, alpha=0.6):
        super().__init__()
        self.resnet = resnet_model
        self.patch_cnn = patch_model
        self.alpha = alpha

    def forward(self, global_img, patch_tensor_list):
        """
        global_img: (1, 3, 224, 224)
        patch_tensor_list: list of (1, 3, 64, 64) tensors
        """
        resnet_logits = self.resnet(global_img)
        p_resnet = F.softmax(resnet_logits, dim=1)

        if not patch_tensor_list:
            return p_resnet

        patch_probs = []
        for p in patch_tensor_list:
            p_logits = self.patch_cnn(p)
            patch_probs.append(F.softmax(p_logits, dim=1))

        mean_p_patch = torch.mean(torch.stack(patch_probs), dim=0)
        p_hybrid = self.alpha * p_resnet + (1.0 - self.alpha) * mean_p_patch
        return p_hybrid
