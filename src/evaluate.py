"""
CLI Evaluation and Single-Image Inference for Alzheimer's Disease Classification.
"""

import argparse
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from models import build_resnet18, PatchCNN, HybridEnsemble
from gradcam import GradCAM, extract_top_patches

CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def predict_single_image(image_path, resnet_path, patch_path=None, alpha=0.6, device='cpu'):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_pil = Image.open(image_path).convert('RGB')
    tensor_img = transform(img_pil).to(device)
    batch_img = tensor_img.unsqueeze(0)

    # 1. Load ResNet-18
    resnet = build_resnet18(num_classes=4, pretrained=False, freeze_early=False)
    if os.path.exists(resnet_path):
        resnet.load_state_dict(torch.load(resnet_path, map_location=device))
    resnet.to(device).eval()

    # 2. Compute Grad-CAM
    gcam = GradCAM(resnet, resnet.layer4)
    heatmap, pred_idx = gcam.generate_heatmap(batch_img)

    # 3. PatchCNN & Hybrid
    if patch_path and os.path.exists(patch_path):
        patch_model = PatchCNN(num_classes=4)
        patch_model.load_state_dict(torch.load(patch_path, map_location=device))
        patch_model.to(device).eval()

        patches, _ = extract_top_patches(tensor_img, heatmap)
        patch_tensors = [
            torch.tensor(p, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            for p in patches
        ]

        hybrid = HybridEnsemble(resnet, patch_model, alpha=alpha)
        probs = hybrid(batch_img, patch_tensors).squeeze().detach().cpu().numpy()
    else:
        with torch.no_grad():
            logits = resnet(batch_img)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    predicted_class = CLASSES[np.argmax(probs)]
    return predicted_class, probs


def main():
    parser = argparse.ArgumentParser(description="Alzheimer's Disease MRI Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to brain MRI image")
    parser.add_argument("--resnet_weights", type=str, default="weights/resnet_global.pth", help="Path to ResNet weights")
    parser.add_argument("--patch_weights", type=str, default="weights/patch_cnn.pth", help="Path to PatchCNN weights")
    parser.add_argument("--alpha", type=float, default=0.6, help="Weight for ResNet-18 (default: 0.6)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {args.image} using {device}")
    pred, probs = predict_single_image(args.image, args.resnet_weights, args.patch_weights, args.alpha, device)

    print("\n--- Prediction Results ---")
    print(f"Predicted Diagnosis: {pred} (Confidence: {np.max(probs) * 100:.2f}%)")
    for c, p in zip(CLASSES, probs):
        print(f"  {c:<20}: {p * 100:.2f}%")


if __name__ == "__main__":
    main()
