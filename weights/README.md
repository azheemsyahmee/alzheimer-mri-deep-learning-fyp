# Pretrained Model Weights & Evaluation Cache

This directory stores model checkpoint weights and pre-computed evaluation caches.

## Weight Files

| File | Architecture | Size | Description |
| :--- | :--- | :---: | :--- |
| `resnet_global.pth` | ResNet-18 (Stage 1) | ~42.7 MB | Global feature extractor fine-tuned on balanced brain MRI scans. |
| `patch_cnn.pth` | PatchCNN (Stage 2) | ~8.1 MB | Local lesion classifier trained on Grad-CAM localized patches. |

### Download Model Checkpoints
Because large binary weights can inflate repository size, download weights from the [GitHub Releases](../../releases) section and place them in this `weights/` directory:
- `resnet_global.pth`
- `patch_cnn.pth`

## Evaluation Metrics Cache

The repository includes pre-computed evaluation cache files:
- `results_cache.npz`: Stores test set confusion matrices, loss curves, and per-class precision/recall/F1 metrics across all 5 benchmarked models.
- `alpha_comparison_results.npz`: Stores the sensitivity sweep data across ensemble fusion weights ($\alpha \in [0.6, 0.8]$).

These caches allow running the notebook to reproduce all scientific charts without having to re-train the models for hours.
