# GradCAM Pet Sentiment Analyzer

A Streamlit app for visualizing what a deep learning model looks at when predicting pet emotions, using multiple Class Activation Map (CAM) methods and ViT attention rollout.

## What is GradCAM?

GradCAM (Gradient-weighted Class Activation Mapping) produces a heatmap over an input image showing which spatial regions most influenced a model's prediction. It works by computing the gradient of the predicted class score with respect to the activations of a chosen convolutional layer, then weighting those activation maps by their gradient importance.

The result is overlaid on the original image, making it possible to audit model behavior and understand what visual features drive a classification decision.

## Supported Explainability Methods

### CAM Variants (via `pytorch-grad-cam`)

| Method | How it differs |
|---|---|
| GradCAM | Gradients of the class score w.r.t. the target layer activations |
| HiResCAM | Element-wise gradient product for sharper, high-resolution maps |
| GradCAM++ | Improved weighting to better handle multi-instance scenes |
| XGradCAM | Normalized gradient CAM with improved faithfulness |
| ScoreCAM | Score-based perturbation approach; no gradients needed |
| AblationCAM | Occludes channels to measure each one's contribution |
| EigenCAM | Uses principal components of feature maps instead of class gradients |
| LayerCAM | Generates per-spatial-location class activation scores |

### ViT Attention Methods (via `vit_grad_rollout.py`)

| Method | How it works |
|---|---|
| ViT Attention Rollout | Propagates raw attention matrices across Transformer layers and reads out the CLS token's total attention over image patches |
| ViT Gradient Rollout | Weights each attention matrix by its gradient before rollout, giving class-discriminative attention maps |

## Target Layer Options

The app lets you choose which layer to hook into for CAM generation:

- **CNN: Last Conv Block** - Spatial features from the ResNet-50 backbone; produces coarser but spatially grounded maps.
- **ViT: Last Transformer Norm (after FFN)** - Captures high-level semantic features after the feed-forward sublayer.
- **ViT: Last Transformer Norm (after Attention)** - Captures attention-conditioned features before the FFN.

## Smoothing Options

Two optional post-processing smoothers improve map quality:

- **AugSmooth** - Averages CAM outputs over augmented versions of the input (flips, crops, color jitter) to reduce noise.
- **EigenSmooth** - Applies PCA to feature map channels to suppress noisy activation patterns.

## Repository Contents

| File | Description |
|---|---|
| `gradcamapp.py` | Main Streamlit app: inference, CAM generation, emotion plotting |
| `vit_grad_rollout.py` | Gradient rollout implementation for the ViT attention layers |
| `timing_combined.json` | Per-epoch training throughput log (50 epochs, ~490-498 img/s) |

> The trained model weights (`pet_sentiment_model.pth`) are not included due to file size. Place the file in the same directory before running.

## Requirements

```
streamlit
torch
torchvision
opencv-python
Pillow
numpy
matplotlib
pytorch-grad-cam
```

```bash
pip install streamlit torch torchvision opencv-python Pillow numpy matplotlib pytorch-grad-cam
```

## Running the App

```bash
streamlit run gradcamapp.py
```

Upload a pet face image, select an explainability method and target layer from the sidebar, and the app displays the original image alongside the generated heatmap.
