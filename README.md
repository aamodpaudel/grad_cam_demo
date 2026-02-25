# Pet Sentiment Analyzer with GradCAM Explainability

A Streamlit application that classifies pet facial emotions using a Hybrid CNN-ViT model and visualizes model decisions through multiple gradient-based explainability methods.

## Overview

This project combines a ResNet-50 CNN backbone with a Transformer encoder to classify pet face images into four emotion categories: **Angry**, **Happy**, **Relaxed**, and **Sad**. The predicted emotion is then mapped onto a 2D valence-arousal space to provide a more nuanced sentiment reading. Multiple CAM (Class Activation Map) methods and ViT attention rollout techniques are available for inspecting what the model attends to.

## Repository Contents

| File | Description |
|---|---|
| `gradcamapp.py` | Main Streamlit application: model definition, inference, CAM generation, and emotion plotting |
| `vit_grad_rollout.py` | Gradient-based attention rollout implementation for the ViT component |
| `timing_combined.json` | Training throughput data for 50 epochs (images/second and wall-clock time per epoch) |

> **Note:** The trained model weights file (`pet_sentiment_model.pth`) is not included in this repository due to file size. Place it in the same directory as `gradcamapp.py` before running the app.

## Model Architecture

The `HybridCNNViT` model follows a two-stage design:

1. **CNN Backbone (ResNet-50)** - Extracts spatial feature maps (2048-channel) from the input image.
2. **Feature Projection** - A 1x1 convolution maps CNN features to 768-dimensional tokens compatible with the Transformer.
3. **Transformer Encoder** - Four layers of multi-head self-attention (8 heads, 768 model dim, 3072 FFN dim, GELU activation) process the 49 patch tokens plus a learnable CLS token.
4. **Classifier Head** - A linear layer maps the CLS token output to 4 emotion classes.

## Supported Explainability Methods

### CAM Methods (via `pytorch-grad-cam`)

| Method | Description |
|---|---|
| GradCAM | Gradient-weighted class activation mapping |
| HiResCAM | High-resolution variant of GradCAM |
| ScoreCAM | Score-based CAM without gradients |
| GradCAM++ | Improved gradient weighting for multi-instance detection |
| AblationCAM | Perturbation-based importance scoring |
| XGradCAM | Normalized gradient CAM |
| EigenCAM | Uses principal components of feature maps |
| LayerCAM | Per-spatial-location class scores |

CAM methods can target either the CNN last convolutional block or the final Transformer normalization layers.

### ViT Attention Methods (via `vit_grad_rollout.py`)

| Method | Description |
|---|---|
| ViT Attention Rollout | Rolls up gradient-weighted attention across Transformer layers |
| ViT Gradient Rollout | Gradient rollout to the CLS token for the predicted class |

## Emotion Mapping

Predicted classes are placed in a 2D valence-arousal space:

| Quadrant | Valence | Arousal | Emotions |
|---|---|---|---|
| Q1 (top-right) | Positive | High | Happy, Excited, Pleased |
| Q2 (top-left) | Negative | High | Angry, Annoyed, Enraged |
| Q3 (bottom-left) | Negative | Low | Sad, Bored, Depressed |
| Q4 (bottom-right) | Positive | Low | Relaxed, Calm, Content |

Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) approximations derived from OpenCV Haar cascade detection are used to determine the exact position within the predicted quadrant.

## Training Throughput

The `timing_combined.json` file contains per-epoch training statistics collected over 50 epochs:

- **Epoch 1:** 297.00 img/s (67.34 s) - initial warm-up
- **Epochs 2-50:** ~490-498 img/s (~40.2-40.8 s per epoch)
- **Total training time:** 2049.31 s (~34.2 minutes)
- **Average per epoch:** 40.99 s

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

Install dependencies:

```bash
pip install streamlit torch torchvision opencv-python Pillow numpy matplotlib pytorch-grad-cam
```

## Running the App

```bash
streamlit run gradcamapp.py
```

Ensure `pet_sentiment_model.pth` is in the same directory as `gradcamapp.py` before launching.

## Usage

1. Launch the app with the command above.
2. Use the sidebar to select an explainability method and target layer.
3. Upload a pet face image (JPG or PNG).
4. The app displays the original image alongside the explanation heatmap.
5. Below the images, the predicted emotion category, specific emotion label, and a valence-arousal plot are shown.

## Notes

- The model runs on GPU if CUDA is available; otherwise it falls back to CPU.
- AugSmooth and EigenSmooth post-processing options are available in the sidebar for CAM methods.
- The ViT attention rollout implementation registers hooks on layers named `attn_drop`. If the model does not contain such layers, the rollout will produce an empty result.
