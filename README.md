# BrainTrain

End-to-end deep learning pipeline for brain MRI classification with explainability.

## Overview

Train and evaluate deep learning models on brain MRI data with built-in explainability methods to visualize model predictions.

## Features

- **Training** - Train deep learning models on brain MRI scans
- **Testing** - Evaluate model performance on test sets
- **Explainability** - Generate GradCAM and saliency maps for predictions

## Project Structure

```
.
├── architectures/          # Neural network models
├── dataloaders/           # Dataset loaders
├── train.py               # Model training script
├── test.py                # Model evaluation script
├── heatmap.py             # GradCAM and saliency visualization
└── config.py              # Configuration and paths
```

## Quick Start

### 1. Train Model

```bash
python train.py
```

Trains the model on your brain MRI dataset and saves checkpoints.

### 2. Test Model

```bash
python test.py
```

Evaluates the trained model on the test set and reports performance metrics.

### 3. Generate Explainability Maps

```bash
python heatmap.py
```

Creates GradCAM and saliency maps to visualize which brain regions influenced model predictions.

## Requirements

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

## Configuration

Edit `config.py` to customize:
- Data paths
- Model architecture
- Training hyperparameters
- Explainability settings

## Output

- **Model checkpoints**: Saved in model directory
- **Test results**: Performance metrics and predictions
- **Heatmaps**: GradCAM and saliency visualizations (NIfTI and PNG formats)

## Explainability Methods

- **GradCAM**: Highlights important regions using gradient-weighted class activation mapping
- **Saliency Maps**: Shows pixel-level importance for predictions