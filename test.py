"""
Test script for brain MRI classification
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    roc_curve, precision_recall_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, balanced_accuracy_score)
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataloaders import dataloader
from architectures import sfcn_cls, sfcn_ssl2, head, lora_layers
# Import configuration
import config as cfg


def load_model(model_path, device):
    """Load trained model"""
    # Create model architecture based on training mode
    if cfg.TRAINING_MODE == 'sfcn':
        model = sfcn_cls.SFCN(output_dim=cfg.N_CLASSES).to(device)
    
    elif cfg.TRAINING_MODE in ['linear', 'ssl-finetuned', 'lora']:
        backbone = sfcn_ssl2.SFCN()
        
        # For LoRA, apply LoRA layers before loading weights
        if cfg.TRAINING_MODE == 'lora':
            backbone = lora_layers.apply_lora_to_model(
                backbone,
                rank=cfg.LORA_RANK,
                alpha=cfg.LORA_ALPHA,
                target_modules=cfg.LORA_TARGET_MODULES
            )
        
        model = head.ClassifierHeadMLP_(backbone, output_dim=cfg.N_CLASSES).to(device)
    
    else:
        raise ValueError(f"Invalid TRAINING_MODE: {cfg.TRAINING_MODE}")
    
    # Load checkpoint
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Successfully loaded {cfg.TRAINING_MODE} model")
    model.eval()
    return model


def bootstrap_auc(y_true, y_score, curve="roc", n_bootstraps=1000, seed=42):
    """Calculate AUC with bootstrap confidence intervals"""
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        if curve == "roc":
            fpr, tpr, _ = roc_curve(y_true[indices], y_score[indices])
            score = auc(fpr, tpr)
        elif curve == "prc":
            precision, recall, _ = precision_recall_curve(y_true[indices], y_score[indices])
            score = auc(recall, precision)
        
        bootstrapped_scores.append(score)
    
    lower = np.percentile(bootstrapped_scores, 2.5)
    upper = np.percentile(bootstrapped_scores, 97.5)
    return np.mean(bootstrapped_scores), lower, upper


def plot_roc_curve(y_true, y_score, test_cohort, save_path=None):
    """Plot ROC curve with confidence intervals"""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    roc_mean, roc_lower, roc_upper = bootstrap_auc(y_true, y_score, curve="roc")
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2,
             label=f"ROC (AUC = {roc_auc:.2f} [{roc_lower:.2f}–{roc_upper:.2f}])")
    plt.plot([0, 1], [0, 1], lw=1, ls="--", color="gray")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title(f"ROC Curve — {cfg.TRAINING_MODE} on {test_cohort}", fontsize=14)
    plt.legend(loc="lower right", frameon=False, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    plt.show()
    plt.close()


def plot_prc_curve(y_true, y_score, test_cohort, save_path=None):
    """Plot Precision-Recall curve with confidence intervals"""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    prc_auc = auc(recall, precision)
    prc_mean, prc_lower, prc_upper = bootstrap_auc(y_true, y_score, curve="prc")
    pos_rate = y_true.mean()
    
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, lw=2,
             label=f"PRC (AUC = {prc_auc:.2f} [{prc_lower:.2f}–{prc_upper:.2f}])")
    plt.hlines(pos_rate, 0, 1, colors="gray", linestyles="--",
               label=f"Baseline = {pos_rate:.3f}")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.title(f"PRC Curve — {cfg.TRAINING_MODE} on {test_cohort}", fontsize=14)
    plt.legend(loc="lower left", frameon=False, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PRC curve saved to {save_path}")
    plt.show()
    plt.close()


def plot_confusion_matrix(y_true, y_score, threshold='youden', save_path=None):
    """Plot confusion matrix at specified threshold"""
    # Determine threshold
    if threshold == 'youden':
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        threshold_value = thresholds[np.argmax(tpr - fpr)]
    elif isinstance(threshold, (int, float)):
        threshold_value = threshold
    else:
        threshold_value = 0.5
    
    # Make predictions
    y_pred = (y_score >= threshold_value).astype(int)
    
    # Calculate confusion matrix and metrics
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = (tp + tn) / cm.sum()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bacc = balanced_accuracy_score(y_true, y_pred)
    
    print(f"\nThreshold: {threshold_value:.3f}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}")
    print(f"Specificity={spec:.3f}, F1={f1:.3f}, Balanced Acc={bacc:.3f}")
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(cm, display_labels=["0", "1"])
    disp.plot(cmap="Blues", colorbar=False, values_format="d")
    plt.grid(False)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.show()
    plt.close()
    
    return {
        'threshold': threshold_value,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'specificity': spec, 'f1': f1, 'balanced_accuracy': bacc
    }


def test(test_csv, test_cohort, model_path, output_dir):
    """Main test function"""
    device = cfg.DEVICE
    
    print("\n" + "="*70)
    print("TESTING")
    print("="*70)
    print(f"Test cohort: {test_cohort}")
    print(f"Test CSV: {test_csv}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = load_model(model_path, device)
    
    # Create test dataset
    tensor_dir = f'/mnt/bulk-neptune/radhika/project/images/{test_cohort}/npy{cfg.IMG_SIZE}'
    test_dataset = dataloader.BrainDataset(
        csv_file=cfg.CSV_TEST,
        root_dir=cfg.TENSOR_DIR_TEST,
        column_name=cfg.COLUMN_NAME,
        num_rows=None,
        num_classes=cfg.N_CLASSES,
        task='classification'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        drop_last=False
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Run inference
    test_outputs_binary = []
    test_labels = []
    test_eids = []
    
    print("\nRunning inference...")
    with torch.no_grad():
        for eid, images, labels in tqdm(test_loader, desc="Testing"):
            test_eids.extend(eid)
            images = images.to(device)
            labels = labels.float().to(device)
            binary_labels = labels[:, 1]
            test_labels.extend(binary_labels.tolist())
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            binary_outputs = probs[:, 1]
            test_outputs_binary.extend(binary_outputs.tolist())
    
    # Convert to numpy
    y_true = np.array(test_labels).astype(int)
    y_score = np.array(test_outputs_binary).astype(float)
    
    # Calculate metrics with bootstrapping
    print("\nCalculating metrics...")
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    
    # Bootstrap confidence intervals
    bootstrapped_auroc = []
    bootstrapped_auprc = []
    rng = np.random.RandomState(42)
    
    for _ in range(1000):
        indices = rng.randint(0, len(y_true), len(y_true))
        y_true_sample = y_true[indices]
        y_score_sample = y_score[indices]
        
        if len(np.unique(y_true_sample)) < 2:
            continue
        
        bootstrapped_auroc.append(roc_auc_score(y_true_sample, y_score_sample))
        bootstrapped_auprc.append(average_precision_score(y_true_sample, y_score_sample))
    
    # Calculate confidence intervals
    ci_auroc_lower = np.percentile(bootstrapped_auroc, 2.5)
    ci_auroc_upper = np.percentile(bootstrapped_auroc, 97.5)
    ci_auprc_lower = np.percentile(bootstrapped_auprc, 2.5)
    ci_auprc_upper = np.percentile(bootstrapped_auprc, 97.5)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"AUROC: {auroc:.3f} (95% CI: {ci_auroc_lower:.3f}–{ci_auroc_upper:.3f})")
    print(f"AUPRC: {auprc:.3f} (95% CI: {ci_auprc_lower:.3f}–{ci_auprc_upper:.3f})")
    print("="*70)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'eid': test_eids,
        'label': test_labels,
        'prediction': test_outputs_binary
    })
    pred_path = os.path.join(output_dir, cfg.EXPERIMENT_NAME)
    predictions_df.to_csv(pred_path, index=False)
    print(f"\nPredictions saved to {pred_path}")
    
    # Save summary metrics
    summary_df = pd.DataFrame([{
        'test_cohort': test_cohort,
        'AUROC': auroc,
        'AUROC_CI_lower': ci_auroc_lower,
        'AUROC_CI_upper': ci_auroc_upper,
        'AUPRC': auprc,
        'AUPRC_CI_lower': ci_auprc_lower,
        'AUPRC_CI_upper': ci_auprc_upper
    }])
    summary_path = os.path.join(output_dir, 'summary_metrics.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary metrics saved to {summary_path}")
    
    # Plot ROC curve
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plot_roc_curve(y_true, y_score, test_cohort, save_path=roc_path)
    
    # Plot PRC curve
    prc_path = os.path.join(output_dir, 'prc_curve.png')
    plot_prc_curve(y_true, y_score, test_cohort, save_path=prc_path)
    
    # Plot confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    cm_metrics = plot_confusion_matrix(y_true, y_score, threshold='youden', save_path=cm_path)
    
    # Save confusion matrix metrics
    cm_df = pd.DataFrame([cm_metrics])
    cm_metrics_path = os.path.join(output_dir, 'confusion_matrix_metrics.csv')
    cm_df.to_csv(cm_metrics_path, index=False)
    print(f"Confusion matrix metrics saved to {cm_metrics_path}")
    
    print("\nTesting completed!")
    return summary_df


def main():
    """Main function"""
    # Setup
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.DEVICE)
    torch.manual_seed(42)
    
    # Test configuration - MODIFY THESE
    MODEL_PATH = os.path.join(cfg.SAVE_MODEL_DIR, f'{cfg.EXPERIMENT_NAME}_best.pth')
  
    print("\n" + "="*70)
    print("TEST CONFIGURATION")
    print("="*70)
    print(f"Training mode: {cfg.TRAINING_MODE}")
    print(f"Model: {cfg.MODEL_PATH}")
    print(f"Test cohort: {cfg.TEST_COHORT}")
    print(f"Test CSV: {cfg.CSV_TEST}")
    print(f"Output directory: {cfg.SCORES_TEST_DIR}")
    print("="*70)
    
    # Run testing
    results = test(cfg.CSV_TEST, cfg.TEST_COHORT, cfg.MODEL_PATH, cfg.SCORES_TEST_DIR)
    
    print("\n✓ All done!")


if __name__ == "__main__":
    main()
