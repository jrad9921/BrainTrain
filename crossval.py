"""
Cross-validation script for brain MRI classification
Performs k-fold cross-validation on the entire dataset
"""
import os
import time
import datetime
import random
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm import tqdm
from dataloaders import dataloader
import config as cfg
from src import models, distribution, criterions


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch_info=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    outputs_list = []
    labels_list = []
    eids_list = []
    
    # Create description for progress bar
    desc = "Training"
    if epoch_info:
        desc = f"Epoch {epoch_info['epoch']}/{epoch_info['total']} | LR {epoch_info['lr']:.6f} | No improve: {epoch_info['no_improve']}/{epoch_info['patience']}"
    
    for eid, images, labels in tqdm(train_loader, desc=desc, total=len(train_loader), leave=False):
        images = images.to(device)
        
        if cfg.TASK == 'classification':
            labels = labels.float().to(device)
            outputs = model(images).to(device)
            
            binary_labels = labels[:, 1]
            probs = torch.nn.functional.softmax(outputs, dim=1)
            binary_outputs = probs[:, 1]
            
            loss = criterion(outputs, labels)
            
            outputs_list.extend(binary_outputs.tolist())
            labels_list.extend(binary_labels.tolist())
        
        elif cfg.TASK == 'regression':
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(images).to(device)
            
            loss = criterion(outputs, labels)
            
            outputs_list.extend(outputs.squeeze().tolist())
            labels_list.extend(labels.squeeze().tolist())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        eids_list.extend(eid)
    
    avg_loss = running_loss / len(train_loader)
    return avg_loss, eids_list, outputs_list, labels_list


def validate_epoch(model, val_loader, criterion, device, epoch_info=None):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    outputs_list = []
    labels_list = []
    eids_list = []
    
    # Create description for progress bar
    desc = "Validation"
    if epoch_info:
        desc = f"Validating | Best: {epoch_info.get('best_loss', float('inf')):.4f}"
    
    with torch.no_grad():
        for eid, images, labels in tqdm(val_loader, desc=desc, total=len(val_loader), leave=False):
            images = images.to(device)
            
            if cfg.TASK == 'classification':
                labels = labels.float().to(device)
                outputs = model(images).to(device)
                
                binary_labels = labels[:, 1]
                probs = torch.nn.functional.softmax(outputs, dim=1)
                binary_outputs = probs[:, 1]
                
                loss = criterion(outputs, labels)
                
                outputs_list.extend(binary_outputs.tolist())
                labels_list.extend(binary_labels.tolist())
            
            elif cfg.TASK == 'regression':
                labels = labels.float().unsqueeze(1).to(device)
                outputs = model(images).to(device)
                
                loss = criterion(outputs, labels)
                
                outputs_list.extend(outputs.squeeze().tolist())
                labels_list.extend(labels.squeeze().tolist())
            
            running_loss += loss.item()
            eids_list.extend(eid)
    
    avg_loss = running_loss / len(val_loader)
    return avg_loss, eids_list, outputs_list, labels_list


def save_predictions(eids, labels, predictions, save_path):
    """Save predictions to CSV"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    data = {
        'eid': eids,
        'label': labels,
        'prediction': predictions,
    }
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f'Predictions saved to {save_path}')


def train_fold(fold, train_indices, val_indices, test_indices, full_dataset, device, train_labels=None):
    """Train a single fold with train/val/test split"""
    print(f"\n{'='*70}")
    print(f"FOLD {fold + 1}")
    print(f"{'='*70}")
    
    # Create data loaders for this fold
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    test_subset = Subset(full_dataset, test_indices)
    
    train_loader = DataLoader(
        train_subset, 
        batch_size=cfg.BATCH_SIZE, 
        num_workers=cfg.NUM_WORKERS,
        shuffle=True
    )
    val_loader = DataLoader(
        val_subset, 
        batch_size=cfg.BATCH_SIZE, 
        num_workers=cfg.NUM_WORKERS
    )
    test_loader = DataLoader(
        test_subset, 
        batch_size=cfg.BATCH_SIZE, 
        num_workers=cfg.NUM_WORKERS
    )
    
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    print(f"Test samples: {len(test_indices)}")
    
    if cfg.TASK == 'classification' and train_labels is not None:
        fold_train_labels = [train_labels[i] for i in train_indices]
        fold_val_labels = [train_labels[i] for i in val_indices]
        fold_test_labels = [train_labels[i] for i in test_indices]
        print(f"Training distribution: {Counter(fold_train_labels)}")
        print(f"Validation distribution: {Counter(fold_val_labels)}")
        print(f"Test distribution: {Counter(fold_test_labels)}")
    
    # Create model and optimizer
    model, optimizer = models.create_model(device)
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cfg.SCHEDULER_MODE,
        factor=cfg.SCHEDULER_FACTOR,
        patience=cfg.SCHEDULER_PATIENCE
    )
    
    # Get criterion (use train_labels from full dataset if available)
    criterion = criterions.get_criterion(device, train_labels)
    
    # Training loop for this fold
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_val_outputs = None
    best_val_labels = None
    best_val_eids = None
    best_train_outputs = None
    best_train_labels = None
    best_train_eids = None
    best_epoch = 0
    best_model_state = None
    
    fold_history = []
    
    for epoch in range(cfg.NUM_EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        
        # Prepare epoch info for progress bars
        epoch_info = {
            'epoch': epoch + 1,
            'total': cfg.NUM_EPOCHS,
            'lr': current_lr,
            'no_improve': early_stop_counter,
            'patience': cfg.PATIENCE,
            'best_loss': best_val_loss if best_val_loss != float('inf') else 0
        }
        
        # Train
        train_loss, train_eids, train_preds, train_lbls = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch_info
        )
        
        # Validate
        val_loss, val_eids, val_preds, val_lbls = validate_epoch(
            model, val_loader, criterion, device, epoch_info
        )
        
        scheduler.step(val_loss)
        
        fold_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': current_lr
        })
        
        # Save best model for this fold
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_outputs = val_preds
            best_val_labels = val_lbls
            best_val_eids = val_eids
            best_train_outputs = train_preds
            best_train_labels = train_lbls
            best_train_eids = train_eids
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= cfg.PATIENCE:
            print(f'Early stopping at epoch {epoch + 1}')
            break
    
    print(f"Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4f}")
    
    # Evaluate on test set with best model
    print(f"Evaluating on test set...")
    model.load_state_dict(best_model_state)
    test_loss, test_eids, test_preds, test_lbls = validate_epoch(
        model, test_loader, criterion, device
    )
    print(f"Test loss: {test_loss:.4f}")
    
    return {
        'fold': fold,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'val_eids': best_val_eids,
        'val_labels': best_val_labels,
        'val_preds': best_val_outputs,
        'train_eids': best_train_eids,
        'train_labels': best_train_labels,
        'train_preds': best_train_outputs,
        'test_eids': test_eids,
        'test_labels': test_lbls,
        'test_preds': test_preds,
        'test_loss': test_loss,
        'history': fold_history,
        'model_state': best_model_state,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices
    }


def cross_validate():
    """Main cross-validation function"""
    device = cfg.DEVICE
    
    print("\n" + "="*70)
    print("STARTING CROSS-VALIDATION")
    print("="*70)
    start_time = time.time()
    
    # Load single CSV from data/{cohort} folder
    print("\nLoading dataset...")
    
    # Construct path to single CSV file
    if hasattr(cfg, 'DATA_DIR') and hasattr(cfg, 'TRAIN_COHORT'):
        cohort_dir = os.path.join(cfg.DATA_DIR, cfg.TRAIN_COHORT)
        # Look for CSV file in the cohort directory
        csv_files = [f for f in os.listdir(cohort_dir) if f.endswith('.csv')]
        if len(csv_files) == 0:
            raise ValueError(f"No CSV file found in {cohort_dir}")
        elif len(csv_files) > 1:
            print(f"Multiple CSV files found in {cohort_dir}: {csv_files}")
            print(f"Using first file: {csv_files[0]}")
        
        csv_path = os.path.join(cohort_dir, csv_files[0])
    else:
        # Fallback to CSV_FULL if DATA_DIR not configured
        if hasattr(cfg, 'CSV_FULL') and cfg.CSV_FULL:
            csv_path = cfg.CSV_FULL
        else:
            raise ValueError("No CSV file path configured")
    
    print(f"Loading CSV from: {csv_path}")
    combined_df = pd.read_csv(csv_path)
    print(f"Total samples: {len(combined_df)}")
    
    # Create full dataset
    full_dataset = dataloader.BrainDataset(
        csv_path,
        cfg.TENSOR_DIR,
        cfg.COLUMN_NAME,
        task=cfg.TASK,
        num_classes=cfg.N_CLASSES if cfg.TASK == 'classification' else None,
        num_rows=cfg.NROWS
    )
    
    # Get labels for stratification
    if cfg.TASK == 'classification':
        all_labels = full_dataset.annotations[cfg.COLUMN_NAME].values
        print(f"Overall distribution: {Counter(all_labels)}")
    else:
        all_labels = None
        print(f"Total samples: {len(full_dataset)}")
    
    # Set up K-fold cross-validation
    n_splits = getattr(cfg, 'N_FOLDS', 3)  # Default to 3-fold
    
    if cfg.TASK == 'classification':
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.SEED)
        splits = list(kfold.split(np.zeros(len(full_dataset)), all_labels))
    else:
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.SEED)
        splits = list(kfold.split(np.zeros(len(full_dataset))))
    
    # For 3-fold: create train/val/test splits where each fold gets 33.33% as test
    # Fold 1: train on fold2+fold3[0:50%], val on fold3[50:100%], test on fold1
    # Fold 2: train on fold3+fold1[0:50%], val on fold1[50:100%], test on fold2
    # Fold 3: train on fold1+fold2[0:50%], val on fold2[50:100%], test on fold3
    
    final_splits = []
    for i in range(n_splits):
        # Current fold is TEST set
        test_idx = splits[i][1]
        
        # Get the OTHER folds for train/val
        other_folds = []
        for j in range(n_splits):
            if j != i:
                other_folds.extend(splits[j][1].tolist())
        
        # Split the other folds into train (50%) and val (50%)
        # This gives us 33% train, 33% val, 33% test overall
        other_folds = np.array(other_folds)
        
        if cfg.TASK == 'classification':
            # Stratified split of the other folds
            other_labels = [all_labels[idx] for idx in other_folds]
            train_val_kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=cfg.SEED+i)
            train_idx_local, val_idx_local = next(train_val_kfold.split(np.zeros(len(other_folds)), other_labels))
        else:
            # Random split of the other folds
            n_other = len(other_folds)
            indices = np.arange(n_other)
            np.random.seed(cfg.SEED + i)
            np.random.shuffle(indices)
            split_point = n_other // 2
            train_idx_local = indices[:split_point]
            val_idx_local = indices[split_point:]
        
        # Map back to original dataset indices
        train_idx = other_folds[train_idx_local].tolist()
        val_idx = other_folds[val_idx_local].tolist()
        test_idx = test_idx.tolist()
        
        final_splits.append((train_idx, val_idx, test_idx))
    
    # Create output directories

    trainlog_dir = os.path.join(cfg.LOG_DIR, 'trainlog', cfg.TRAINING_MODE)
    vallog_dir = os.path.join(cfg.LOG_DIR, 'vallog', cfg.TRAINING_MODE)
    timelog_dir = os.path.join(cfg.LOG_DIR, 'timelog', cfg.TRAINING_MODE)
    model_dir = os.path.join(cfg.MODEL_DIR, cfg.TRAINING_MODE)
    scores_dir = os.path.join(cfg.SCORES_DIR, cfg.TRAINING_MODE)
    
    os.makedirs(trainlog_dir, exist_ok=True)
    os.makedirs(vallog_dir, exist_ok=True)
    os.makedirs(timelog_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(scores_dir, exist_ok=True)
    
    # Train each fold
    fold_results = []
    all_val_eids = []
    all_val_labels = []
    all_val_preds = []
    all_test_eids = []
    all_test_labels = []
    all_test_preds = []
    
    for fold, (train_idx, val_idx, test_idx) in enumerate(final_splits):
        fold_result = train_fold(
            fold, 
            train_idx, 
            val_idx,
            test_idx,
            full_dataset, 
            device,
            train_labels=all_labels.tolist() if all_labels is not None else None
        )
        
        fold_results.append(fold_result)
        
        # Accumulate validation predictions from all folds
        all_val_eids.extend(fold_result['val_eids'])
        all_val_labels.extend(fold_result['val_labels'])
        all_val_preds.extend(fold_result['val_preds'])
        
        # Accumulate test predictions from all folds
        all_test_eids.extend(fold_result['test_eids'])
        all_test_labels.extend(fold_result['test_labels'])
        all_test_preds.extend(fold_result['test_preds'])
        
        # Save fold-specific predictions
        train_pred_path = os.path.join(scores_dir, 'train', cfg.TRAIN_COHORT, 
                                       f"{cfg.EXPERIMENT_NAME}_fold{fold+1}.csv")
        val_pred_path = os.path.join(scores_dir, 'val', cfg.TRAIN_COHORT,
                                     f"{cfg.EXPERIMENT_NAME}_fold{fold+1}.csv")
        test_pred_path = os.path.join(scores_dir, 'test', cfg.TRAIN_COHORT,
                                      f"{cfg.EXPERIMENT_NAME}_fold{fold+1}.csv")
        
        save_predictions(fold_result['train_eids'], fold_result['train_labels'], 
                        fold_result['train_preds'], train_pred_path)
        save_predictions(fold_result['val_eids'], fold_result['val_labels'], 
                        fold_result['val_preds'], val_pred_path)
        save_predictions(fold_result['test_eids'], fold_result['test_labels'], 
                        fold_result['test_preds'], test_pred_path)
        
        # Save fold model
        model_path = os.path.join(model_dir, f"{cfg.EXPERIMENT_NAME}_fold{fold+1}.pth")
        checkpoint = {
            "fold": fold + 1,
            "best_epoch": fold_result['best_epoch'],
            "state_dict": fold_result['model_state'],
            "val_loss": fold_result['best_val_loss'],
            "train_indices": fold_result['train_indices'],
            "val_indices": fold_result['val_indices']
        }
        torch.save(checkpoint, model_path)
        
        # Save fold training history
        history_df = pd.DataFrame(fold_result['history'])
        history_path = os.path.join(trainlog_dir, f"{cfg.EXPERIMENT_NAME}_fold{fold+1}.csv")
        history_df.to_csv(history_path, index=False)
    
    # Save concatenated validation predictions from ALL folds
    all_val_pred_path = os.path.join(scores_dir, 'val', cfg.TRAIN_COHORT,
                                     f"{cfg.EXPERIMENT_NAME}_{n_splits}folds.csv")
    save_predictions(all_val_eids, all_val_labels, all_val_preds, all_val_pred_path)
    
    # Save concatenated test predictions from ALL folds (entire dataset)
    all_test_pred_path = os.path.join(scores_dir, 'test', cfg.TRAIN_COHORT,
                                      f"{cfg.EXPERIMENT_NAME}_all_folds.csv")
    save_predictions(all_test_eids, all_test_labels, all_test_preds, all_test_pred_path)
    
    print(f"\n✓ All validation predictions saved to: {all_val_pred_path}")
    print(f"  Total subjects: {len(all_val_eids)}")
    
    print(f"\n✓ All test predictions saved to: {all_test_pred_path}")
    print(f"  Total subjects: {len(all_test_eids)}")
    
    if cfg.TASK == 'classification':
        print(f"  Val label distribution: {Counter(all_val_labels)}")
        print(f"  Test label distribution: {Counter(all_test_labels)}")
    
    # Calculate cross-validation statistics
    val_losses = [r['best_val_loss'] for r in fold_results]
    test_losses = [r['test_loss'] for r in fold_results]
    mean_val_loss = np.mean(val_losses)
    std_val_loss = np.std(val_losses)
    mean_test_loss = np.mean(test_losses)
    std_test_loss = np.std(test_losses)
    
    # Save summary results
    duration = time.time() - start_time
    
    summary_file = os.path.join(vallog_dir, f"{cfg.EXPERIMENT_NAME}_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Cross-Validation Results ({n_splits}-Fold)\n")
        f.write(f"{'='*70}\n")
        f.write(f"\nValidation Performance (for model selection):\n")
        f.write(f"  Mean Validation Loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}\n")
        f.write(f"\nTest Performance (held-out data):\n")
        f.write(f"  Mean Test Loss: {mean_test_loss:.4f} ± {std_test_loss:.4f}\n")
        f.write(f"\nPer-Fold Results:\n")
        for r in fold_results:
            f.write(f"  Fold {r['fold']+1}:\n")
            f.write(f"    Validation Loss: {r['best_val_loss']:.4f} (epoch {r['best_epoch']})\n")
            f.write(f"    Test Loss: {r['test_loss']:.4f}\n")
        f.write(f"\nData Split Per Fold:\n")
        f.write(f"  Train: ~{100/n_splits:.1f}%\n")
        f.write(f"  Val: ~{100/n_splits:.1f}%\n")
        f.write(f"  Test: ~{100/n_splits:.1f}%\n")
        f.write(f"\nOutput Files:\n")
        f.write(f"  Validation predictions: {all_val_pred_path}\n")
        f.write(f"  Test predictions: {all_test_pred_path}\n")
        if cfg.TASK == 'classification':
            f.write(f"  Val label distribution: {Counter(all_val_labels)}\n")
            f.write(f"  Test label distribution: {Counter(all_test_labels)}\n")
        f.write(f"\nTotal Duration: {duration:.2f}s ({duration/60:.2f} min)\n")
    
    timelog_file = os.path.join(timelog_dir, f"{cfg.EXPERIMENT_NAME}.txt")
    with open(timelog_file, "w") as f:
        f.write(f"Duration: {duration:.2f}s ({duration/60:.2f} min)\n")
        f.write(f"Average per fold: {duration/n_splits:.2f}s ({duration/n_splits/60:.2f} min)\n")
        f.write(f"Start: {datetime.datetime.fromtimestamp(start_time)}\n")
        f.write(f"End: {datetime.datetime.fromtimestamp(time.time())}\n")
    
    # Print summary
    print("\n" + "="*70)
    print("CROSS-VALIDATION SUMMARY")
    print("="*70)
    print(f"Number of folds: {n_splits}")
    print(f"Split per fold: ~{100/n_splits:.1f}% train, ~{100/n_splits:.1f}% val, ~{100/n_splits:.1f}% test")
    print(f"\nValidation Performance (for model selection):")
    print(f"  Mean validation loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
    print(f"\nTest Performance (held-out data):")
    print(f"  Mean test loss: {mean_test_loss:.4f} ± {std_test_loss:.4f}")
    print(f"\nPer-fold results:")
    for r in fold_results:
        print(f"  Fold {r['fold']+1}: Val={r['best_val_loss']:.4f} (epoch {r['best_epoch']}), Test={r['test_loss']:.4f}")
    print(f"\nAll subjects evaluated:")
    print(f"  Validation: {len(all_val_eids)} subjects -> {all_val_pred_path}")
    print(f"  Test: {len(all_test_eids)} subjects -> {all_test_pred_path}")
    print(f"\nTotal duration: {duration:.2f}s ({duration/60:.2f} min)")
    print(f"Average per fold: {duration/n_splits:.2f}s ({duration/n_splits/60:.2f} min)")
    print("="*70)
    
    return fold_results, mean_val_loss, std_val_loss, mean_test_loss, std_test_loss, all_val_pred_path, all_test_pred_path


def main():
    """Main function"""
    # Setup
    set_seed(cfg.SEED)
    
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.DEVICE)
    
    # Print config
    n_folds = getattr(cfg, 'N_FOLDS', 3)  # Default to 3-fold
    
    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)
    print(f"Mode: {n_folds}-Fold Cross-Validation")
    print(f"Task: {cfg.TASK}")
    print(f"Cohort: {cfg.TRAIN_COHORT}")
    print(f"Batch Size: {cfg.BATCH_SIZE}")
    print(f"Learning Rate: {cfg.LEARNING_RATE}")
    print(f"Epochs per fold: {cfg.NUM_EPOCHS}")
    print(f"Device: {cfg.DEVICE}")
    print(f"Experiment: {cfg.EXPERIMENT_NAME}")
    print("="*70)
    
    # Run cross-validation
    cross_validate()


if __name__ == "__main__":
    main()
