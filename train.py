"""
Main training script for brain MRI classification
"""
import os
import sys
import time
import datetime
import random
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import monai
from dataloaders import dataloader
from architectures import sfcn_cls, sfcn_ssl2, head, lora_layers
# Import configuration
import config as cfg

def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_directories():
    """Create all necessary output directories"""
    dirs = [
        cfg.SAVE_MODEL_DIR,
        cfg.SCORES_TRAIN_DIR,
        cfg.SCORES_VAL_DIR,
        cfg.TRAINLOG_DIR,
        cfg.VALLOG_DIR,
        cfg.TIMELOG_DIR,
    ]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    print("Output directories created")


def check_data_distribution(csv_path, column_name, task='classification'):
    """Check and print data distribution"""
    df = pd.read_csv(csv_path)
    print(f"\nDataset shape: {df.shape}")
    
    if task == 'classification':
        ratio = (df[column_name] == 1).sum() / (df[column_name] == 0).sum()
        counts = df[column_name].value_counts()
        print(f"Class distribution:\n{counts}")
        print(f"Ratio (positive/negative): {ratio:.3f}")
    
    elif task == 'regression':
        print(f"Target variable: {column_name}")
        print(f"Mean: {df[column_name].mean():.2f}")
        print(f"Std: {df[column_name].std():.2f}")
        print(f"Min: {df[column_name].min():.2f}")
        print(f"Max: {df[column_name].max():.2f}")
        print(f"Median: {df[column_name].median():.2f}")
    
    return df


def create_model(device):
    """Create model based on training mode and task"""
    
    # Determine output dimension based on task
    if cfg.TASK == 'regression':
        output_dim = 1
    else:
        output_dim = cfg.N_CLASSES
    
    if cfg.TRAINING_MODE == 'sfcn':
        model = sfcn_cls.SFCN(output_dim=output_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), cfg.LEARNING_RATE)
        print(f"Using SFCN for {cfg.TASK}")
    
    elif cfg.TRAINING_MODE == 'dense':
        model = monai.networks.nets.DenseNet121(
            spatial_dims=3, 
            in_channels=cfg.N_CHANNELS, 
            out_channels=output_dim
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), cfg.LEARNING_RATE)
        print(f"Using DenseNet121 for {cfg.TASK}")
    
    elif cfg.TRAINING_MODE in ['linear', 'ssl-finetuned']:
        backbone = sfcn_ssl2.SFCN()
        checkpoint = torch.load(cfg.PRETRAINED_MODEL, map_location=device)
        backbone.load_state_dict(checkpoint['state_dict'], strict=False)
        model = head.ClassifierHeadMLP_(backbone, output_dim=output_dim).to(device)
        
        if cfg.TRAINING_MODE == 'linear':
            for param in model.backbone.parameters():
                param.requires_grad = False
            optimizer = torch.optim.AdamW(model.classifier.parameters(), cfg.LEARNING_RATE)
            print(f"Using Linear Probing for {cfg.TASK}")
        else:
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), cfg.LEARNING_RATE)
            print(f"Using SSL Fine-tuning for {cfg.TASK}")
    
    elif cfg.TRAINING_MODE == 'lora':
        backbone = sfcn_ssl2.SFCN()
        checkpoint = torch.load(cfg.PRETRAINED_MODEL, map_location=device)
        backbone.load_state_dict(checkpoint['state_dict'], strict=False)
        
        backbone = lora_layers.apply_lora_to_model(
            backbone,
            rank=cfg.LORA_RANK,
            alpha=cfg.LORA_ALPHA,
            target_modules=cfg.LORA_TARGET_MODULES
        )
        
        for name, param in backbone.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
        
        model = head.ClassifierHeadMLP_(backbone, output_dim=output_dim)
        model = model.to(device)
        
        lora_params = [p for n, p in model.backbone.named_parameters() 
                      if 'lora' in n and p.requires_grad]
        classifier_params = list(model.classifier.parameters())
        
        optimizer = torch.optim.AdamW(lora_params + classifier_params, cfg.LEARNING_RATE)
        
        print(f"LoRA applied for {cfg.TASK}")
        print(f"Trainable LoRA params: {sum(p.numel() for p in lora_params):,}")
        print(f"Trainable classifier params: {sum(p.numel() for p in classifier_params):,}")
    
    else:
        raise ValueError(f"Invalid TRAINING_MODE: {cfg.TRAINING_MODE}")
    
    return model, optimizer

def get_criterion(device, train_labels=None):
    """Get loss function based on task"""
    if cfg.TASK == 'classification':
        if train_labels is not None:
            class_weights = compute_class_weight(
                'balanced',
                classes=np.array([0, 1]),
                y=train_labels
            )
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
            print(f"Class weights: {class_weights}")
            criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)
    
    elif cfg.TASK == 'regression':
        criterion =nn.MSELoss().to(device)
        print("Using MSE Loss for regression")
    else:
        raise ValueError(f"Invalid TASK: {cfg.TASK}")
    
    return criterion

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    outputs_list = []
    labels_list = []
    eids_list = []
    
    for eid, images, labels in tqdm(train_loader, desc="Training", total=len(train_loader)):
        images = images.to(device)
        
        if cfg.TASK == 'classification':
            labels = labels.float().to(device)
            outputs = model(images).to(device)
            
            # Get binary labels and predictions
            binary_labels = labels[:, 1]
            probs = torch.nn.functional.softmax(outputs, dim=1)
            binary_outputs = probs[:, 1]
            
            loss = criterion(outputs, labels)
            
            outputs_list.extend(binary_outputs.tolist())
            labels_list.extend(binary_labels.tolist())
        
        elif cfg.TASK == 'regression':
            labels = labels.float().unsqueeze(1).to(device)  # Shape: [batch_size, 1]
            outputs = model(images).to(device)  # Shape: [batch_size, 1]
            
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


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    outputs_list = []
    labels_list = []
    eids_list = []
    
    with torch.no_grad():
        for eid, images, labels in tqdm(val_loader, desc="Validation", total=len(val_loader)):
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
    data = {
        'eid': eids,
        'label': labels,
        'prediction': predictions,
    }
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f'Predictions saved to {save_path}')


def train():
    """Main training function"""
    device = cfg.DEVICE
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    start_time = time.time()
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = dataloader.BrainDataset(
        cfg.CSV_TRAIN,
        cfg.TENSOR_DIR,
        cfg.COLUMN_NAME,
        task=cfg.TASK,
        num_classes=cfg.N_CLASSES if cfg.TASK == 'classification' else None,
        num_rows=cfg.NROWS
    )
    
    val_dataset = dataloader.BrainDataset(
        cfg.CSV_VAL,
        cfg.TENSOR_DIR,
        cfg.COLUMN_NAME,
        task=cfg.TASK,
        num_classes=cfg.N_CLASSES if cfg.TASK == 'classification' else None,
        num_rows=cfg.NROWS
    )
    
    # Check distribution
    if cfg.TASK == 'classification':
        train_labels = train_dataset.annotations[cfg.COLUMN_NAME].values.tolist()
        val_labels = val_dataset.annotations[cfg.COLUMN_NAME].values.tolist()
        print(f"\nTraining set - {Counter(train_labels)}")
        print(f"Validation set - {Counter(val_labels)}")
    else:
        train_labels = None  # Important: set to None for regression
        print(f"\nTraining set size: {len(train_dataset)}")
        print(f"Validation set size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        num_workers=cfg.NUM_WORKERS,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.BATCH_SIZE, 
        num_workers=cfg.NUM_WORKERS
    )
    
    # Create model and optimizer
    print("\nInitializing model...")
    model, optimizer = create_model(device)
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cfg.SCHEDULER_MODE,
        factor=cfg.SCHEDULER_FACTOR,
        patience=cfg.SCHEDULER_PATIENCE
    )
    
    # Get appropriate criterion - THIS IS THE KEY FIX
    criterion = get_criterion(device, train_labels)

    # Initialize logging
    trainlog_file = os.path.join(cfg.TRAINLOG_DIR, f"{cfg.EXPERIMENT_NAME}.txt")
    with open(trainlog_file, "w") as log:
        log.write('Epoch, Training Loss, Validation Loss, Learning Rate\n')
    
    # Training loop
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_val_outputs = None
    best_val_labels = None
    best_val_eids = None
    
    print("\n" + "="*70)
    print("TRAINING LOOP")
    print("="*70)
    
    for epoch in range(cfg.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{cfg.NUM_EPOCHS} ---")
        
        # Train
        train_loss, train_eids, train_preds, train_lbls = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_eids, val_preds, val_lbls = validate_epoch(
            model, val_loader, criterion, device
        )
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            print(f"✓ New best model! (previous: {best_val_loss:.4f}, current: {val_loss:.4f})")
            best_val_loss = val_loss
            best_val_outputs = val_preds
            best_val_labels = val_lbls
            best_val_eids = val_eids
            
            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss
            }
            
            model_path = os.path.join(cfg.SAVE_MODEL_DIR, f"{cfg.EXPERIMENT_NAME}_best.pth")
            torch.save(checkpoint, model_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"No improvement ({early_stop_counter}/{cfg.PATIENCE})")
        
        # Early stopping
        if early_stop_counter >= cfg.PATIENCE:
            print(f'\n⚠ Early stopping triggered after {epoch + 1} epochs')
            break
        
        # Log to file
        with open(trainlog_file, "a") as log:
            log.write(f'{epoch + 1}, {train_loss:.4f}, {val_loss:.4f}, {current_lr:.6e}\n')
    
    # Save final predictions
    print("\nSaving predictions...")
    save_predictions(
        train_eids, train_lbls, train_preds,
        os.path.join(cfg.SCORES_TRAIN_DIR, f"{cfg.EXPERIMENT_NAME}.csv")
    )
    save_predictions(
        best_val_eids, best_val_labels, best_val_outputs,
        os.path.join(cfg.SCORES_VAL_DIR, f"{cfg.EXPERIMENT_NAME}.csv")
    )
    
    # Log results
    duration = time.time() - start_time
    
    vallog_file = os.path.join(cfg.VALLOG_DIR, f"{cfg.EXPERIMENT_NAME}.txt")
    with open(vallog_file, "w") as log:
        log.write(f'Training completed\n')
        log.write(f'Best Validation Loss: {best_val_loss:.4f}\n')
        log.write(f'Stopped at epoch: {epoch + 1}\n')
    
    timelog_file = os.path.join(cfg.TIMELOG_DIR, f"{cfg.EXPERIMENT_NAME}.txt")
    with open(timelog_file, "w") as log:
        log.write(f"Duration: {duration:.2f}s ({duration/60:.2f} min)\n")
        log.write(f"Start: {datetime.datetime.fromtimestamp(start_time)}\n")
        log.write(f"End: {datetime.datetime.fromtimestamp(time.time())}\n")
        log.write(f"Total params: {sum(p.numel() for p in model.parameters()):,}\n")
        log.write(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Duration: {duration:.2f}s ({duration/60:.2f} min)")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved: {cfg.SAVE_MODEL_DIR}/{cfg.EXPERIMENT_NAME}_best.pth")
    print("="*70)
    
    return model, best_val_loss


def main():
    """Main function"""
    # Setup
    set_seed(cfg.SEED)
    
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.DEVICE)
    
    # Create directories
    create_directories()
    
    # Print config
    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)
    print(f"Training Mode: {cfg.TRAINING_MODE}")
    print(f"Cohort: {cfg.TRAIN_COHORT}")
    print(f"Batch Size: {cfg.BATCH_SIZE}")
    print(f"Learning Rate: {cfg.LEARNING_RATE}")
    print(f"Epochs: {cfg.NUM_EPOCHS}")
    print(f"Device: {cfg.DEVICE}")
    print(f"Experiment: {cfg.EXPERIMENT_NAME}")
    print("="*70)
    
    # Check data distributions
    print("\n=== Training Data ===")
    check_data_distribution(cfg.CSV_TRAIN, cfg.COLUMN_NAME)
    
    print("\n=== Validation Data ===")
    check_data_distribution(cfg.CSV_VAL, cfg.COLUMN_NAME)
    
    # Train model
    model, best_val_loss = train()


if __name__ == "__main__":
    main()
