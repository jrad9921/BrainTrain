import config as cfg
import torch
import torch.nn as nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def get_criterion(device, train_labels=None):
    """Get loss function based on task"""
    if cfg.TASK == 'classification':
        if train_labels is not None:
            classes = np.unique(train_labels)   # Key change for multiclass
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=train_labels
            )
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
            print(f"Class weights: {class_weights}")
            criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
        else:
            criterion = nn.CrossEntropyLoss().to(device)

    elif cfg.TASK == 'regression':
        criterion = nn.MSELoss().to(device)
        print("Using MSE Loss for regression")

    else:
        raise ValueError(f"Invalid TASK: {cfg.TASK}")
    
    return criterion
