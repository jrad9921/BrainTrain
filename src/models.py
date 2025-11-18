
import config as cfg
from architectures import sfcn_cls, sfcn_ssl2, head, lora_layers
import torch
import monai

def create_model(device):
    """Create model based on training mode and task"""
    output_dim = 1 if cfg.TASK == 'regression' else cfg.N_CLASSES
    
    if cfg.TRAINING_MODE == 'sfcn':
        model = sfcn_cls.SFCN(output_dim=output_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), cfg.LEARNING_RATE)
        print(f"Using SFCN for {cfg.TASK}")
    
    elif cfg.TRAINING_MODE == 'dense':
        model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=cfg.N_CHANNELS, out_channels=output_dim).to(device)
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
        
        backbone = lora_layers.apply_lora_to_model(backbone,rank=cfg.LORA_RANK,alpha=cfg.LORA_ALPHA, target_modules=cfg.LORA_TARGET_MODULES)
        
        for name, param in backbone.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
        
        model = head.ClassifierHeadMLP_(backbone, output_dim=output_dim).to(device)
        
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

# ------------------------------
# Model loader
# ------------------------------
def load_model(model_path, device):
    model, _ = create_model(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model
