import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Conv3DWithLoRA(nn.Module):
    def __init__(self, original_conv, rank, alpha):
        super().__init__()
        self.original_conv = original_conv
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA parameters
        weight_shape = original_conv.weight.shape
        self.out_features = weight_shape[0]
        self.in_features = weight_shape[1] * weight_shape[2] * weight_shape[3] * weight_shape[4]
        
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # Original convolution
        result = self.original_conv(x)
        
        # Compute adapted weight
        # lora_B and lora_A are nn.Parameters, so they should be on the right device
        # But just to be safe, we explicitly move to input device
        lora_weight = (self.lora_B @ self.lora_A).view(self.original_conv.weight.shape) * self.scaling
        
        # Apply LoRA correction
        lora_out = F.conv3d(
            x, 
            lora_weight,
            bias=None,
            stride=self.original_conv.stride,
            padding=self.original_conv.padding,
            dilation=self.original_conv.dilation,
            groups=self.original_conv.groups
        )
        
        return result + lora_out


def apply_lora_to_conv3d(conv_module, rank=4, alpha=1):
    """Apply LoRA to a Conv3d layer."""
    return Conv3DWithLoRA(conv_module, rank, alpha)


def apply_lora_to_model(model, rank=4, alpha=1, target_modules=None):
    """
    Apply LoRA to Conv3d layers in the model.
    For SFCN, we target the convolutional layers in feature_extractor.
    """
    if target_modules is None:
        target_modules = ['conv_']
    
    modified = False
    for name, module in list(model.named_modules()):
        # Check if this is a Conv3d layer in a target module
        if isinstance(module, nn.Conv3d):
            # Check if the module name contains any of our target strings
            if any(target in name for target in target_modules):
                # Get the parent module and child name
                parts = name.split('.')
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                child_name = parts[-1]
                
                # Replace with LoRA version
                print(f"Applying LoRA to: {name}")
                setattr(parent, child_name, apply_lora_to_conv3d(module, rank, alpha))
                modified = True
    
    if not modified:
        print("WARNING: No layers were modified with LoRA!")
        print("Available modules:")
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv3d):
                print(f"  Conv3d layer: {name}")
    
    return model
