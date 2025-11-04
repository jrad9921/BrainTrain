import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, backbone, output_dim=2, dropout=True):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(0.5) if dropout else nn.Identity(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

class ClassifierHeadMLP(nn.Module):
    def __init__(self, backbone, output_dim=2, dropout=True):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Linear(64, output_dim)
                )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

import torch.nn as nn

class ClassifierHeadMLP_(nn.Module):
    def __init__(self, backbone, output_dim):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(64, output_dim)  # assumes final feature dim is 64

    def forward(self, x):
        feats = self.backbone(x, return_projection=False)  # Only use features
        return self.classifier(feats)

class RegressorHeadMLP_(nn.Module):
    def __init__(self, backbone, output_dim=1):
        super().__init__()
        self.backbone = backbone
        self.regressor = nn.Linear(64, output_dim)  # assumes final feature dim is 64

    def forward(self, x):
        # Only use backbone features (no projection head if SSL)
        feats = self.backbone(x, return_projection=False)
        return self.regressor(feats)
