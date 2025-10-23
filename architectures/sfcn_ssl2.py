import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class SFCN(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], projection_dim=128):
        super(SFCN, self).__init__()
        self.feature_extractor = nn.Sequential()
        for i in range(len(channel_number)):
            in_channel = 1 if i == 0 else channel_number[i - 1]
            out_channel = channel_number[i]
            is_last = i == len(channel_number) - 1
            self.feature_extractor.add_module(
                f'conv_{i}', 
                self.conv_layer(
                    in_channel, out_channel, 
                    maxpool=not is_last, 
                    kernel_size=(1 if is_last else 3),
                    padding=(0 if is_last else 1)
                )
            )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # Flexible for various input shapes
        self.projection_head = ProjectionHead(input_dim=channel_number[-1], output_dim=projection_dim)

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        layers = [
            nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        ]
        if maxpool:
            layers.insert(2, nn.MaxPool3d(2, stride=maxpool_stride))
        return nn.Sequential(*layers)

    def forward(self, x, return_projection=True):
        x = self.feature_extractor(x)        # → (B, C, D, H, W)
        x = self.avgpool(x).squeeze(-1).squeeze(-1).squeeze(-1)  # → (B, C)

        if return_projection:
            z = self.projection_head(x)      # → (B, projection_dim)
            return x, z                      # return both representations
        else:
            return x                         # just return features (for linear probing, etc.)
