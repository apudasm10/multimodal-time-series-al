import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x): return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TwoStreamTCN(nn.Module):
    def __init__(self, num_classes):
        super(TwoStreamTCN, self).__init__()
        
        # --- Stream A: Motion (IMU + Mag) ---
        # Inputs: Acc(3) + Gyr(3) + Mag(3) = 9 Channels
        self.motion_branch = nn.Sequential(
            nn.BatchNorm1d(9), # Normalize raw inputs
            TemporalBlock(9, 32, kernel_size=3, stride=1, dilation=1, padding=2),
            TemporalBlock(32, 64, kernel_size=3, stride=1, dilation=2, padding=4),
            TemporalBlock(64, 64, kernel_size=3, stride=1, dilation=4, padding=8),
            nn.AdaptiveAvgPool1d(1) # Output: (Batch, 64, 1)
        )
        
        # --- Stream B: Audio (Mic) ---
        # Audio is longer, so we might want more pooling or larger strides if needed.
        self.audio_branch = nn.Sequential(
            nn.BatchNorm1d(1), 
            TemporalBlock(1, 16, kernel_size=3, stride=1, dilation=1, padding=2),
            nn.MaxPool1d(2), # Downsample audio to reduce computation
            TemporalBlock(16, 32, kernel_size=3, stride=1, dilation=2, padding=4),
            nn.MaxPool1d(2),
            TemporalBlock(32, 64, kernel_size=3, stride=1, dilation=4, padding=8),
            nn.AdaptiveAvgPool1d(1) # Output: (Batch, 64, 1)
        )
        
        # --- Fusion & Classifier ---
        # 64 (Motion) + 64 (Audio) = 128
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_acc, x_gyr, x_mag, x_mic):

        x_motion = torch.cat([x_acc, x_gyr, x_mag], dim=1)
        
        # 3. Forward Passes
        feat_motion = self.motion_branch(x_motion).squeeze(-1) # (N, 64)
        feat_audio = self.audio_branch(x_mic).squeeze(-1)      # (N, 64)
        
        # 4. Fusion
        combined = torch.cat([feat_motion, feat_audio], dim=1) # (N, 128)
        
        return self.classifier(combined)
    

class TCN(nn.Module):
    def __init__(self, num_classes, output_mode='logits'):
        super(TCN, self).__init__()
        self.output_mode = output_mode
        
        self.motion_branch = nn.Sequential(
            nn.BatchNorm1d(9),
            TemporalBlock(9, 32, kernel_size=3, stride=1, dilation=1, padding=2),
            TemporalBlock(32, 64, kernel_size=3, stride=1, dilation=2, padding=4),
            TemporalBlock(64, 128, kernel_size=3, stride=1, dilation=4, padding=8),
            TemporalBlock(128, 128, kernel_size=3, stride=1, dilation=8, padding=16),
            nn.AdaptiveAvgPool1d(1)
        )
    
        self.embedding = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, num_classes)
        )

    def forward(self, X):
        x_acc, x_gyr, x_mag = X

        x_motion = torch.cat([x_acc, x_gyr, x_mag], dim=1)
        
        feat_motion = self.motion_branch(x_motion).squeeze(-1)
        embed = self.embedding(feat_motion)
        logits = self.classifier(embed)

        if self.output_mode == "embedding":
            return embed, logits
        else:
            return logits