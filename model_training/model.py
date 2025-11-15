"""
Neural network model for chess position evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessEvaluatorModel(nn.Module):
    """
    CNN-based model for evaluating chess positions.
    
    Architecture:
    - Input: (batch_size, 8, 8, 19) or (batch_size, 19, 8, 8)
    - Convolutional layers with residual connections
    - Fully connected layers
    - Output: Single scalar (evaluation in centipawns)
    """
    
    def __init__(self, input_channels=19, hidden_channels=128, num_residual_blocks=4):
        """
        Args:
            input_channels: Number of input channels (default 19)
            hidden_channels: Number of hidden channels in conv layers
            num_residual_blocks: Number of residual blocks
        """
        super(ChessEvaluatorModel, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_residual_blocks)
        ])
        
        # Final convolution
        self.conv_final = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn_final = nn.BatchNorm2d(hidden_channels)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_channels * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 19, 8, 8) or (batch_size, 8, 8, 19)
            
        Returns:
            Evaluation scores of shape (batch_size, 1)
        """
        # Handle channel-last format (batch_size, 8, 8, 19) -> (batch_size, 19, 8, 8)
        if x.dim() == 4 and x.shape[-1] == self.input_channels:
            x = x.permute(0, 3, 1, 2)
        
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Final convolution
        x = F.relu(self.bn_final(self.conv_final(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        # Output (evaluation in centipawns)
        x = self.fc_out(x)
        
        return x.squeeze(-1)  # Remove last dimension if batch_size > 1


class ResidualBlock(nn.Module):
    """Residual block with two convolutions."""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual  # Residual connection
        x = F.relu(x)
        return x


def create_model(input_channels=19, hidden_channels=128, num_residual_blocks=4, device='cpu'):
    """
    Create and initialize a chess evaluator model.
    
    Args:
        input_channels: Number of input channels
        hidden_channels: Number of hidden channels
        num_residual_blocks: Number of residual blocks
        device: Device to place model on
        
    Returns:
        Initialized model
    """
    model = ChessEvaluatorModel(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        num_residual_blocks=num_residual_blocks
    )
    model = model.to(device)
    
    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    
    return model

