"""
ML model evaluator for chess positions.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional
import chess

from .encoder import PositionEncoder


class ChessEvaluatorModel(nn.Module):
    """Model architecture matching the training code."""
    
    def __init__(self, input_channels=19, hidden_channels=128, num_residual_blocks=4):
        super(ChessEvaluatorModel, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_residual_blocks)
        ])
        
        self.conv_final = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn_final = nn.BatchNorm2d(hidden_channels)
        
        self.fc1 = nn.Linear(hidden_channels * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        if x.dim() == 4 and x.shape[-1] == self.input_channels:
            x = x.permute(0, 3, 1, 2)
        
        x = torch.relu(self.bn1(self.conv1(x)))
        for block in self.residual_blocks:
            x = block(x)
        x = torch.relu(self.bn_final(self.conv_final(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc_out(x)
        return x.squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        x = torch.relu(x)
        return x


class MLEvaluator:
    """
    ML model wrapper for position evaluation.
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Args:
            model_path: Path to saved model file
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model()
        self.encoder = PositionEncoder()
        
        print(f"Loaded ML model from {model_path} on {self.device}")
    
    def _load_model(self):
        """Load the model from file."""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model config from checkpoint
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            model = ChessEvaluatorModel(
                input_channels=config.get('input_channels', 19),
                hidden_channels=config.get('hidden_channels', 128),
                num_residual_blocks=config.get('num_residual_blocks', 4)
            )
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Try to infer from model file or use defaults
            model = ChessEvaluatorModel(
                input_channels=19,
                hidden_channels=32,  # Default for quick model
                num_residual_blocks=1
            )
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        return model
    
    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate a chess position.
        
        Args:
            board: chess.Board object
            
        Returns:
            Evaluation in centipawns (positive = white advantage)
        """
        # Handle terminal positions
        if board.is_checkmate():
            return -10000 if board.turn == chess.WHITE else 10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        # Encode position
        position_tensor = self.encoder.encode_position(board)
        
        # Convert to torch tensor (channels first: 19, 8, 8)
        position_tensor = position_tensor.transpose(2, 0, 1)  # (8, 8, 19) -> (19, 8, 8)
        position_tensor = torch.from_numpy(position_tensor).float().unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            evaluation = self.model(position_tensor).item()
        
        # Evaluation is from white's perspective
        # If black to move, we want evaluation from black's perspective for search
        # But we standardize to white's perspective
        return float(evaluation)

