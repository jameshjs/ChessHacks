# Training script for chess position evaluation model.
# Neural network model for chess position evaluation.
# Dataset loader for Leela Chess Zero Self-Play Games Dataset from Kaggle.
#Position encoder for chess boards.
#Converts chess.Board objects to neural network input tensors.

import numpy as np
import chess
from typing import Union


class PositionEncoder:
    """
    Encodes chess positions into 8x8x19 tensors for neural network input.
    
    Channels:
    - 0-11: Piece positions (6 piece types × 2 colors)
    - 12-15: Castling rights (white kingside, white queenside, black kingside, black queenside)
    - 16: En passant square
    - 17: Side to move (0=white, 1=black)
    - 18: Move count (normalized)
    """
    
    PIECE_TO_CHANNEL = {
        chess.PAWN: 0,
        chess.ROOK: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }
    
    def __init__(self):
        self.num_channels = 19
    
    def encode_position(self, board: chess.Board) -> np.ndarray:
        """
        Encode a chess position into a 8x8x19 numpy array.
        
        Args:
            board: chess.Board object
            
        Returns:
            numpy array of shape (8, 8, 19) with dtype float32
        """
        # Initialize tensor
        tensor = np.zeros((8, 8, self.num_channels), dtype=np.float32)
        
        # Encode piece positions (channels 0-11)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                row = 7 - (square // 8)  # Flip vertically (chess uses rank 1-8, array uses 0-7)
                col = square % 8
                
                piece_type = piece.piece_type
                color = piece.color
                
                # Channel = piece_type_index + (6 if black, 0 if white)
                channel = self.PIECE_TO_CHANNEL[piece_type] + (6 if color == chess.BLACK else 0)
                tensor[row, col, channel] = 1.0
        
        # Encode castling rights (channels 12-15)
        channel_idx = 12
        if board.has_kingside_castling_rights(chess.WHITE):
            tensor[:, :, channel_idx] = 1.0
        channel_idx += 1
        
        if board.has_queenside_castling_rights(chess.WHITE):
            tensor[:, :, channel_idx] = 1.0
        channel_idx += 1
        
        if board.has_kingside_castling_rights(chess.BLACK):
            tensor[:, :, channel_idx] = 1.0
        channel_idx += 1
        
        if board.has_queenside_castling_rights(chess.BLACK):
            tensor[:, :, channel_idx] = 1.0
        
        # Encode en passant square (channel 16)
        if board.ep_square is not None:
            ep_row = 7 - (board.ep_square // 8)
            ep_col = board.ep_square % 8
            tensor[ep_row, ep_col, 16] = 1.0
        
        # Encode side to move (channel 17)
        if board.turn == chess.BLACK:
            tensor[:, :, 17] = 1.0
        
        # Encode move count (channel 18) - normalized
        # Half-move clock (50-move rule) normalized to [0, 1]
        tensor[:, :, 18] = min(board.halfmove_clock / 100.0, 1.0)
        
        return tensor
    
    def encode_batch(self, boards: list[chess.Board]) -> np.ndarray:
        """
        Encode multiple positions into a batch tensor.
        
        Args:
            boards: List of chess.Board objects
            
        Returns:
            numpy array of shape (batch_size, 8, 8, 19)
        """
        encoded = [self.encode_position(board) for board in boards]
        return np.stack(encoded, axis=0)
    
    def encode_from_fen(self, fen: str) -> np.ndarray:
        """
        Encode a position from a FEN string.
        
        Args:
            fen: FEN string
            
        Returns:
            numpy array of shape (8, 8, 19)
        """
        board = chess.Board(fen)
        return self.encode_position(board)



import chess
import chess.pgn
import numpy as np
import kagglehub
import pandas as pd
import os
import re
import io
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
import random


class ChessPositionDataset(Dataset):
    """
    Dataset for chess position evaluation.
    Loads positions from the Leela Chess Zero Self-Play Games Dataset on Kaggle.
    """
    
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        kaggle_dataset: str = "anthonytherrien/leela-chess-zero-self-play-chess-games-dataset-3",
        max_samples: Optional[int] = None,
        use_game_outcome: bool = True,
        evaluation_range: Tuple[float, float] = (-1000, 1000),
    ):
        """
        Args:
            dataset_path: Path to already downloaded dataset (if None, will download)
            kaggle_dataset: Kaggle dataset identifier
            max_samples: Maximum number of samples to load (None for all)
            use_game_outcome: If True, use game outcome as label; else use simple evaluation
            evaluation_range: Range for evaluation labels (min, max) in centipawns
        """
        # Download dataset from Kaggle if path not provided
        if dataset_path is None:
            print(f"Downloading dataset from Kaggle: {kaggle_dataset}...")
            dataset_path = kagglehub.dataset_download(kaggle_dataset)
            print(f"Dataset downloaded to: {dataset_path}")
        
        self.dataset_path = Path(dataset_path)
        
        # Load data from Kaggle dataset
        print(f"Loading dataset from {self.dataset_path}...")
        self.data = self._load_kaggle_dataset()
        
        # Note: max_samples is handled during preprocessing, not here
        if isinstance(self.data, list) and len(self.data) > 0:
            if isinstance(self.data[0], (str, Path)):
                print(f"Found {len(self.data)} PGN file(s) to process")
            else:
                print(f"Loaded {len(self.data)} samples")
        
        self.use_game_outcome = use_game_outcome
        self.evaluation_range = evaluation_range
        self.max_samples = max_samples  # Store for use in preprocessing
        
        # Preprocess data
        self.positions = []
        self.labels = []
        self._preprocess_data()
    
    def _load_kaggle_dataset(self):
        """
        Load dataset from Kaggle download path.
        Handles PGN files (primary) and other formats (CSV, parquet, etc.) as fallback.
        """
        # First, try to find PGN files
        pgn_files = list(self.dataset_path.rglob('*.pgn'))
        
        if pgn_files:
            print(f"Found {len(pgn_files)} PGN file(s)")
            # Return list of PGN file paths to process
            return pgn_files
        
        # Fallback to other formats
        data_files = []
        for ext in ['*.csv', '*.parquet', '*.json', '*.jsonl']:
            data_files.extend(list(self.dataset_path.rglob(ext)))
        
        if not data_files:
            # Try looking for common filenames
            common_names = ['train.csv', 'data.csv', 'games.csv', 'positions.csv', 
                          'train.parquet', 'data.parquet', 'games.parquet']
            for name in common_names:
                file_path = self.dataset_path / name
                if file_path.exists():
                    data_files = [file_path]
                    break
        
        if not data_files:
            raise FileNotFoundError(
                f"No data files found in {self.dataset_path}. "
                f"Expected PGN, CSV, parquet, or JSON files."
            )
        
        # Load the first data file found
        data_file = data_files[0]
        print(f"Loading data from: {data_file}")
        
        if data_file.suffix == '.csv':
            df = pd.read_csv(data_file)
            return df.to_dict('records')
        elif data_file.suffix == '.parquet':
            df = pd.read_parquet(data_file)
            return df.to_dict('records')
        elif data_file.suffix in ['.json', '.jsonl']:
            df = pd.read_json(data_file, lines=(data_file.suffix == '.jsonl'))
            return df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {data_file.suffix}")
    
    def _preprocess_data(self):
        """Preprocess dataset to extract positions and labels."""
        print("Preprocessing data...")
        
        # Check if data is PGN files or dict records
        if isinstance(self.data, list) and len(self.data) > 0:
            if isinstance(self.data[0], (str, Path)):
                # It's a list of PGN file paths
                self._process_pgn_files()
                return
            elif isinstance(self.data[0], dict):
                # It's a list of dict records (CSV/parquet format)
                self._process_dict_records()
                return
        
        print(f"Preprocessed {len(self.positions)} valid positions")
    
    def _process_pgn_files(self):
        """Process PGN files to extract positions and evaluations."""
        position_count = 0
        
        for file_idx, pgn_file in enumerate(self.data):
            if file_idx % 10 == 0:
                print(f"Processing PGN file {file_idx + 1}/{len(self.data)}")
            
            try:
                with open(pgn_file, 'r', encoding='utf-8', errors='ignore') as f:
                    while True:
                        game = chess.pgn.read_game(f)
                        if game is None:
                            break
                        
                        # Process this game
                        positions_from_game = self._extract_positions_from_game(game)
                        position_count += len(positions_from_game)
                        
                        # Add to dataset
                        for fen, eval_score in positions_from_game:
                            self.positions.append(fen)
                            self.labels.append(eval_score)
                        
                        # Limit total samples if specified
                        if self.max_samples and len(self.positions) >= self.max_samples:
                            self.positions = self.positions[:self.max_samples]
                            self.labels = self.labels[:self.max_samples]
                            print(f"Reached max_samples limit: {self.max_samples}")
                            return
                            
            except Exception as e:
                print(f"Error processing PGN file {pgn_file}: {e}")
                continue
        
        print(f"Extracted {len(self.positions)} positions from PGN files")
    
    def _extract_positions_from_game(self, game):
        """
        Extract positions and evaluations from a PGN game.
        
        Returns:
            List of (fen, evaluation) tuples
        """
        positions = []
        board = game.board()
        
        # Process each move in the game
        for node in game.mainline():
            move = node.move
            
            # Get evaluation from move comment
            eval_score = self._parse_evaluation_from_comment(node.comment)
            
            # Make the move to get the position after the move
            board.push(move)
            
            # Skip terminal positions
            if board.is_checkmate() or board.is_stalemate():
                board.pop()
                break
            
            # Get FEN of position after move
            fen = board.fen()
            
            # Convert evaluation to centipawns and adjust for side to move
            if eval_score is not None:
                # Evaluation in comment is from perspective of side that just moved
                # We standardize to white's perspective: positive = white advantage
                # Convert from pawns to centipawns
                eval_centipawns = eval_score * 100
                
                # If black just moved (white to move now), the eval was from black's perspective
                # Flip to get white's perspective
                if board.turn == chess.WHITE:  # White to move, so black just moved
                    eval_centipawns = -eval_centipawns
                # If black to move (white just moved), eval is already from white's perspective
                
                # Clip to range
                eval_centipawns = np.clip(eval_centipawns, self.evaluation_range[0], self.evaluation_range[1])
                positions.append((fen, float(eval_centipawns)))
            elif not self.use_game_outcome:
                # Use simple material evaluation if no evaluation available
                eval_score = self._get_simple_evaluation(board)
                positions.append((fen, eval_score))
        
        return positions
    
    def _parse_evaluation_from_comment(self, comment: Optional[str]) -> Optional[float]:
        """
        Parse evaluation from PGN move comment.
        Format: { +0.29/5 0.14s } or { -0.19/4 0.15s }
        
        Returns:
            Evaluation in pawns, or None if not found
        """
        if not comment:
            return None
        
        # Match pattern like "+0.29" or "-0.19" at the start of comment
        # Pattern: optional +/-, digits, decimal point, digits
        match = re.search(r'([+-]?\d+\.?\d*)', comment)
        if match:
            try:
                eval_score = float(match.group(1))
                return eval_score
            except ValueError:
                return None
        
        return None
    
    def _process_dict_records(self):
        """Process dict records (from CSV/parquet files)."""
        for idx, sample in enumerate(self.data):
            if idx % 10000 == 0:
                print(f"Processing sample {idx}/{len(self.data)}")
            
            try:
                # Try different possible FEN column names
                fen = None
                for col_name in ['FEN', 'fen', 'position', 'Position', 'board', 'Board']:
                    if col_name in sample:
                        fen = sample[col_name]
                        break
                
                if not fen or (isinstance(fen, float) and pd.isna(fen)):
                    continue
                
                # Create board from FEN
                try:
                    board = chess.Board(fen)
                except:
                    continue
                
                # Skip terminal positions
                if board.is_checkmate() or board.is_stalemate():
                    continue
                
                # Get label
                if self.use_game_outcome:
                    label = self._get_outcome_label(sample, board)
                else:
                    label = self._get_simple_evaluation(board)
                
                # Skip if label is None
                if label is None:
                    continue
                
                self.positions.append(fen)
                self.labels.append(label)
                
            except Exception as e:
                # Skip problematic samples
                continue
    
    def _get_outcome_label(self, sample: dict, board: chess.Board) -> Optional[float]:
        """
        Get label from game outcome.
        For Leela Chess Zero dataset, tries to extract evaluation or game result.
        
        Args:
            sample: Dataset sample
            board: Chess board
            
        Returns:
            Evaluation label in centipawns, or None if invalid
        """
        # Try to get evaluation directly (Leela datasets often have this)
        eval_cols = ['evaluation', 'Evaluation', 'eval', 'Eval', 'score', 'Score', 'value', 'Value']
        for col in eval_cols:
            if col in sample and not pd.isna(sample[col]):
                try:
                    eval_score = float(sample[col])
                    # Convert to centipawns if needed (Leela uses different scales)
                    # Assuming it's already in a reasonable range, just clip it
                    eval_score = np.clip(eval_score, self.evaluation_range[0], self.evaluation_range[1])
                    return float(eval_score)
                except (ValueError, TypeError):
                    continue
        
        # Try to get game outcome
        winner = sample.get('Winner') or sample.get('winner')
        loser = sample.get('Loser') or sample.get('loser')
        
        if not winner or not loser:
            # If no outcome, use material-based evaluation
            return self._get_simple_evaluation(board)
        
        # Determine if current position's side to move is the winner
        # This is a simplification - we don't know which player is which from the position
        # We'll use a heuristic: if white to move, assume white is more likely to win
        # In practice, you might want to use engine evaluations or more sophisticated labeling
        
        # For now, use a simple heuristic based on material
        material_diff = self._material_difference(board)
        
        # Scale material difference to evaluation range
        # Material difference is roughly in [-9, 9] (queen = 9, etc.)
        # Scale to evaluation range
        eval_score = (material_diff / 9.0) * (self.evaluation_range[1] - self.evaluation_range[0]) / 2
        
        # Add some noise based on game outcome
        if winner == "White" and board.turn == chess.WHITE:
            eval_score += random.uniform(0, 200)
        elif winner == "Black" and board.turn == chess.BLACK:
            eval_score += random.uniform(0, 200)
        else:
            eval_score -= random.uniform(0, 200)
        
        # Clip to range
        eval_score = np.clip(eval_score, self.evaluation_range[0], self.evaluation_range[1])
        
        return float(eval_score)
    
    def _get_simple_evaluation(self, board: chess.Board) -> float:
        """
        Get simple evaluation based on material and position.
        
        Args:
            board: Chess board
            
        Returns:
            Evaluation in centipawns
        """
        # Material values (in pawns)
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0,
        }
        
        # Calculate material difference
        material_diff = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    material_diff += value
                else:
                    material_diff -= value
        
        # Convert to centipawns and scale
        eval_score = material_diff * 100
        
        # Add position bonuses (simplified)
        # Center control, piece activity, etc. could be added here
        
        # Clip to range
        eval_score = np.clip(eval_score, self.evaluation_range[0], self.evaluation_range[1])
        
        return float(eval_score)
    
    def _material_difference(self, board: chess.Board) -> float:
        """Calculate material difference (white - black)."""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }
        
        material_diff = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    material_diff += value
                else:
                    material_diff -= value
        
        return material_diff
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        fen = self.positions[idx]
        label = self.labels[idx]
        
        # Create board and encode
        board = chess.Board(fen)
        encoder = PositionEncoder()
        position_tensor = encoder.encode_position(board)
        
        # Convert to tensor format (channels first: 19, 8, 8)
        position_tensor = position_tensor.transpose(2, 0, 1)  # (8, 8, 19) -> (19, 8, 8)
        
        return {
            'position': position_tensor.astype(np.float32),
            'label': np.float32(label),
            'fen': fen
        }



import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessEvaluatorModel(nn.Module):
    """
    Improved CNN-based model for evaluating chess positions.
    
    Architecture:
    - Input: (batch_size, 8, 8, 19) or (batch_size, 19, 8, 8)
    - Initial convolution with batch norm
    - Multiple improved residual blocks with SE attention
    - Global average pooling for better feature aggregation
    - Fully connected layers with residual connections
    - Output: Single scalar (evaluation in centipawns)
    """
    
    def __init__(self, input_channels=19, hidden_channels=128, num_residual_blocks=6, use_se=True):
        """
        Args:
            input_channels: Number of input channels (default 19)
            hidden_channels: Number of hidden channels in conv layers
            num_residual_blocks: Number of residual blocks
            use_se: Whether to use squeeze-and-excitation blocks
        """
        super(ChessEvaluatorModel, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.use_se = use_se
        
        # Initial convolution with wider kernel for better initial feature extraction
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        
        # Use improved residual blocks with SE attention
        if use_se:
            self.residual_blocks = nn.ModuleList([
                ImprovedResidualBlock(hidden_channels, use_se=True) for _ in range(num_residual_blocks)
            ])
        else:
            self.residual_blocks = nn.ModuleList([
                ResidualBlock(hidden_channels) for _ in range(num_residual_blocks)
            ])
        
        # Additional convolution layer for feature refinement
        self.conv_final = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn_final = nn.BatchNorm2d(hidden_channels)
        
        # Global average pooling for better spatial feature aggregation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers with residual-like connections
        self.fc1 =nn.Linear(hidden_channels, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, 1)
        
        # Batch normalization for FC layers
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.bn_fc3 = nn.BatchNorm1d(128)
        
        # Dropout for regularization (lower dropout for better capacity)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.2)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 19, 8, 8) or (batch_size, 8, 8, 19)
            
        Returns:
            Evaluation scores of shape (batch_size,)
        """
        # Handle channel-last format (batch_size, 8, 8, 19) -> (batch_size, 19, 8, 8)
        if x.dim() == 4 and x.shape[-1] == self.input_channels:
            x = x.permute(0, 3, 1, 2)
        
        # Initial convolution with GELU activation
        x = F.gelu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Final convolution
        x = F.gelu(self.bn_final(self.conv_final(x)))
        
        # Global average pooling for better feature aggregation
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch, hidden_channels)
        
        # Fully connected layers with batch norm and dropout
        x = F.gelu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.gelu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.gelu(self.bn_fc3(self.fc3(x)))
        x = self.dropout3(x)
        
        # Output (evaluation in centipawns)
        x = self.fc_out(x)
        
        return x.squeeze(-1)  # Remove last dimension if batch_size > 1


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class ImprovedResidualBlock(nn.Module):
    """Improved residual block with squeeze-and-excitation and better structure."""
    
    def __init__(self, channels, use_se=True):
        super(ImprovedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels) if use_se else nn.Identity()
        
    def forward(self, x):
        residual = x
        # Use GELU for better gradient flow
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)  # Apply channel attention
        x = x + residual  # Residual connection
        x = F.gelu(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with two convolutions (kept for backward compatibility)."""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = F.gelu(self.bn1(self.conv1(x)))  # Use GELU instead of ReLU
        x = self.bn2(self.conv2(x))
        x = x + residual  # Residual connection
        x = F.gelu(x)
        return x


def create_model(input_channels=19, hidden_channels=128, num_residual_blocks=6, device='cpu', use_se=True):
    """
    Create and initialize an improved chess evaluator model.
    
    Args:
        input_channels: Number of input channels
        hidden_channels: Number of hidden channels
        num_residual_blocks: Number of residual blocks
        device: Device to place model on
        use_se: Whether to use squeeze-and-excitation blocks
        
    Returns:
        Initialized model
    """
    model = ChessEvaluatorModel(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        num_residual_blocks=num_residual_blocks,
        use_se=use_se
    )
    model = model.to(device)
    
    # Improved weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Use He initialization for GELU
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # Xavier initialization for linear layers
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    return model




import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import modal

app = modal.App("chess-model-training")

# Imports will be done inside functions to handle Modal's execution environment

# Create a shared volume for model outputs
volume = modal.Volume.from_name("chess-models", create_if_missing=True)

# Define the image with all dependencies
# All model code is in this file, so we don't need to copy separate files
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install([
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "kagglehub>=0.2.0",
        "pyarrow>=10.0.0",
        "tqdm>=4.65.0",
        "python-chess>=1.999",
    ])
    .env({"KAGGLE_USERNAME": os.getenv("KAGGLE_USERNAME", "")})
    .env({"KAGGLE_KEY": os.getenv("KAGGLE_KEY", "")})
)

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, max_grad_norm=1.0):
    """Train for one epoch with gradient clipping."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        positions = batch['position'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(positions)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'avg_loss': total_loss / num_batches})
    
    return total_loss / num_batches

def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            positions = batch['position'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(positions)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def train_main(
    dataset: str = 'anthonytherrien/leela-chess-zero-self-play-chess-games-dataset-3',
    max_samples: int = None,
    batch_size: int = 64,
    epochs: int = 10,
    lr: float = 0.001,
    hidden_channels: int = 192,  # Increased default capacity
    num_residual_blocks: int = 6,  # More blocks for better depth
    output_dir: str = '../models',
    device: str = 'auto',
    use_game_outcome: bool = False,
    use_se: bool = True,  # Use squeeze-and-excitation by default
    use_huber_loss: bool = True,  # Use Huber loss for better robustness
    weight_decay: float = 1e-4,  # L2 regularization
    max_grad_norm: float = 1.0  # Gradient clipping
):
    """Main training function that can be called directly or via Modal."""
    # Import here to handle Modal's execution environment
    
    # Convert None string to None for max_samples
    if max_samples == "None" or max_samples == "":
        max_samples = None
    elif isinstance(max_samples, str):
        max_samples = int(max_samples)
    
    # Convert boolean string to bool
    if isinstance(use_game_outcome, str):
        use_game_outcome = use_game_outcome.lower() in ('true', '1', 'yes')
    
    # Convert boolean strings to bool
    if isinstance(use_se, str):
        use_se = use_se.lower() in ('true', '1', 'yes')
    if isinstance(use_huber_loss, str):
        use_huber_loss = use_huber_loss.lower() in ('true', '1', 'yes')
    
    # Create args-like object for compatibility
    class Args:
        pass
    args = Args()
    args.dataset = dataset
    args.max_samples = max_samples
    args.batch_size = batch_size
    args.epochs = epochs
    args.lr = lr
    args.hidden_channels = hidden_channels
    args.num_residual_blocks = num_residual_blocks
    args.output_dir = output_dir
    args.device = device
    args.use_game_outcome = use_game_outcome
    args.use_se = use_se
    args.use_huber_loss = use_huber_loss
    args.weight_decay = weight_decay
    args.max_grad_norm = max_grad_norm
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    try:
        full_dataset = ChessPositionDataset(
            kaggle_dataset=args.dataset,
            max_samples=args.max_samples,
            use_game_outcome=args.use_game_outcome,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Split into train and validation
    print("Splitting dataset into train and validation...")
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    print(f"Using {len(train_dataset)} samples for training, {len(val_dataset)} for validation")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    # Create model
    print("Creating model...")
    model = create_model(
        input_channels=19,
        hidden_channels=args.hidden_channels,
        num_residual_blocks=args.num_residual_blocks,
        device=device,
        use_se=args.use_se
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} total parameters ({trainable_params:,} trainable)")
    
    # Loss function: Use Huber loss for better robustness to outliers
    if args.use_huber_loss:
        criterion = nn.HuberLoss(delta=100.0)  # Delta=100 centipawns threshold
        print("Using Huber loss (more robust to outliers)")
    else:
        criterion = nn.MSELoss()
        print("Using MSE loss")
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Improved learning rate scheduler
    # Note: verbose parameter was removed in newer PyTorch versions
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train with gradient clipping
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, 
            max_grad_norm=args.max_grad_norm
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}")
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = output_dir / 'chess_evaluator_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'model_config': {
                    'input_channels': 19,
                    'hidden_channels': args.hidden_channels,
                    'num_residual_blocks': args.num_residual_blocks,
                    'use_se': args.use_se,
                }
            }, model_path)
            print(f"Saved best model to {model_path}")
        
        # Save checkpoint every epoch
        checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss,
            'model_config': {
                'input_channels': 19,
                'hidden_channels': args.hidden_channels,
                'num_residual_blocks': args.num_residual_blocks,
                'use_se': args.use_se,
            }
        }, checkpoint_path)
    
    # Save final model
    final_model_path = output_dir / 'chess_evaluator_final.pth'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'train_loss': train_loss,
        'model_config': {
            'input_channels': 19,
            'hidden_channels': args.hidden_channels,
            'num_residual_blocks': args.num_residual_blocks,
            'use_se': args.use_se,
        }
    }, final_model_path)
    print(f"\nSaved final model to {final_model_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")


@app.function(
    image=image,
    gpu="T4",  # Use T4 GPU (or "A10G" for better performance)
    volumes={"/models": volume},
    timeout=3600 * 4,  # 4 hours timeout
)
def train_cloud(
    dataset: str = 'anthonytherrien/leela-chess-zero-self-play-chess-games-dataset-3',
    max_samples: int = None,
    batch_size: int = 64,
    epochs: int = 10,
    lr: float = 0.001,
    hidden_channels: int = 192,
    num_residual_blocks: int = 6,
    output_dir: str = '/models',
    device: str = 'cuda',
    use_game_outcome: bool = False,
    use_se: bool = True,
    use_huber_loss: bool = True,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0
):
    """
    Modal cloud function for training on GPU.
    This runs on Modal's cloud infrastructure with GPU support.
    All model classes (ChessEvaluatorModel, ChessPositionDataset, etc.) are defined
    in this file, so no additional imports are needed.
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run training
    train_main(
        dataset=dataset,
        max_samples=max_samples,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        hidden_channels=hidden_channels,
        num_residual_blocks=num_residual_blocks,
        output_dir=output_dir,
        device=device,  # Force CUDA on Modal
        use_game_outcome=use_game_outcome,
        use_se=use_se,
        use_huber_loss=use_huber_loss,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm
    )
    
    # Commit volume to persist models (volume is mounted at /models)
    volume.commit()
    print(f"Models saved to volume. Download with: modal volume download chess-models /models")


def main(
    dataset: str = 'anthonytherrien/leela-chess-zero-self-play-chess-games-dataset-3',
    max_samples: int = None,
    batch_size: int = 64,
    epochs: int = 10,
    lr: float = 0.001,
    hidden_channels: int = 192,
    num_residual_blocks: int = 6,
    output_dir: str = '/models',
    device: str = 'cuda',
    use_game_outcome: bool = False,
    use_se: bool = True,
    use_huber_loss: bool = True,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    local: bool = False
):
    """
    Entrypoint for Modal.
    By default, runs on cloud GPU. Set local=True to run locally.
    """
    if local:
        # Run locally for testing
        train_main(
            dataset=dataset,
            max_samples=max_samples,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            hidden_channels=hidden_channels,
            num_residual_blocks=num_residual_blocks,
            output_dir='../models',  # Local path
            device='auto',
            use_game_outcome=use_game_outcome,
            use_se=use_se,
            use_huber_loss=use_huber_loss,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm
        )
    else:
        # Run on Modal cloud with GPU
        print("Running training on Modal cloud with GPU...")
        train_cloud.remote(
            dataset=dataset,
            max_samples=max_samples,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            hidden_channels=hidden_channels,
            num_residual_blocks=num_residual_blocks,
            output_dir=output_dir,
            device=device,
            use_game_outcome=use_game_outcome,
            use_se=use_se,
            use_huber_loss=use_huber_loss,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm
        )


def main_cli():
    """Command-line interface using argparse."""
    parser = argparse.ArgumentParser(description='Train chess position evaluation model')
    parser.add_argument('--dataset', type=str, default='anthonytherrien/leela-chess-zero-self-play-chess-games-dataset-3',
                        help='Kaggle dataset identifier')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to use (None for all)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden-channels', type=int, default=128,
                        help='Number of hidden channels')
    parser.add_argument('--num-residual-blocks', type=int, default=6,
                        help='Number of residual blocks')
    parser.add_argument('--output-dir', type=str, default='../models',
                        help='Output directory for saved models')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--use-game-outcome', action='store_true',
                        help='Use game outcome as label (else use simple evaluation)')
    parser.add_argument('--use-se', action='store_true', default=True,
                        help='Use squeeze-and-excitation blocks')
    parser.add_argument('--no-se', dest='use_se', action='store_false',
                        help='Disable squeeze-and-excitation blocks')
    parser.add_argument('--use-huber-loss', action='store_true', default=True,
                        help='Use Huber loss instead of MSE')
    parser.add_argument('--use-mse-loss', dest='use_huber_loss', action='store_false',
                        help='Use MSE loss instead of Huber')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    
    args = parser.parse_args()
    
    train_main(
        dataset=args.dataset,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden_channels=args.hidden_channels,
        num_residual_blocks=args.num_residual_blocks,
        output_dir=args.output_dir,
        device=args.device,
        use_game_outcome=args.use_game_outcome,
        use_se=args.use_se,
        use_huber_loss=args.use_huber_loss,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm
    )


if __name__ == '__main__':
    main_cli()

