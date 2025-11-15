"""
Dataset loader for Magnus Carlsen Lichess Games Dataset.
"""

import chess
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
import random
from position_encoder import PositionEncoder


class ChessPositionDataset(Dataset):
    """
    Dataset for chess position evaluation.
    Loads positions from the Magnus Carlsen Lichess Games Dataset.
    """
    
    def __init__(
        self,
        dataset_name: str = "luca-g97/Magnus-Carlsen-Lichess-Games-Dataset-FEN",
        split: str = "train",
        max_samples: Optional[int] = None,
        use_game_outcome: bool = True,
        evaluation_range: Tuple[float, float] = (-1000, 1000),
    ):
        """
        Args:
            dataset_name: Hugging Face dataset name
            split: Dataset split ('train', 'validation', 'test')
            max_samples: Maximum number of samples to load (None for all)
            use_game_outcome: If True, use game outcome as label; else use simple evaluation
            evaluation_range: Range for evaluation labels (min, max) in centipawns
        """
        print(f"Loading dataset {dataset_name}...")
        self.dataset = load_dataset(dataset_name, split=split, streaming=False)
        
        # Convert to list and limit samples if needed
        self.data = list(self.dataset)
        if max_samples is not None:
            self.data = self.data[:max_samples]
        
        print(f"Loaded {len(self.data)} samples")
        
        self.use_game_outcome = use_game_outcome
        self.evaluation_range = evaluation_range
        
        # Preprocess data
        self.positions = []
        self.labels = []
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess dataset to extract positions and labels."""
        print("Preprocessing data...")
        
        for idx, sample in enumerate(self.data):
            if idx % 10000 == 0:
                print(f"Processing sample {idx}/{len(self.data)}")
            
            try:
                fen = sample.get('FEN')
                if not fen:
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
        
        print(f"Preprocessed {len(self.positions)} valid positions")
    
    def _get_outcome_label(self, sample: dict, board: chess.Board) -> Optional[float]:
        """
        Get label from game outcome.
        
        Args:
            sample: Dataset sample
            board: Chess board
            
        Returns:
            Evaluation label in centipawns, or None if invalid
        """
        winner = sample.get('Winner')
        loser = sample.get('Loser')
        
        if not winner or not loser:
            return None
        
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

