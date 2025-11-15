"""
Dataset loader for Leela Chess Zero Self-Play Games Dataset from Kaggle.
"""

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
from position_encoder import PositionEncoder


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

