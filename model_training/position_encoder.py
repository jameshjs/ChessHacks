"""
Position encoder for chess boards.
Converts chess.Board objects to neural network input tensors.
"""

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

