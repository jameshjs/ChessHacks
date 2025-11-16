from calendar import c
from .utils import chess_manager, GameContext

from chess import Move
import chess
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable
 



def evaluate_board(board: chess.Board):

    piece_value= {
        chess.PAWN: 100.0, 
        chess.KNIGHT: 320.0, 
        chess.BISHOP: 330.0, 
        chess.ROOK:500.0, 
        chess.QUEEN: 900.0,
        chess.King: 0.0}

    PAWN_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5,  5,  5,  5,  5,  5,  5,  5,
    1,  1,  2,  3,  3,  2,  1,  1,
    0,  0,  1,  2,  2,  1,  0,  0,
    0,  0,  0,  2,  2,  0,  0,  0,
    0,  0, -1,  0,  0, -1,  0,  0,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
    ]

    KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
    ]

    BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
    ]

    ROOK_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
    ]

    QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
    ]

    KING_TABLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
    ]

    position_table ={
        chess.PAWN: PAWN_TABLE,
        chess.KNIGHT: KNIGHT_TABLE,
        chess.BISHOP: BISHOP_TABLE,
        chess.ROOK: ROOK_TABLE,
        chess.QUEEN: QUEEN_TABLE,
        chess.KING: KING_TABLE}

    eval=0


    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if(piece):
            
            if(piece.color==chess.WHITE):
                eval+=piece_value[piece.piece_type]
                eval+=position_table[piece.piece_type][square]
            else:
                eval-=piece_value[piece.piece_type]
                eval+=position_table[piece.piece_type][square]




    return eval
            
class minimax_searcher:

    def __init__(self, evaluator: Callable[[chess.Board], float], max_depth: int = 3):
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.nodes_searched = 0         # Initialize counter
    def search(self, board: chess.Board) :
        self.nodes_searched=0









######################################################






def board_to_tensor(board : chess.Board): 
    #
    piece_encode= {
        chess.PAWN: 0, 
        chess.KNIGHT: 1, 
        chess.BISHOP: 2, 
        chess.ROOK:3, 
        chess.QUEEN: 4,
        chess.KING: 5 
        }

    tensor=np.zeros ( (13, 8, 8), dtype=np.float32) 
    #creates a 12 x 8 x 8 tensor. the 8x8 represents the board, 
    #while the array of 12 is acts as a one-hot encoding for the pieces

    #iterates over all squares on a chessboard
    if board.turn ==chess.WHITE:
        tensor[12, : ,:] =1.0
    for square in chess.SQUARES:
        piece=board.piece_at(square)
        if piece:
            row = chess.square_rank(square)
            col= chess.square_file(square)
            if piece.color == chess.WHITE:
                piece_value= piece_encode[piece.piece_type]
                tensor[piece_value, row, col] =1.0
                
            else: 
                piece_value= piece_encode[piece.piece_type] + 6
                tensor[piece_value, row, col]=1.0
            
    return tensor
class CnnModel (nn.Module):
    def __init__(self):
        super().__init__()  # Initialize parent class
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(13, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), #activation
            
            # Conv block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),#activation
            
            # Conv block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),#activation
            )
       
        self.classifier = nn.Sequential(
            nn.Flatten(),                      # Step 1: (256, 8, 8) → (16384,)
            nn.Linear(256 * 8 * 8, 1024),     # Step 2: 16384 → 1024 neurons
            nn.ReLU(),                         # Step 3: Activation
            nn.Dropout(0.3),                   # Step 4: Regularization (prevent overfitting)
            nn.Linear(1024, 4096)             # Step 5: 1024 → 4096 (OUTPUT LAYER)
            )
    
    def forward(self, x):
        x = self.features(x)      # Step 1: go through feature layer and detect features
        x = self.classifier(x)    # Step 2: Make prediction
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model instance
model_K = CnnModel().to(device)

# Load saved weights
model_path = Path(__file__).parent.parent / 'models' / 'best_model.pth'
model_K.load_state_dict(torch.load(str(model_path), map_location=device))
# Set to evaluation mode
model_K.eval()
def predict_legal_move(model, fen, device, top_k=5):
    board= chess.Board(fen)
    board_tensor=torch.FloatTensor(board_to_tensor(board))
    board_tensor = board_tensor.unsqueeze(0).to(device)  # shape=(1,13,8,8)
    with torch.no_grad():
        output = model(board_tensor)
        probs = torch.softmax(output, dim=1)[0]
    legal_moves = []
    for move in board.legal_moves:
        idx = move.from_square * 64 + move.to_square
        legal_moves.append((move, probs[idx].item()))

    legal_moves.sort(key=lambda x: x[1], reverse=True)
    return legal_moves[:5]



@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """
    Main entrypoint for making moves.
    """
    print("Making move prediction...")
    
    try:
        # Get top 5 legal moves with probabilities
        top_moves = predict_legal_move(model_K, ctx.board.fen(), device, top_k=5)
        
        if not top_moves:
            raise ValueError("No legal moves available")
        
        # Convert to dict for logProbabilities
        move_probs = {move: prob for move, prob in top_moves}
        
        # Log probabilities for UI
        ctx.logProbabilities(move_probs)
        
        # Return best move (just the Move object, not the tuple)
        best_move = top_moves[0][0]  # ← FIXED: Extract move from tuple
        print(f"Selected move: {best_move.uci()} (confidence: {top_moves[0][1]:.4f})")
        
        return best_move


    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to random move
        import random
        legal_moves = list(ctx.board.generate_legal_moves())
        if not legal_moves:
            ctx.logProbabilities({})
            raise ValueError("No legal moves available")
        
        move = random.choice(legal_moves)
        ctx.logProbabilities({m: 1.0/len(legal_moves) for m in legal_moves})
        return move

@chess_manager.reset
def reset_func(ctx: GameContext):
    """
    Reset function called when a new game begins.
    """
    global _searcher
    if _searcher:
        _searcher.nodes_searched = 0
    print("New game started, resetting search state")
