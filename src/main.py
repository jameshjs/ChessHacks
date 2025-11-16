from asyncio.constants import DEBUG_STACK_DEPTH
from calendar import c
from re import M
from .utils import chess_manager, GameContext

from chess import Move
import chess
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable
import time

 

    


nn_cache = {}

def evaluate_board(board: chess.Board):

    piece_value= {
        chess.PAWN: 100.0, 
        chess.KNIGHT: 320.0, 
        chess.BISHOP: 330.0, 
        chess.ROOK:500.0, 
        chess.QUEEN: 900.0,
        chess.KING: 0.0}

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

    eval = 0

    # ============================================================
    # 1. MATERIAL + PST
    # ============================================================

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue

        value = piece_value[piece.piece_type]
        pst   = position_table[piece.piece_type]

        if piece.color == chess.WHITE:
            eval += value
            eval += pst[square]               # White uses normal index
        else:
            eval -= value
            eval -= pst[chess.square_mirror(square)]  # Black uses mirrored index

    # ============================================================
    # 2. HANGING PIECES
    # ============================================================

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue

        attackers = board.attackers(not piece.color, square)
        defenders = board.attackers(piece.color, square)

        if attackers and not defenders:
            value = piece_value[piece.piece_type]
            if piece.color == chess.WHITE:
                eval -= value     # white loses a piece
            else:
                eval += value     # black loses a piece

    # ============================================================
    # 3. MOBILITY
    # ============================================================

    original_turn = board.turn

    board.turn = chess.WHITE
    white_mob = len(list(board.legal_moves))

    board.turn = chess.BLACK
    black_mob = len(list(board.legal_moves))

    board.turn = original_turn

    eval += 10 * (white_mob - black_mob)

    return eval



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
    if fen in nn_cache:
        return nn_cache[fen]
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
    result= legal_moves[:top_k]
    nn_cache[fen] = result

    return result

class MinimaxSearcher:

    def __init__(self, evaluator, max_depth=3):
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.nodes_searched = 0

    def search(self, board):
        self.nodes_searched = 0

        maximizing_player = (board.turn == chess.WHITE)

        if maximizing_player:
            best_value = float('-inf')
        else:
            best_value = float('inf')

        best_move = None

        policy_moves = [m for (m, _) in predict_legal_move(model_K, board.fen(), device, top_k=10)]
        tactical_moves = [m for m in board.legal_moves if board.is_capture(m) or board.gives_check(m)]

        moves = policy_moves[:]  

        for m in tactical_moves:
            if m not in moves:
                moves.append(m)



        for move in moves:
            board.push(move)
            value = self._minimax(board,
                                  depth=self.max_depth - 1,
                                  alpha=float('-inf'),
                                  beta=float('inf'),
                                  maximizing=not maximizing_player)
            board.pop()

            if maximizing_player:
                if value > best_value:
                    best_value = value
                    best_move = move
            else:
                if value < best_value:
                    best_value = value
                    best_move = move

        return best_move


    def _minimax(self, board, depth, alpha, beta, maximizing):
        self.nodes_searched += 1
       

        if depth == 0 or board.is_game_over():
            return self.evaluator(board)
        if depth >= 2:
               policy_moves = [m for (m, _) in predict_legal_move(model_K, board.fen(), device, top_k=10)]
               tactical_moves = [m for m in board.legal_moves if board.is_capture(m) or board.gives_check(m)]
               moves_searched = policy_moves[:]  

               for m in tactical_moves:
                    if m not in moves_searched:
                        moves_searched.append(m)



        else:
            moves_searched= list(board.legal_moves)

        if maximizing:
            max_eval = float('-inf')

            for move in moves_searched:
                board.push(move)
                value = self._minimax(board, depth - 1, alpha, beta, False)
                board.pop()

                max_eval = max(max_eval, value)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves_searched:
                board.push(move)
                value = self._minimax(board, depth - 1, alpha, beta, True)
                board.pop()

                min_eval = min(min_eval, value)
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return min_eval




searcher = MinimaxSearcher(evaluator=evaluate_board, max_depth=4)

    



@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """
    Main entrypoint for making moves.
    """
    print("Thinking with Minimax...")

    board = ctx.board
    
    try:
        # Run minimax to get best move

        start = time.time()
        best_move = searcher.search(board)
        end = time.time()

        print(f"Move time: {end - start:.3f} seconds")

        if best_move is None:
            raise ValueError("Minimax returned no move")

        
        return best_move

    except Exception as e:
        print(f"[ERROR] Minimax crashed: {e}")
        import traceback
        traceback.print_exc()

        # Fallback: random move
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            ctx.logProbabilities({})
            raise ValueError("No legal moves")

        move = random.choice(legal_moves)
        ctx.logProbabilities({m: 1.0 / len(legal_moves) for m in legal_moves})
        print(f"Fallback random move: {move.uci()}")
        return move

@chess_manager.reset
def reset_func(ctx: GameContext):
    """
    Reset function called when a new game begins.
    """
    nn_cache.clear()

