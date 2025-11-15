"""
Minimax search algorithm with alpha-beta pruning.
"""

import chess
from typing import Optional, Callable, Dict
import time

from .time_manager import TimeManager


class MinimaxSearcher:
    """
    Minimax search with alpha-beta pruning and iterative deepening.
    """
    
    def __init__(
        self,
        evaluator: Callable[[chess.Board], float],
        max_depth: int = 3,
        enable_quiescence: bool = True,
        quiescence_depth: int = 2
    ):
        """
        Args:
            evaluator: Function that takes a Board and returns evaluation
            max_depth: Maximum search depth
            enable_quiescence: Whether to use quiescence search
            quiescence_depth: Depth for quiescence search
        """
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.enable_quiescence = enable_quiescence
        self.quiescence_depth = quiescence_depth
        self.time_manager: Optional[TimeManager] = None
        
        # Statistics
        self.nodes_searched = 0
    
    def search(self, board: chess.Board, time_manager: Optional[TimeManager] = None) -> tuple[chess.Move, float, Dict[chess.Move, float]]:
        """
        Search for best move using iterative deepening.
        
        Args:
            board: Current chess position
            time_manager: Time manager (optional)
            
        Returns:
            (best_move, best_eval, move_probs)
        """
        self.time_manager = time_manager
        self.nodes_searched = 0
        
        legal_moves = list(board.generate_legal_moves())
        if not legal_moves:
            return None, -10000 if board.turn == chess.WHITE else 10000, {}
        
        # Order moves (captures first, then others)
        legal_moves = self._order_moves(board, legal_moves)
        
        best_move = legal_moves[0]
        best_eval = -float('inf') if board.turn == chess.WHITE else float('inf')
        
        # Iterative deepening
        for depth in range(1, self.max_depth + 1):
            if time_manager and time_manager.should_stop():
                break
            
            current_best_move = None
            current_best_eval = -float('inf') if board.turn == chess.WHITE else float('inf')
            
            # Search each move
            for move in legal_moves:
                if time_manager and time_manager.should_stop():
                    break
                
                board.push(move)
                eval_score = self._minimax(
                    board, depth - 1,
                    -float('inf'), float('inf'),
                    board.turn == chess.WHITE
                )
                board.pop()
                
                if board.turn == chess.WHITE:
                    if eval_score > current_best_eval:
                        current_best_eval = eval_score
                        current_best_move = move
                else:
                    if eval_score < current_best_eval:
                        current_best_eval = eval_score
                        current_best_move = move
            
            if current_best_move:
                best_move = current_best_move
                best_eval = current_best_eval
        
        # Calculate move probabilities from evaluations
        move_probs = self._calculate_move_probs(board, legal_moves)
        
        return best_move, best_eval, move_probs
    
    def _minimax(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool
    ) -> float:
        """Minimax with alpha-beta pruning."""
        self.nodes_searched += 1
        
        # Check time limit
        if self.time_manager and self.time_manager.should_stop():
            return self.evaluator(board)
        
        # Terminal node
        if board.is_checkmate():
            return -10000 if maximizing else 10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        # Leaf node
        if depth == 0:
            if self.enable_quiescence:
                return self._quiescence_search(board, self.quiescence_depth, alpha, beta, maximizing)
            return self.evaluator(board)
        
        legal_moves = list(board.generate_legal_moves())
        if not legal_moves:
            return 0  # Stalemate
        
        # Order moves
        legal_moves = self._order_moves(board, legal_moves)
        
        if maximizing:
            max_eval = -float('inf')
            for move in legal_moves:
                board.push(move)
                eval_score = self._minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval_score = self._minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval
    
    def _quiescence_search(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool
    ) -> float:
        """Quiescence search for tactical positions."""
        stand_pat = self.evaluator(board)
        
        if depth == 0:
            return stand_pat
        
        if maximizing:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
        else:
            if stand_pat <= alpha:
                return alpha
            beta = min(beta, stand_pat)
        
        # Only search captures and checks
        legal_moves = [m for m in board.generate_legal_moves() 
                      if board.is_capture(m) or board.gives_check(m)]
        
        if not legal_moves:
            return stand_pat
        
        legal_moves = self._order_moves(board, legal_moves)
        
        if maximizing:
            for move in legal_moves:
                board.push(move)
                eval_score = self._quiescence_search(board, depth - 1, alpha, beta, False)
                board.pop()
                if eval_score >= beta:
                    return beta
                alpha = max(alpha, eval_score)
            return alpha
        else:
            for move in legal_moves:
                board.push(move)
                eval_score = self._quiescence_search(board, depth - 1, alpha, beta, True)
                board.pop()
                if eval_score <= alpha:
                    return alpha
                beta = min(beta, eval_score)
            return beta
    
    def _order_moves(self, board: chess.Board, moves: list) -> list:
        """Order moves for better alpha-beta pruning (captures first)."""
        def move_key(move):
            score = 0
            if board.is_capture(move):
                # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
                captured = board.piece_at(move.to_square)
                moving = board.piece_at(move.from_square)
                if captured:
                    piece_values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}
                    score += 1000 + piece_values.get(captured.piece_type, 0) * 10
                    if moving:
                        score -= piece_values.get(moving.piece_type, 0)
            if board.gives_check(move):
                score += 100
            return -score  # Negative for descending order
        
        return sorted(moves, key=move_key)
    
    def _calculate_move_probs(self, board: chess.Board, moves: list) -> Dict[chess.Move, float]:
        """Calculate move probabilities from evaluations."""
        if not moves:
            return {}
        
        # Get evaluations for all moves
        move_evals = {}
        for move in moves:
            board.push(move)
            eval_score = self.evaluator(board)
            # Convert to perspective of side to move
            if board.turn == chess.BLACK:
                eval_score = -eval_score
            move_evals[move] = eval_score
            board.pop()
        
        # Convert to probabilities using softmax
        import numpy as np
        eval_array = np.array([move_evals[m] for m in moves])
        # Normalize to prevent overflow
        eval_array = eval_array - np.max(eval_array)
        probs = np.exp(eval_array) / np.sum(np.exp(eval_array))
        
        return {move: float(prob) for move, prob in zip(moves, probs)}

