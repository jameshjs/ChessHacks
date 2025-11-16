from .utils import chess_manager, GameContext
from .minimax import MLEvaluator, MinimaxSearcher, TimeManager
from chess import Move
import os
from pathlib import Path

# Model configuration
# Get model path from environment or use default
_default_path = Path(__file__).parent.parent / 'models' / 'chess_evaluator_final.pth'
MODEL_PATH = os.getenv('MODEL_PATH', str(_default_path))
MODEL_PATH = Path(MODEL_PATH)
if not MODEL_PATH.is_absolute():
    # Resolve relative to project root
    MODEL_PATH = Path(__file__).parent.parent / MODEL_PATH

# Global variables for model and searcher
_ml_evaluator = None
_searcher = None

def initialize_model():
    """Initialize the ML model evaluator."""
    global _ml_evaluator, _searcher
    
    if _ml_evaluator is not None:
        return _ml_evaluator, _searcher
    
    try:
        print(f"Loading model from {MODEL_PATH}...")
        _ml_evaluator = MLEvaluator(str(MODEL_PATH))
        _searcher = MinimaxSearcher(
            evaluator=_ml_evaluator.evaluate,
            max_depth=5,  # Increased depth for better play
            enable_quiescence=True,
            quiescence_depth=4  # Increased quiescence depth
        )
        print("Model loaded successfully!")
        return _ml_evaluator, _searcher
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Initialize on import
try:
    initialize_model()
except Exception as e:
    print(f"Warning: Could not initialize model: {e}")
    print("Bot will not work until model is available")

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """
    Main entrypoint for making moves.
    Uses minimax search with ML evaluation.
    """
    global _ml_evaluator, _searcher
    
    # Ensure model is loaded
    if _ml_evaluator is None or _searcher is None:
        try:
            initialize_model()
        except Exception as e:
            # Fallback to random if model fails
            print(f"Model initialization failed: {e}, using random moves")
            legal_moves = list(ctx.board.generate_legal_moves())
            if not legal_moves:
                ctx.logProbabilities({})
                raise ValueError("No legal moves available")
            move_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
            ctx.logProbabilities(move_probs)
            import random
            return random.choice(legal_moves)
    
    print(f"Thinking... (time left: {ctx.timeLeft}ms)")
    
    # Create time manager
    time_manager = TimeManager(ctx.timeLeft, safety_margin=0.1)
    
    # Search for best move
    try:
        best_move, best_eval, move_probs = _searcher.search(ctx.board, time_manager)
        
        if best_move is None:
            legal_moves = list(ctx.board.generate_legal_moves())
            if not legal_moves:
                ctx.logProbabilities({})
                raise ValueError("No legal moves available")
            # Fallback to first legal move
            best_move = legal_moves[0]
            move_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
        
        # Log probabilities
        ctx.logProbabilities(move_probs)
        
        print(f"Best move: {best_move.uci()}, Evaluation: {best_eval:.2f}")
        return best_move
        
    except Exception as e:
        print(f"Search error: {e}")
        # Fallback to random move
        legal_moves = list(ctx.board.generate_legal_moves())
        if not legal_moves:
            ctx.logProbabilities({})
            raise ValueError("No legal moves available")
        move_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
        ctx.logProbabilities(move_probs)
        import random
        return random.choice(legal_moves)


@chess_manager.reset
def reset_func(ctx: GameContext):
    """
    Reset function called when a new game begins.
    """
    global _searcher
    if _searcher:
        _searcher.nodes_searched = 0
    print("New game started, resetting search state")
