# Minimax Machine Learning Algorithm - Implementation Plan

## Overview

This document outlines the plan for implementing a minimax algorithm enhanced with machine learning for chess position evaluation. The algorithm will combine traditional game tree search (minimax with alpha-beta pruning) with a neural network for position evaluation.

**Note:** The ML model will be trained externally. This plan focuses on loading and integrating a pre-trained model into the chess bot.

## Quick Reference: Model Loading

### Model Requirements
- **Format**: PyTorch (`.pth`/`.pt`), TensorFlow (`.h5`/SavedModel), or ONNX (`.onnx`)
- **Input**: `(8, 8, channels)` numpy array or tensor (typically 19 channels)
- **Output**: Single float (evaluation in centipawns)
- **Location**: Set via `MODEL_PATH` environment variable or default `models/chess_evaluator.pth`

### Quick Setup
```bash
# 1. Place your trained model file
mkdir -p models
cp /path/to/your/model.pth models/chess_evaluator.pth

# 2. Set environment variable (optional)
export MODEL_PATH=models/chess_evaluator.pth
export MODEL_TYPE=pytorch  # or 'tensorflow', 'onnx', 'auto'

# 3. Install dependencies
pip install torch numpy  # or tensorflow, onnxruntime

# 4. Run bot
cd devtools && npm run dev
```

### Model Interface Contract
Before training, coordinate these specifications:
- Input encoding format (channel order, normalization)
- Output evaluation convention (white perspective vs. side-to-move)
- Model file format and save method
- Model architecture class (if using PyTorch state dict)

## Architecture Components

### 1. Core Components

#### 1.1 Minimax Search Engine
- **Purpose**: Search the game tree to find optimal moves
- **Features**:
  - Minimax algorithm with alpha-beta pruning
  - Iterative deepening for time management
  - Move ordering heuristics (captures, checks, killer moves)
  - Transposition table for caching positions
  - Quiescence search for tactical positions

#### 1.2 Machine Learning Evaluator
- **Purpose**: Evaluate chess positions using a neural network
- **Features**:
  - Neural network model (CNN or Transformer-based)
  - Position encoding (board representation)
  - Model loading and inference
  - Batch processing for efficiency

#### 1.3 Position Encoder
- **Purpose**: Convert chess board positions to neural network input
- **Features**:
  - Piece-centric encoding (12 channels: 6 piece types × 2 colors)
  - Additional feature planes (castling rights, en passant, move count, etc.)
  - Normalization and preprocessing

#### 1.4 Time Manager
- **Purpose**: Allocate search time based on remaining time
- **Features**:
  - Dynamic time allocation
  - Safety margins for time management
  - Early termination when time is low

## Implementation Steps

### Phase 1: Foundation Setup

#### Step 1.1: Create Module Structure
```
src/
├── main.py (entry point - modify existing)
├── utils/
│   ├── __init__.py
│   └── decorator.py
├── minimax/
│   ├── __init__.py
│   ├── search.py (minimax algorithm)
│   ├── evaluator.py (ML model wrapper)
│   ├── encoder.py (position encoding)
│   └── time_manager.py (time allocation)
└── models/
    └── (model files or download logic)
```

#### Step 1.2: Dependencies
Add to `requirements.txt` based on model format:

**For PyTorch models:**
- `torch` (PyTorch library)
- `numpy` (for array operations)

**For TensorFlow models:**
- `tensorflow` (TensorFlow library)
- `numpy` (for array operations)

**For ONNX models:**
- `onnxruntime` (ONNX Runtime for inference)
- `numpy` (for array operations)

**Common (always needed):**
- `numpy` (array operations and position encoding)
- `chess` (already in requirements.txt)

### Phase 2: Position Encoding

#### Step 2.1: Board Representation
- Implement function to convert `chess.Board` to tensor/array
- Use piece-centric encoding:
  - 8×8×12 channels (6 piece types × 2 colors)
  - Each channel represents positions of one piece type for one color
  - Additional channels for:
    - Castling rights (4 channels)
    - En passant square (1 channel)
    - Side to move (1 channel)
    - Move count/ply (1 channel)
    - Total: 8×8×19 input tensor

#### Step 2.2: Encoding Function
```python
def encode_position(board: Board) -> np.ndarray:
    # Convert board to 8×8×19 tensor
    # Return normalized array ready for model input
```

### Phase 3: Machine Learning Model Loading & Integration

#### Step 3.1: Model Format Considerations

**Supported Model Formats:**
- **PyTorch** (`.pth`, `.pt`): Most common, flexible
- **TensorFlow/Keras** (`.h5`, `.pb`, SavedModel): Alternative option
- **ONNX** (`.onnx`): Cross-platform, optimized for inference
- **TorchScript** (`.pt`): PyTorch's optimized format
- **Pickle** (`.pkl`): For custom model classes

**Model Location Options:**
1. Local file path: `models/chess_evaluator.pth`
2. Relative to project root: `./models/model.pth`
3. Environment variable: `MODEL_PATH` env var
4. Configuration file: Specify in config
5. HuggingFace Hub: Load from `huggingface_hub`

#### Step 3.2: Model Interface Requirements

**Expected Model Interface:**
The model should accept input and return evaluation scores. Define clear interface:

```python
# Model should accept:
# Input: numpy array or torch tensor of shape (batch_size, 8, 8, channels)
#        or (8, 8, channels) for single position
# Output: Single scalar (float) or array of scalars for batch
#         Evaluation in centipawns (positive = white advantage, negative = black advantage)
```

**Model Wrapper Class:**
```python
class MLEvaluator:
    def __init__(self, model_path: str, model_type: str = 'auto'):
        """
        model_type: 'pytorch', 'tensorflow', 'onnx', 'auto'
        """
        self.model = self._load_model(model_path, model_type)
        self.device = self._get_device()  # CPU or GPU
        self.model.eval()  # Set to evaluation mode
    
    def _load_model(self, path: str, model_type: str):
        # Detect format and load accordingly
        pass
    
    def evaluate(self, board: Board) -> float:
        # Single position evaluation
        pass
    
    def evaluate_batch(self, boards: list[Board]) -> list[float]:
        # Batch evaluation for efficiency
        pass
```

#### Step 3.3: Model Loading Implementation

**Loading Strategy in `main.py`:**
```python
# At module level (top of main.py, runs once on import)
import os
from pathlib import Path

# Determine model path
MODEL_PATH = os.getenv('MODEL_PATH', 'models/chess_evaluator.pth')
MODEL_PATH = Path(MODEL_PATH)

# Validate model exists
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

# Load model (lazy loading option)
_ml_evaluator = None

def get_evaluator():
    global _ml_evaluator
    if _ml_evaluator is None:
        from .minimax.evaluator import MLEvaluator
        _ml_evaluator = MLEvaluator(str(MODEL_PATH))
    return _ml_evaluator
```

**PyTorch Model Loading:**
```python
import torch

def load_pytorch_model(path: str):
    # Option 1: Model class + state dict
    from models.chess_model import ChessEvaluator  # Import your model class
    model = ChessEvaluator()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model
    
    # Option 2: Full model save (torch.save(model, path))
    # model = torch.load(path, map_location='cpu')
    # model.eval()
    # return model
    
    # Option 3: TorchScript (most optimized)
    # model = torch.jit.load(path, map_location='cpu')
    # model.eval()
    # return model
```

**TensorFlow Model Loading:**
```python
import tensorflow as tf

def load_tensorflow_model(path: str):
    # Option 1: SavedModel format
    model = tf.saved_model.load(path)
    return model
    
    # Option 2: Keras .h5 format
    # model = tf.keras.models.load_model(path)
    # return model
```

**ONNX Model Loading:**
```python
import onnxruntime as ort

def load_onnx_model(path: str):
    # ONNX Runtime for inference
    sess = ort.InferenceSession(
        path,
        providers=['CPUExecutionProvider']  # or 'CUDAExecutionProvider'
    )
    return sess

def evaluate_onnx(sess, input_array):
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: input_array})
    return output[0]
```

#### Step 3.4: Model Path Configuration

**Configuration Options:**
1. **Environment Variable** (Recommended for deployment):
   ```bash
   export MODEL_PATH=/path/to/model.pth
   ```

2. **Configuration File**:
   ```python
   # config.py
   MODEL_CONFIG = {
       'path': 'models/chess_evaluator.pth',
       'type': 'pytorch',  # 'pytorch', 'tensorflow', 'onnx', 'auto'
       'device': 'auto',  # 'cpu', 'cuda', 'auto'
       'batch_size': 32,
   }
   ```

3. **Command Line Argument** (if using CLI):
   ```python
   import argparse
   parser.add_argument('--model-path', default='models/model.pth')
   ```

4. **Hardcoded Path** (for development):
   ```python
   MODEL_PATH = 'models/chess_evaluator.pth'
   ```

#### Step 3.5: Model Validation & Error Handling

**Model Validation:**
```python
def validate_model(model, encoder):
    """Test model with sample position to ensure it works"""
    from chess import Board
    
    test_board = Board()
    test_input = encoder.encode_position(test_board)
    
    try:
        output = model.evaluate(test_board)  # or model inference
        assert isinstance(output, (int, float)), "Model must return numeric evaluation"
        return True
    except Exception as e:
        raise ValueError(f"Model validation failed: {e}")
```

**Error Handling:**
```python
class MLEvaluator:
    def __init__(self, model_path: str):
        try:
            self.model = self._load_model(model_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        
        # Fallback evaluator for errors
        self.fallback_evaluator = SimpleEvaluator()
        self.use_fallback = False
    
    def evaluate(self, board: Board) -> float:
        if self.use_fallback:
            return self.fallback_evaluator.evaluate(board)
        
        try:
            # Model inference
            return self._model_evaluate(board)
        except Exception as e:
            print(f"Model evaluation error: {e}, using fallback")
            self.use_fallback = True
            return self.fallback_evaluator.evaluate(board)
```

#### Step 3.6: Model Initialization in main.py

**Module-Level Initialization:**
```python
# In main.py, at the top (after imports)
import os
from pathlib import Path

# Model configuration
MODEL_PATH = os.getenv('MODEL_PATH', 'models/chess_evaluator.pth')
MODEL_TYPE = os.getenv('MODEL_TYPE', 'auto')  # 'pytorch', 'tensorflow', 'onnx', 'auto'

# Lazy loading - only load when first needed
_ml_evaluator = None
_model_loaded = False

def initialize_model():
    """Initialize the ML model evaluator"""
    global _ml_evaluator, _model_loaded
    
    if _model_loaded:
        return _ml_evaluator
    
    try:
        from .minimax.evaluator import MLEvaluator
        model_path = Path(MODEL_PATH)
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                f"Set MODEL_PATH environment variable or place model at default location."
            )
        
        _ml_evaluator = MLEvaluator(str(model_path), model_type=MODEL_TYPE)
        _model_loaded = True
        print(f"Successfully loaded model from {MODEL_PATH}")
        
        return _ml_evaluator
    except Exception as e:
        print(f"Warning: Failed to load ML model: {e}")
        print("Falling back to simple evaluation function")
        _model_loaded = True  # Prevent repeated attempts
        return None
```

#### Step 3.7: Model Input/Output Specification

**Input Format Agreement:**
Coordinate with model training to ensure input format matches:

```python
# Expected input shape: (batch_size, 8, 8, channels)
# Channels typically include:
# - 12 piece channels (6 types × 2 colors)
# - 4 castling rights
# - 1 en passant
# - 1 side to move
# - 1 move count (optional)
# Total: 19 channels (or as agreed with training)

# Input normalization:
# - Piece positions: 0 or 1 (binary)
# - Castling rights: 0 or 1
# - En passant: 0 or 1
# - Side to move: 0 (white) or 1 (black)
# - Move count: normalized (e.g., /100)
```

**Output Format:**
```python
# Model should return:
# - Single position: float (evaluation in centipawns)
# - Batch: numpy array or list of floats
# - Evaluation range: typically -10000 to +10000 (or as trained)
# - Positive = advantage for side to move (or white, depending on convention)
```

#### Step 3.8: Evaluation Function Implementation

```python
def evaluate_position(board: Board, model, encoder) -> float:
    """
    Evaluate a chess position using the ML model.
    
    Args:
        board: chess.Board object
        model: Loaded ML model
        encoder: Position encoder instance
    
    Returns:
        float: Position evaluation in centipawns
    """
    # Handle terminal positions first
    if board.is_checkmate():
        return -10000 if board.turn else 10000  # Negative if current player is mated
    if board.is_stalemate() or board.is_insufficient_material():
        return 0  # Draw
    
    # Encode position
    position_tensor = encoder.encode_position(board)
    
    # Run model inference
    with torch.no_grad():  # For PyTorch
        evaluation = model(position_tensor)
    
    # Ensure evaluation is from current player's perspective
    if not board.turn:  # Black to move
        evaluation = -evaluation
    
    return float(evaluation)
```

### Phase 4: Minimax Algorithm

#### Step 4.1: Basic Minimax
- Implement recursive minimax function
- Handle terminal positions (checkmate, stalemate, draw)
- Return best move and evaluation

#### Step 4.2: Alpha-Beta Pruning
- Add alpha-beta pruning to reduce search space
- Implement move ordering:
  1. Captures (MVV-LVA: Most Valuable Victim - Least Valuable Attacker)
  2. Checks
  3. Killer moves (moves that caused beta cutoffs)
  4. History heuristic
  5. Remaining moves

#### Step 4.3: Iterative Deepening
- Start with depth 1, increment until time runs out
- Use previous iteration's best move for move ordering
- Return best move from deepest completed search

#### Step 4.4: Transposition Table
- Cache evaluated positions
- Store: position hash, depth, evaluation, best move, node type (exact/cut/all)
- Use Zobrist hashing for position keys

#### Step 4.5: Quiescence Search
- After reaching depth limit, search captures and checks
- Prevents horizon effect
- Continue until quiet position

### Phase 5: Integration with Framework

#### Step 5.1: Modify `test_func` (Entrypoint)
- Initialize minimax searcher with ML evaluator
- Call minimax search with current board
- Extract move probabilities from search tree
- Log probabilities using `ctx.logProbabilities()`
- Return best move

#### Step 5.2: Time Management
- Use `ctx.timeLeft` to allocate search time
- Reserve time for move execution
- Implement time-checking in search loop

#### Step 5.3: Move Probability Calculation
- Extract evaluation scores from search tree
- Convert to probabilities (softmax or similar)
- Map to all legal moves (some may not be in search tree)

#### Step 5.4: Reset Function
- Clear transposition table
- Reset killer moves and history heuristics
- Reset any cached evaluations

### Phase 6: Optimization

#### Step 6.1: Performance Optimization
- Batch model evaluations when possible
- Cache model evaluations in transposition table
- Optimize position encoding (use bitboards if needed)
- Profile and optimize hot paths

#### Step 6.2: Search Optimization
- Implement null-move pruning
- Implement late move reductions (LMR)
- Implement futility pruning
- Implement razoring

#### Step 6.3: Model Optimization
- Quantize model for faster inference
- Use ONNX or TensorRT for deployment
- Optimize batch size for inference

## Detailed Implementation Specifications

### Minimax Search Function Signature
```python
def minimax_search(
    board: Board,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    evaluator: Callable,
    transposition_table: dict,
    killer_moves: list,
    history: dict,
    time_limit: float
) -> tuple[float, Move | None]:
    """
    Returns: (evaluation, best_move)
    """
```

### ML Evaluator Interface
```python
class MLEvaluator:
    def __init__(self, model_path: str):
        # Load model
        pass
    
    def evaluate(self, board: Board) -> float:
        # Return position evaluation
        pass
    
    def evaluate_batch(self, boards: list[Board]) -> list[float]:
        # Batch evaluation for efficiency
        pass
```

### Time Manager Interface
```python
class TimeManager:
    def __init__(self, time_left_ms: int):
        self.time_left = time_left_ms
        self.start_time = time.time()
    
    def allocate_time(self) -> float:
        # Return allocated time for this move
    
    def should_stop(self) -> bool:
        # Check if search should stop
```

## Testing Strategy

### Unit Tests
1. Test position encoding (verify correct representation)
2. Test minimax on simple positions (mate in 1, 2, etc.)
3. Test alpha-beta pruning correctness
4. Test time management

### Integration Tests
1. Test full search on known positions
2. Test move probability logging
3. Test reset functionality
4. Test with various time constraints

### Performance Tests
1. Measure search speed (nodes per second)
2. Measure model inference time
3. Profile memory usage
4. Test with different search depths

## Model Training (External)

**Note:** Model training will be done externally. The following information is for coordination with the training process.

### Model Input/Output Contract

**Critical:** The model training must produce a model that matches these specifications:

1. **Input Format:**
   - Shape: `(batch_size, 8, 8, channels)` or `(8, 8, channels)` for single position
   - Channels: As agreed (typically 19: 12 piece channels + 4 castling + 1 en passant + 1 side to move + 1 move count)
   - Data type: `float32` numpy array or torch tensor
   - Normalization: Binary (0/1) for piece positions, normalized for continuous features

2. **Output Format:**
   - Single position: `float` (evaluation in centipawns)
   - Batch: Array/list of floats
   - Evaluation convention: Positive = advantage for white (or side to move, as agreed)
   - Range: Typically -10000 to +10000

3. **Model File Format:**
   - Preferred: PyTorch (`.pth` or `.pt`)
   - Alternative: ONNX (`.onnx`) for cross-platform
   - Include model architecture class if using state dict format

### Coordination Checklist

Before training, coordinate:
- [ ] Input encoding format (channel order, normalization)
- [ ] Output evaluation convention (white perspective vs. side-to-move)
- [ ] Model file format and save method
- [ ] Model architecture (if loading state dict, need class definition)
- [ ] Expected inference speed/performance
- [ ] Batch processing support (if available)

## Configuration

### Hyperparameters to Tune
- Search depth limits
- Time allocation strategy
- Alpha-beta window sizes
- Model confidence thresholds
- Move ordering weights

### Configuration File (Optional)
```python
CONFIG = {
    'max_depth': 5,
    'time_allocation_factor': 0.1,  # Use 10% of time per move
    'enable_quiescence': True,
    'quiescence_depth': 3,
    'transposition_table_size': 1000000,
    'model_path': 'models/chess_evaluator.pth',
    'batch_size': 32,
}
```

## Risk Mitigation

### Potential Issues
1. **Model loading failures**: Fallback to simple evaluation
2. **Timeouts**: Always return a move, even if search incomplete
3. **Memory issues**: Limit transposition table size
4. **Slow inference**: Cache evaluations, use batch processing
5. **Illegal moves**: Always validate moves before returning

### Error Handling
- Try-except blocks around model inference
- Validation of all moves
- Graceful degradation (fallback to simpler evaluation)

## Future Enhancements

1. **Monte Carlo Tree Search (MCTS)**: Alternative to minimax
2. **Self-play training**: Improve model through self-play
3. **Opening book**: Precomputed opening moves
4. **Endgame tablebase**: Perfect play in endgames
5. **Multi-threading**: Parallel search
6. **GPU acceleration**: Faster model inference

## Implementation Priority

### High Priority (MVP)
1. **Model loading infrastructure** (Phase 3.2-3.6)
   - Model loader for chosen format (PyTorch/TensorFlow/ONNX)
   - Model path configuration (environment variable)
   - Error handling and fallback evaluator
   - Model validation
2. **Position encoding** (Phase 2)
   - Board to tensor conversion
   - Match model's expected input format
3. **Basic minimax with alpha-beta** (Phase 4.1-4.2)
   - Core search algorithm
   - Integration with ML evaluator
4. **Integration with framework** (Phase 5)
   - Modify `test_func` entrypoint
   - Time management using `ctx.timeLeft`
   - Move probability logging
5. **Basic time management** (Phase 1.4, 5.2)

### Medium Priority
1. Transposition table
2. Move ordering heuristics
3. Iterative deepening
4. Quiescence search
5. Batch model evaluation

### Low Priority (Optimizations)
1. Advanced pruning techniques
2. Model quantization
3. Multi-threading
4. Opening book
5. Endgame tablebase

## Notes

### Implementation Strategy
- **Start simple**: Get basic minimax working first, then integrate ML model
- **Test incrementally**: Test each component independently (encoder, evaluator, search)
- **Model loading**: Test model loading with a dummy/sample model first
- **Profile early**: Identify bottlenecks before optimizing
- **Time management**: Bot must always return a move in time - critical for competition

### Model Integration Notes
- **Model path**: Use environment variable `MODEL_PATH` for flexibility
- **Lazy loading**: Load model only when first needed (not at import time)
- **Error handling**: Always have fallback evaluator if model fails
- **Input validation**: Ensure position encoding matches model's expected format
- **Output validation**: Verify model returns expected evaluation format
- **Coordinate with training**: Ensure input/output format matches between training and inference

### File Structure for Models
```
project_root/
├── models/
│   ├── chess_evaluator.pth  (or .onnx, .h5, etc.)
│   └── README.md  (document model format, input/output spec)
├── src/
│   └── main.py
└── requirements.txt
```

### Environment Setup
```bash
# Set model path
export MODEL_PATH=models/chess_evaluator.pth
export MODEL_TYPE=pytorch  # or 'tensorflow', 'onnx', 'auto'

# Run bot
cd devtools && npm run dev
```

