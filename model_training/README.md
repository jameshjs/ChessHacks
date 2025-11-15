# Chess Model Training

This directory contains code for training a neural network to evaluate chess positions using the Magnus Carlsen Lichess Games Dataset.

## Setup

1. Install training dependencies:
```bash
pip install -r requirements_training.txt
```

2. Ensure you have the base requirements installed:
```bash
pip install -r ../requirements.txt
```

## Dataset

The training uses the [Magnus Carlsen Lichess Games Dataset](https://huggingface.co/datasets/luca-g97/Magnus-Carlsen-Lichess-Games-Dataset-FEN) from Hugging Face, which contains:
- FEN strings for positions
- Game outcomes (winner/loser)
- Move information
- 1.12M samples

**Note:** The dataset will be automatically downloaded from Hugging Face on first use. If the dataset doesn't have a validation split, the training script will automatically create one by splitting the training data (90% train, 10% validation).

## Training

### Quick Training (Minimal Model - For Testing)

Train a very small model quickly for testing purposes:
```bash
python train_quick.py
```

This will:
- Use only 1000 samples (configurable)
- Train for 2 epochs
- Create a minimal model (32 hidden channels, 1 residual block)
- Complete in ~1-5 minutes depending on hardware

**Recommended for:**
- Testing the full pipeline
- Quick iteration during development
- Verifying setup works correctly

**Customize quick training:**
```bash
python train_quick.py --max-samples 500 --epochs 1 --hidden-channels 16
```

### Full Training

Train with default settings:
```bash
python train.py
```

### Custom Training

Train with custom parameters:
```bash
python train.py \
    --max-samples 100000 \
    --batch-size 128 \
    --epochs 20 \
    --lr 0.001 \
    --hidden-channels 256 \
    --num-residual-blocks 6 \
    --output-dir ../models
```

### Arguments

- `--dataset`: Hugging Face dataset name (default: `luca-g97/Magnus-Carlsen-Lichess-Games-Dataset-FEN`)
- `--max-samples`: Maximum number of samples to use (None for all)
- `--batch-size`: Batch size (default: 64)
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 0.001)
- `--hidden-channels`: Number of hidden channels in conv layers (default: 128)
- `--num-residual-blocks`: Number of residual blocks (default: 4)
- `--output-dir`: Output directory for saved models (default: `../models`)
- `--device`: Device to use (`auto`, `cpu`, or `cuda`) (default: `auto`)
- `--train-split`: Training split name (default: `train`)
- `--val-split`: Validation split name (default: `validation`)
- `--use-game-outcome`: Use game outcome as label instead of simple evaluation

## Model Architecture

The model uses a CNN architecture with:
- Input: 8×8×19 tensor (piece positions + castling rights + en passant + side to move + move count)
- Convolutional layers with residual connections
- Fully connected layers
- Output: Single scalar (evaluation in centipawns)

## Position Encoding

Positions are encoded as 8×8×19 tensors:
- Channels 0-11: Piece positions (6 piece types × 2 colors)
- Channels 12-15: Castling rights (white kingside, white queenside, black kingside, black queenside)
- Channel 16: En passant square
- Channel 17: Side to move (0=white, 1=black)
- Channel 18: Move count (normalized)

## Output

Trained models are saved to the `output-dir` (default: `../models/`):

**Quick training:**
- `chess_evaluator_quick.pth`: Best quick model
- `chess_evaluator_quick_final.pth`: Final quick model

**Full training:**
- `chess_evaluator_best.pth`: Best model based on validation loss
- `chess_evaluator_final.pth`: Final model after all epochs
- `checkpoint_epoch_N.pth`: Checkpoint for each epoch

## Loading the Model

To use the trained model in your bot, place it in the `models/` directory and set the `MODEL_PATH` environment variable:

**For quick model:**
```bash
cp models/chess_evaluator_quick.pth models/chess_evaluator.pth
export MODEL_PATH=models/chess_evaluator.pth
```

**For full model:**
```bash
cp models/chess_evaluator_best.pth models/chess_evaluator.pth
export MODEL_PATH=models/chess_evaluator.pth
```

## Notes

- The dataset is large (1.12M samples). Consider using `--max-samples` for faster training during development.
- Training on GPU is recommended for faster training.
- The model uses game outcomes as labels, which is a simplification. For better performance, consider using engine evaluations or more sophisticated labeling.

