# Setup Guide: Playing Against Your Chess Bot

This guide will help you set up and play against your ML-powered chess bot using the web interface.

## Prerequisites

- Python 3.8+ installed
- Node.js 18+ and npm installed
- Your trained model at `models/chess_evaluator_quick.pth` (or set `MODEL_PATH`)

## Step-by-Step Setup

### Step 1: Set Up Python Environment

1. **Create and activate virtual environment:**
   ```bash
   # From project root
   python -m venv .venv
   
   # On macOS/Linux:
   source .venv/bin/activate
   
   # On Windows:
   .venv\Scripts\activate
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify model exists:**
   ```bash
   # Check if your model file exists
   ls models/chess_evaluator_quick.pth
   
   # If using a different model, set the path:
   export MODEL_PATH=models/your_model.pth
   ```

### Step 2: Set Up Node.js Environment

1. **Navigate to devtools directory:**
   ```bash
   cd devtools
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Create environment file (if needed):**
   ```bash
   # Check if .env.local exists, if not create it
   # The app should work without it, but you can customize ports if needed
   ```

### Step 3: Verify Model Path

The bot will automatically look for the model at:
- Default: `../models/chess_evaluator_quick.pth` (relative to `src/main.py`)
- Or set via environment variable: `MODEL_PATH`

**To use a different model:**
```bash
export MODEL_PATH=/absolute/path/to/your/model.pth
```

### Step 4: Start the Development Server

1. **From the devtools directory, start the Next.js app:**
   ```bash
   cd devtools
   npm run dev
   ```

   This will:
   - Start the Next.js frontend (usually on http://localhost:3000)
   - Automatically start the Python backend (`serve.py`) as a subprocess
   - Enable hot module reloading (changes to your bot code will auto-reload)

2. **You should see output like:**
   ```
   ▲ Next.js 16.0.3
   - Local:        http://localhost:3000
   - Ready in X ms
   ```

### Step 5: Open the Web Interface

1. **Open your browser and navigate to:**
   ```
   http://localhost:3000
   ```

2. **You should see:**
   - A chess board interface
   - Options to play against the bot
   - Analysis features showing move probabilities

### Step 6: Play Against the Bot

1. **Start a new game:**
   - Look for a "New Game" or "Play" button
   - Choose your color (White or Black)
   - The bot will play the opposite color

2. **Make moves:**
   - Click on a piece, then click on the destination square
   - Or drag and drop pieces
   - The bot will automatically respond after your move

3. **View bot analysis:**
   - The interface should show:
     - Move probabilities the bot calculated
     - Evaluation scores
     - Best move suggestions

## Troubleshooting

### Issue: Model Not Found

**Error:** `FileNotFoundError: Model not found at ...`

**Solution:**
1. Verify the model file exists:
   ```bash
   ls -la models/chess_evaluator_quick.pth
   ```

2. Set the correct path:
   ```bash
   export MODEL_PATH=$(pwd)/models/chess_evaluator_quick.pth
   ```

3. Or copy your model to the expected location:
   ```bash
   cp /path/to/your/model.pth models/chess_evaluator_quick.pth
   ```

### Issue: Import Errors

**Error:** `ImportError: attempted relative import with no known parent package`

**Solution:**
- Don't run `main.py` directly! The devtools will run it automatically
- Make sure you're running `npm run dev` from the `devtools` directory
- The Python code is loaded as a module by `serve.py`

### Issue: Port Already in Use

**Error:** `Port 3000 is already in use` or `Port 5058 is already in use`

**Solution:**
1. **For Next.js (port 3000):**
   ```bash
   # Kill the process using port 3000
   lsof -ti:3000 | xargs kill -9
   
   # Or use a different port
   PORT=3001 npm run dev
   ```

2. **For Python backend (port 5058):**
   ```bash
   # Kill the process using port 5058
   lsof -ti:5058 | xargs kill -9
   
   # Or set a different port
   export SERVE_PORT=5059
   ```

### Issue: Model Loading Fails

**Error:** `Error loading model: ...`

**Solution:**
1. Check that PyTorch is installed:
   ```bash
   pip install torch numpy
   ```

2. Verify the model file is not corrupted:
   ```python
   import torch
   checkpoint = torch.load('models/chess_evaluator_quick.pth', map_location='cpu')
   print(checkpoint.keys())
   ```

3. Ensure the model architecture matches (check `model_config` in checkpoint)

### Issue: Bot Makes Random Moves

**Possible causes:**
1. Model failed to load (check console logs)
2. Model path is incorrect
3. Fallback to random moves is being used

**Solution:**
- Check the terminal output for error messages
- Verify model path is correct
- Ensure all dependencies are installed

## Testing the Setup

### Quick Test

1. **Check Python environment:**
   ```bash
   python -c "import torch; import chess; print('Dependencies OK')"
   ```

2. **Check model can be loaded:**
   ```python
   import torch
   model = torch.load('models/chess_evaluator_quick.pth', map_location='cpu')
   print("Model loaded successfully!")
   ```

3. **Check Node.js:**
   ```bash
   cd devtools
   npm --version
   node --version
   ```

## Advanced Configuration

### Custom Model Path

Set environment variable before starting:
```bash
export MODEL_PATH=/custom/path/to/model.pth
cd devtools
npm run dev
```

### Adjust Search Depth

Edit `src/main.py` to change minimax depth:
```python
_searcher = MinimaxSearcher(
    evaluator=_ml_evaluator.evaluate,
    max_depth=4,  # Increase for stronger play (slower)
    enable_quiescence=True,
    quiescence_depth=2
)
```

### Time Management

The bot uses 10% of remaining time per move. This is configured in `src/main.py`:
```python
time_manager = TimeManager(ctx.timeLeft, safety_margin=0.1)
```

## What to Expect

- **First move:** May take a few seconds as the model loads
- **Subsequent moves:** Should be faster (1-5 seconds depending on search depth)
- **Move quality:** The bot uses minimax with ML evaluation, so it should play reasonably well
- **Analysis:** You should see move probabilities and evaluations in the UI

## Next Steps

1. **Improve the bot:**
   - Train a better model with more data
   - Increase search depth (if you have time)
   - Add opening book
   - Implement endgame tablebase

2. **Customize the UI:**
   - Modify `devtools/app/components/AnalysisBoardWrapper/index.tsx`
   - Add more analysis features
   - Customize the board appearance

3. **Deploy:**
   - Once satisfied, deploy to ChessHacks platform
   - See [ChessHacks docs](https://docs.chesshacks.dev/) for deployment instructions

## Quick Start Commands Summary

```bash
# 1. Set up Python
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Verify model exists
ls models/chess_evaluator_quick.pth

# 3. Set up Node.js (if not done)
cd devtools
npm install

# 4. Start the app
npm run dev

# 5. Open browser
# Navigate to http://localhost:3000
```

Enjoy playing against your bot! 🎮♟️

