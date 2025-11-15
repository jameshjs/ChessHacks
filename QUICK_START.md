# Quick Start: Play Against Your Bot

## ✅ Pre-Flight Checklist

- [ ] Python 3.8+ installed
- [ ] Node.js 18+ installed
- [ ] Model file exists at `models/chess_evaluator_quick.pth`

## 🚀 Quick Setup (5 minutes)

### 1. Python Setup
```bash
# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Model
```bash
# Check model exists
ls models/chess_evaluator_quick.pth

# If using different model, set path:
export MODEL_PATH=models/your_model.pth
```

### 3. Start the App
```bash
cd devtools
npm install  # Only needed first time
npm run dev
```

### 4. Play!
1. Open browser: http://localhost:3000
2. Start a new game
3. Make your move
4. Bot responds automatically!

## 🐛 Common Issues

**Model not found?**
```bash
export MODEL_PATH=$(pwd)/models/chess_evaluator_quick.pth
```

**Port in use?**
```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

**Import errors?**
- Don't run `main.py` directly!
- Use `npm run dev` from `devtools/` directory

## 📖 Full Guide

See `SETUP_GUIDE.md` for detailed instructions and troubleshooting.

