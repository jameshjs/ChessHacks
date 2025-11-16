import torch
from pathlib import Path

model_path = Path("models") / "best_model.pth"
if not model_path.exists():
    raise FileNotFoundError(f"{model_path} not found. Did you run modal volume get?")

# Replace CnnModel with your model class if needed
from src.main import CnnModel  # adjust import if module layout differs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CnnModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model loaded:", model_path, "Device:", device)