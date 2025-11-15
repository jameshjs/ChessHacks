"""
Quick training script for a minimal chess evaluation model.
Trains a small model on a small dataset in minimal time for testing purposes.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from model import create_model
from dataset import ChessPositionDataset

# Optional Modal support
try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

# Create Modal app if available
if MODAL_AVAILABLE:
    app = modal.App("chess-model-training")
else:
    app = None


def train_quick(
    max_samples=1000,
    batch_size=32,
    epochs=2,
    hidden_channels=32,
    num_residual_blocks=1,
    output_dir='../models',
    device='auto'
):
    """
    Quick training of a minimal model.
    
    Args:
        max_samples: Maximum number of training samples
        batch_size: Batch size
        epochs: Number of epochs
        hidden_channels: Number of hidden channels (smaller = faster)
        num_residual_blocks: Number of residual blocks (smaller = faster)
        output_dir: Output directory
        device: Device to use
    """
    # Determine device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Quick Training - Minimal Model")
    print(f"{'='*50}")
    print(f"Device: {device}")
    print(f"Max samples: {max_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Hidden channels: {hidden_channels}")
    print(f"Residual blocks: {num_residual_blocks}")
    print(f"{'='*50}\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load small dataset
    print("Loading dataset (small subset)...")
    try:
        train_dataset = ChessPositionDataset(
            kaggle_dataset="anthonytherrien/leela-chess-zero-self-play-chess-games-dataset-3",
            max_samples=max_samples,
            use_game_outcome=False,  # Use simple evaluation for speed
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative approach...")
        # Try with even smaller sample
        train_dataset = ChessPositionDataset(
            kaggle_dataset="anthonytherrien/leela-chess-zero-self-play-chess-games-dataset-3",
            max_samples=min(max_samples, 500),
            use_game_outcome=False,
        )
    
    if len(train_dataset) == 0:
        raise ValueError("No valid samples found in dataset")
    
    print(f"Loaded {len(train_dataset)} samples\n")
    
    # Split for validation (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    # Create small model
    print("Creating minimal model...")
    model = create_model(
        input_channels=19,
        hidden_channels=hidden_channels,
        num_residual_blocks=num_residual_blocks,
        device=device
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}\n")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # Higher LR for quick training
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        print("-" * 30)
        
        # Train
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            positions = batch['position'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(positions)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / train_batches
        print(f"Train Loss: {avg_train_loss:.4f}")
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                positions = batch['position'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(positions)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        print(f"Val Loss: {avg_val_loss:.4f}\n")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = output_path / 'chess_evaluator_quick.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'model_config': {
                    'input_channels': 19,
                    'hidden_channels': hidden_channels,
                    'num_residual_blocks': num_residual_blocks,
                }
            }, model_path)
            print(f"✓ Saved best model to {model_path}\n")
    
    # Save final model
    final_model_path = output_path / 'chess_evaluator_quick_final.pth'
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_loss,
        'model_config': {
            'input_channels': 19,
            'hidden_channels': hidden_channels,
            'num_residual_blocks': num_residual_blocks,
        }
    }, final_model_path)
    
    print(f"{'='*50}")
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to: {final_model_path}")
    print(f"{'='*50}")
    print(f"\nTo use this model, set:")
    print(f"  export MODEL_PATH={final_model_path}")


# Modal wrapper (optional)
if MODAL_AVAILABLE and app is not None:
    @app.function(
        image=modal.Image.debian_slim().pip_install([
            "torch", "numpy", "datasets", "tqdm", "python-chess"
        ]),
        gpu="T4",  # Use GPU if available
        timeout=3600,  # 1 hour timeout
    )
    def train_quick_modal(
        max_samples=1000,
        batch_size=32,
        epochs=2,
        hidden_channels=32,
        num_residual_blocks=1,
        output_dir='/tmp/models',
        device='cuda'
    ):
        """Modal wrapper for train_quick."""
        return train_quick(
            max_samples=max_samples,
            batch_size=batch_size,
            epochs=epochs,
            hidden_channels=hidden_channels,
            num_residual_blocks=num_residual_blocks,
            output_dir=output_dir,
            device=device
        )


def main():
    parser = argparse.ArgumentParser(
        description='Quick training of minimal chess evaluation model'
    )
    parser.add_argument('--max-samples', type=int, default=1000,
                        help='Maximum number of samples (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs (default: 2)')
    parser.add_argument('--hidden-channels', type=int, default=32,
                        help='Hidden channels (default: 32, smaller = faster)')
    parser.add_argument('--num-residual-blocks', type=int, default=1,
                        help='Residual blocks (default: 1, smaller = faster)')
    parser.add_argument('--output-dir', type=str, default='../models',
                        help='Output directory (default: ../models)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    train_quick(
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_channels=args.hidden_channels,
        num_residual_blocks=args.num_residual_blocks,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == '__main__':
    # Check if running with Modal
    import sys
    if MODAL_AVAILABLE and app is not None and '--modal' in sys.argv:
        # Remove --modal flag and run with Modal
        sys.argv.remove('--modal')
        if app is not None:
            with app.run():
                train_quick_modal.remote(
                    max_samples=1000,
                    batch_size=32,
                    epochs=2,
                    hidden_channels=32,
                    num_residual_blocks=1,
                )
    else:
        # Run locally
        main()

