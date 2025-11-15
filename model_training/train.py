"""
Training script for chess position evaluation model.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path

from model import create_model
from dataset import ChessPositionDataset
from position_encoder import PositionEncoder


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        positions = batch['position'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(positions)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'avg_loss': total_loss / num_batches})
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            positions = batch['position'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(positions)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train chess position evaluation model')
    parser.add_argument('--dataset', type=str, default='anthonytherrien/leela-chess-zero-self-play-chess-games-dataset-3',
                        help='Kaggle dataset identifier')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to use (None for all)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden-channels', type=int, default=128,
                        help='Number of hidden channels')
    parser.add_argument('--num-residual-blocks', type=int, default=4,
                        help='Number of residual blocks')
    parser.add_argument('--output-dir', type=str, default='../models',
                        help='Output directory for saved models')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--use-game-outcome', action='store_true',
                        help='Use game outcome as label (else use simple evaluation)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    try:
        full_dataset = ChessPositionDataset(
            kaggle_dataset=args.dataset,
            max_samples=args.max_samples,
            use_game_outcome=args.use_game_outcome,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Split into train and validation
    print("Splitting dataset into train and validation...")
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    print(f"Using {len(train_dataset)} samples for training, {len(val_dataset)} for validation")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    # Create model
    print("Creating model...")
    model = create_model(
        input_channels=19,
        hidden_channels=args.hidden_channels,
        num_residual_blocks=args.num_residual_blocks,
        device=device
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = output_dir / 'chess_evaluator_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'model_config': {
                    'input_channels': 19,
                    'hidden_channels': args.hidden_channels,
                    'num_residual_blocks': args.num_residual_blocks,
                }
            }, model_path)
            print(f"Saved best model to {model_path}")
        
        # Save checkpoint every epoch
        checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'model_config': {
                'input_channels': 19,
                'hidden_channels': args.hidden_channels,
                'num_residual_blocks': args.num_residual_blocks,
            }
        }, checkpoint_path)
    
    # Save final model
    final_model_path = output_dir / 'chess_evaluator_final.pth'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'model_config': {
            'input_channels': 19,
            'hidden_channels': args.hidden_channels,
            'num_residual_blocks': args.num_residual_blocks,
        }
    }, final_model_path)
    print(f"\nSaved final model to {final_model_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()

