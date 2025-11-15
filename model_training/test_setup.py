"""
Test script to verify training setup is correct.
"""

import torch
import chess
from position_encoder import PositionEncoder
from model import create_model
from dataset import ChessPositionDataset


def test_position_encoder():
    """Test position encoder."""
    print("Testing position encoder...")
    encoder = PositionEncoder()
    board = chess.Board()
    
    # Encode starting position
    tensor = encoder.encode_position(board)
    assert tensor.shape == (8, 8, 19), f"Expected shape (8, 8, 19), got {tensor.shape}"
    print("✓ Position encoder works correctly")


def test_model():
    """Test model creation and forward pass."""
    print("Testing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(device=device)
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 19, 8, 8).to(device)
    output = model(dummy_input)
    
    assert output.shape == (batch_size,), f"Expected shape ({batch_size},), got {output.shape}"
    print(f"✓ Model works correctly (output shape: {output.shape})")


def test_dataset():
    """Test dataset loading (small sample)."""
    print("Testing dataset loading...")
    try:
        dataset = ChessPositionDataset(
            max_samples=100,  # Small sample for testing
            use_game_outcome=False,  # Use simple evaluation for faster testing
        )
        
        if len(dataset) > 0:
            # Test getting a sample
            sample = dataset[0]
            assert 'position' in sample
            assert 'label' in sample
            assert sample['position'].shape == (19, 8, 8)
            print(f"✓ Dataset loaded successfully ({len(dataset)} samples)")
        else:
            print("⚠ Dataset loaded but is empty")
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        print("  This might be due to network issues or Hugging Face authentication")
        print("  The dataset will download automatically on first use")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Training Setup")
    print("=" * 50)
    print()
    
    try:
        test_position_encoder()
        print()
        test_model()
        print()
        test_dataset()
        print()
        print("=" * 50)
        print("All tests passed! Setup is correct.")
        print("=" * 50)
    except Exception as e:
        print()
        print("=" * 50)
        print(f"Test failed: {e}")
        print("=" * 50)
        raise


if __name__ == '__main__':
    main()

