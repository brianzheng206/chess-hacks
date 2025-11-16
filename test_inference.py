#!/usr/bin/env python3
"""
Test inference script for stockfish_949.pt model.
This script demonstrates running inference on the chess model using the architecture
from src/chess_policy (local copy).
"""

import sys
import pathlib

# Add src to path to import local chess_policy
src_path = str(pathlib.Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import torch
import chess
from chess_policy.train import load_checkpoint
from chess_policy.infer import choose_move, policy_logits
from chess_policy.encoding import board_to_tensor
from chess_policy.model import PolicyValueResNet

def test_model_architecture():
    """Verify the model architecture matches the expected structure."""
    print("=" * 60)
    print("Testing Model Architecture")
    print("=" * 60)
    
    # Get the model path
    REPO_ROOT = pathlib.Path(__file__).parent
    MODEL_PATH = str(REPO_ROOT / "stockfish_949.pt")
    
    print(f"Loading model from: {MODEL_PATH}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the model
    model = load_checkpoint(MODEL_PATH, map_location=device)
    model.eval()
    model.to(device)
    
    # Print model architecture info
    print(f"\nModel type: {type(model).__name__}")
    if hasattr(model, 'in_channels'):
        print(f"Input channels: {model.in_channels}")
    if hasattr(model, 'width'):
        print(f"Width (channels): {model.width}")
    if hasattr(model, 'n_blocks'):
        print(f"Number of residual blocks: {model.n_blocks}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model, device

def test_inference(model, device):
    """Test inference on a chess position."""
    print("\n" + "=" * 60)
    print("Testing Inference")
    print("=" * 60)
    
    # Create a test board (starting position)
    board = chess.Board()
    print(f"\nTest position (starting position):")
    print(board)
    print(f"FEN: {board.fen()}")
    print(f"Legal moves: {len(list(board.legal_moves))}")
    
    # Test 1: Get policy logits
    print("\n--- Test 1: Policy Logits ---")
    with torch.no_grad():
        logits = policy_logits(model, board, device=device)
        print(f"Policy logits shape: {logits.shape}")
        print(f"Expected shape: [4672] (64 squares × 73 move types)")
        print(f"Max logit: {logits.max().item():.4f}")
        print(f"Min logit: {logits.min().item():.4f}")
        print(f"Mean logit: {logits.mean().item():.4f}")
    
    # Test 2: Full forward pass (policy + value)
    print("\n--- Test 2: Full Forward Pass (Policy + Value) ---")
    with torch.no_grad():
        # Encode board to tensor
        encoding = board_to_tensor(board)
        # Slice to match model's expected input channels (18 for stockfish_949.pt)
        if encoding.shape[0] > model.in_channels:
            encoding = encoding[:model.in_channels]
        elif encoding.shape[0] < model.in_channels:
            # Pad if needed (shouldn't happen for stockfish_949.pt)
            import numpy as np
            pad = np.zeros((model.in_channels - encoding.shape[0], 8, 8), dtype=encoding.dtype)
            encoding = np.concatenate([encoding, pad], axis=0)
        board_tensor = torch.from_numpy(encoding).unsqueeze(0).to(device)
        print(f"Input tensor shape: {board_tensor.shape}")
        print(f"Expected: [1, {model.in_channels}, 8, 8] (batch, channels, height, width)")
        
        # Forward pass
        output = model(board_tensor)
        
        if isinstance(output, tuple):
            policy_logits_out, value_out = output
            print(f"Policy output shape: {policy_logits_out.shape}")
            print(f"Value output shape: {value_out.shape}")
            print(f"Value: {value_out.item():.4f} (range should be [-1, 1])")
        else:
            print(f"Output shape: {output.shape}")
            print("Note: Model only outputs policy (no value head)")
    
    # Test 3: Choose move
    print("\n--- Test 3: Choose Move ---")
    with torch.no_grad():
        move, probs, value = choose_move(
            board,
            model,
            device=device,
            temperature=0.8,
            sample=False
        )
        print(f"Selected move: {move}")
        print(f"Move UCI: {move.uci() if move else None}")
        print(f"Position value: {value:.4f}")
        print(f"Top 5 move probabilities:")
        top5_indices = torch.topk(probs, k=5).indices
        from chess_policy.move_index import index_to_move
        for i, idx in enumerate(top5_indices):
            prob = probs[idx].item()
            move_candidate = index_to_move(board, int(idx))
            print(f"  {i+1}. {move_candidate} (prob: {prob:.4f})")
    
    # Test 4: Test on a different position
    print("\n--- Test 4: Different Position ---")
    board2 = chess.Board()
    board2.push(chess.Move.from_uci("e2e4"))  # e4
    board2.push(chess.Move.from_uci("e7e5"))  # e5
    board2.push(chess.Move.from_uci("g1f3"))  # Nf3
    
    print(f"\nPosition after 1.e4 e5 2.Nf3:")
    print(board2)
    
    with torch.no_grad():
        move2, probs2, value2 = choose_move(
            board2,
            model,
            device=device,
            temperature=0.8,
            sample=False
        )
        print(f"Selected move: {move2}")
        print(f"Position value: {value2:.4f}")

def main():
    """Main function to run all tests."""
    try:
        # Test model architecture
        model, device = test_model_architecture()
        
        # Test inference
        test_inference(model, device)
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

