from .utils import chess_manager, GameContext
from chess import Move
import sys
import os

# Add chess-engine to path
chess_engine_path = "/home/brianzheng/chess-engine/src"
if chess_engine_path not in sys.path:
    sys.path.insert(0, chess_engine_path)

from chess_policy.train import load_checkpoint
from chess_policy.uci import UciEngine
from chess_policy.infer import choose_move
import torch

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etc.

# Load the chess engine model
# Get the directory where this file is located, then go up to repo root
import pathlib
REPO_ROOT = pathlib.Path(__file__).parent.parent
MODEL_PATH = str(REPO_ROOT / "tcec_codex.pt")
# Opening book disabled - it's biased (plays f5 too often)
# OPENING_BOOK_PATH = str(REPO_ROOT / "opening_book.pkl") if (REPO_ROOT / "opening_book.pkl").exists() else None
OPENING_BOOK_PATH = None

print("Loading chess engine model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    model = load_checkpoint(MODEL_PATH, map_location=device)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Create UCI engine with PUCT search
engine = UciEngine(
    model,
    use_puct=True,
    sims=200,  # Number of PUCT simulations
    c_puct=1.2,  # PUCT exploration constant
    device=device,
    opening_book_path=OPENING_BOOK_PATH,
    opening_max_ply=8,
)
print("Chess engine initialized")


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    print("Cooking move with chess engine...")
    
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")

    # Update engine's board to match current position
    engine.board = ctx.board.copy()
    
    # Convert to list for easier access
    legal_move_list = list(legal_moves)
    
    # Calculate current ply for opening book check
    current_ply = len(ctx.board.move_stack)
    
    # Get move probabilities using choose_move for logging
    # This gives us probabilities over all legal moves
    try:
        _, probs_tensor, value = choose_move(
            ctx.board,
            model,
            device=device,
            temperature=0.8,
            sample=False
        )
        
        # Convert probabilities tensor to dictionary of Move -> probability
        # We need to map from move indices to actual moves
        from chess_policy.move_index import move_to_index, index_to_move
        move_probs = {}
        
        # Get probabilities for legal moves only
        for move in legal_move_list:
            try:
                move_idx = move_to_index(ctx.board, move)
                if move_idx is not None and move_idx < len(probs_tensor):
                    prob = float(probs_tensor[move_idx].item())
                    move_probs[move] = prob
            except Exception:
                pass
        
        # Normalize probabilities to sum to 1 (in case of rounding errors)
        total_prob = sum(move_probs.values())
        if total_prob > 0:
            move_probs = {move: prob / total_prob for move, prob in move_probs.items()}
        else:
            # Fallback: uniform distribution
            move_probs = {move: 1.0 / len(legal_move_list) for move in legal_move_list}
        
        ctx.logProbabilities(move_probs)
    except Exception as e:
        print(f"Warning: Could not compute probabilities: {e}")
        # Fallback: uniform distribution
        move_probs = {move: 1.0 / len(legal_move_list) for move in legal_move_list}
        ctx.logProbabilities(move_probs)
    
    # Use the engine's go() method logic, but adapted for our context
    # Reset stop flag
    engine._stop_flag = False
    
    # Calculate simulations based on time (similar to _sims_from_args logic)
    # Default sims from engine config
    sims = engine.sims
    if ctx.timeLeft > 0:
        # Map time to simulations: ~400 sims per second heuristic
        movetime_ms = ctx.timeLeft
        estimated_sims = max(50, int(movetime_ms * 0.4))
        sims = min(estimated_sims, 800)  # Cap at reasonable max
    
    try:
        # Follow the same logic as UciEngine.go() method
        # 1. Check for tactical opportunities (if in opening phase)
        move = None
        
        # Debug: Check if opening book should be used
        if current_ply < engine.opening_max_ply:
            print(f"DEBUG: In opening phase (ply {current_ply} < {engine.opening_max_ply})")
            print(f"DEBUG: Opening book available: {engine.opening_book is not None}")
            if engine.opening_book is not None:
                print(f"DEBUG: Current position in book: {ctx.board.fen() in engine.opening_book}")
        
        if current_ply < engine.opening_max_ply:
            # Check for tactical opportunities first (checkmate, winning captures)
            tactical_move = engine._check_tactical_opportunity()
            if tactical_move is not None:
                move = tactical_move
                print(f"Selected tactical move: {move.uci()}")
                return move
        
        # 2. Check opening book (if in opening phase and no tactical move found)
        if move is None and current_ply < engine.opening_max_ply:
            opening_move = engine._get_opening_move()
            if opening_move is not None:
                print(f"Found opening book move: {opening_move.uci()}")
                # If we have PUCT, do a quick search to compare
                if engine.use_puct and engine.search is not None:
                    quick_sims = min(100, max(50, sims // 4))
                    search_move = engine.search.search(ctx.board, simulations=quick_sims, temperature=0.1)
                    
                    if search_move is not None and search_move != opening_move:
                        # Compare values
                        test_board_book = ctx.board.copy()
                        test_board_book.push(opening_move)
                        value_book = -engine.search._rollout_value(test_board_book)
                        
                        test_board_search = ctx.board.copy()
                        test_board_search.push(search_move)
                        value_search = -engine.search._rollout_value(test_board_search)
                        
                        print(f"Opening book value: {value_book:.3f}, PUCT search value: {value_search:.3f}")
                        # If search move is significantly better (0.2 advantage or more), use it
                        # But be more conservative - opening book moves are usually good
                        if value_search > value_book + 0.3:  # Increased threshold to favor opening book
                            print(f"Using PUCT move (significantly better): {search_move.uci()}")
                            move = search_move
                        else:
                            print(f"Using opening book move: {opening_move.uci()}")
                            move = opening_move
                    else:
                        print(f"Using opening book move (PUCT found same or None): {opening_move.uci()}")
                        move = opening_move
                else:
                    print(f"Using opening book move (no PUCT): {opening_move.uci()}")
                    move = opening_move
            else:
                print(f"No opening book move found for position (ply {current_ply})")
        
        # 3. Use PUCT search for non-opening positions or if opening book didn't provide a move
        if move is None and engine.use_puct and engine.search is not None:
            mv = engine.search.search(ctx.board, simulations=sims, temperature=0.3)
            
            # Check for repetition and adjust if needed
            if mv is not None:
                test_board = ctx.board.copy()
                test_board.push(mv)
                test_fen = test_board.fen()
                if test_fen in engine._recent_positions[-3:]:
                    # Re-search with higher temperature to avoid repetition
                    mv = engine.search.search(ctx.board, simulations=sims, temperature=0.5)
            
            move = mv
        
        # 4. Fallback to greedy policy if PUCT not available or failed
        if move is None:
            print("Using greedy policy fallback")
            move, _, _ = choose_move(ctx.board, model, device=device, temperature=0.8, sample=False)
        
        # 5. Last resort: pick first legal move
        if move is None or move not in legal_moves:
            print("Warning: Using first legal move as last resort")
            move = legal_move_list[0]
        
        # Track position for repetition detection
        if move is not None:
            test_board = ctx.board.copy()
            test_board.push(move)
            engine._recent_positions.append(test_board.fen())
            if len(engine._recent_positions) > 10:
                engine._recent_positions.pop(0)
        
        print(f"Selected move: {move.uci()}")
        return move
        
    except Exception as e:
        print(f"Error in move selection: {e}, falling back to greedy")
        import traceback
        traceback.print_exc()
        # Fallback to greedy selection
        move, _, _ = choose_move(ctx.board, model, device=device, temperature=0.8, sample=False)
        if move is None or move not in legal_moves:
            move = legal_move_list[0]
        return move


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    print("Resetting chess engine for new game...")
    engine.ucinewgame()  # This resets the board and clears caches
    print("Chess engine reset complete")
