from .utils import chess_manager, GameContext
from chess import Move, Board
import sys
import os
import io
import contextlib
import time
from functools import lru_cache

# Compatibility shim for python-chess transposition_key
# The chess_policy code expects board.transposition_key() as a method
# but some python-chess versions use _transposition_key as an attribute
if not hasattr(Board, 'transposition_key'):
    def transposition_key(self):
        """Compatibility method for transposition_key."""
        return self._transposition_key
    Board.transposition_key = transposition_key

# Use local chess_policy module (copied into src/chess_policy)
from .chess_policy.train import load_checkpoint
from .chess_policy.uci import UciEngine
from .chess_policy.infer import choose_move
import torch

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etc.

# Load the chess engine model
# Get the directory where this file is located, then go up to repo root
import pathlib
REPO_ROOT = pathlib.Path(__file__).parent.parent
# Use stockfish_949.pt from local repository
MODEL_PATH = str(REPO_ROOT / "stockfish_949_fp16.pt")
# Opening book enabled
OPENING_BOOK_PATH = str(REPO_ROOT / "opening_book.pkl") if (REPO_ROOT / "opening_book.pkl").exists() else None

# Early termination constants for value-based decision skipping
VALUE_EARLY_TERMINATION_THRESHOLD = 0.70  # abs(value) above this → skip PUCT (aggressive for bullet)
VALUE_EARLY_TERMINATION_MIN_PLY = 2       # allow earlier termination (was 4)

# Debug flags - set to True only when debugging (significantly impacts performance)
DEBUG_POLICY = False  # Enable verbose policy extraction and diagnostics
DEBUG_TENSOR = False  # Enable raw tensor diagnostics
DEBUG_MOVE_TIME = True  # Measure and print move time

# Verbose output flag - set to False for bullet games to reduce I/O overhead
VERBOSE = False

# Pure policy bullet mode - skips MCTS entirely, just uses neural network policy
PURE_POLICY_BULLET = False

# Hybrid mode: use pure policy when time is low or game is advanced
USE_HYBRID_MODE = True  # Only used if PURE_POLICY_BULLET is True
HYBRID_TIME_THRESHOLD_MS = 30000  # Use pure policy when timeLeft < 30 seconds
HYBRID_PLY_THRESHOLD = 20  # Use pure policy when ply > 20

# Opening book optimization flags
OPENING_COMPARE_WITH_MCTS = False  # Enable MCTS comparison between model move and opening book (slower but more accurate)
OPENING_QUICK_SEARCH = False  # Enable quick MCTS search to find better opening moves (slower but more accurate)

# Time management flags
LOW_TIME_SKIP_MCTS_MS = 8000  # Skip MCTS entirely when time drops below this (8 seconds, tune this)
CRITICAL_TIME_SKIP_MCTS_MS = 5000  # Force greedy policy when time is critically low (5 seconds)
LOSING_BADLY_THRESHOLD = -0.8  # Force greedy policy when losing very badly (value < -0.9)

print("Loading chess engine model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model checkpoint - use load_checkpoint for robust loading with defaults
# This handles missing metadata, different architectures, and backward compatibility
# IMPORTANT: Model loading errors should not prevent server startup
# The server needs to be able to start even if model loading fails initially
model = None
try:
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model file not found at {MODEL_PATH}")
        print("Server will start but model-dependent features will not work")
    else:
        model = load_checkpoint(MODEL_PATH, map_location=device)
        model.to(device)
        model.eval()
        
        print("Model loaded successfully")
        # Verify model loaded correctly by checking a test inference
        import chess
        from .chess_policy.infer import choose_move
        from .chess_policy.move_index import move_to_index
        test_board = chess.Board()
        test_board.push(chess.Move.from_uci('e2e4'))
        test_move, test_probs, test_value = choose_move(test_board, model, device=device, temperature=0.7, sample=False)
        test_legal = list(test_board.generate_legal_moves())
        test_probs_dict = {}
        # Check ALL legal moves, not just first 5
        for mv in test_legal:
            try:
                idx = move_to_index(test_board, mv)
                if idx is not None and idx < len(test_probs):
                    test_probs_dict[mv] = float(test_probs[idx].item())
            except:
                pass
        if test_probs_dict:
            test_total = sum(test_probs_dict.values())
            test_max = max(test_probs_dict.values()) if test_probs_dict else 0
            # Normalize to see actual probabilities
            if test_total > 0:
                test_probs_normalized = {mv: p / test_total for mv, p in test_probs_dict.items()}
                test_max_norm = max(test_probs_normalized.values())
                sorted_test = sorted(test_probs_normalized.items(), key=lambda x: x[1], reverse=True)
                print(f"Model verification: max prob={test_max_norm:.4f} ({test_max_norm*100:.1f}%), sum={test_total:.4f}")
                print(f"  Top 3 moves: {[(mv.uci(), f'{p*100:.1f}%') for mv, p in sorted_test[:3]]}")
            if test_max < 0.1 or test_total < 0.5:
                print("WARNING: Model probabilities are very low - model may not have loaded correctly!")
                print(f"  This suggests the model weights may not have loaded properly.")
            else:
                print(f"Model verification passed: top move has {test_max*100:.1f}% probability")
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    print("WARNING: Server will start but model-dependent features will not work")
    print("This may be expected in deployment environments - model will be loaded on first request")
    # Don't raise - allow server to start even if model loading fails
    model = None

# Performance optimizations: set model to eval mode and disable gradients
if model is not None:
    model.eval()
    # Compile model for faster inference (PyTorch 2.0+)
    try:
        if hasattr(torch, 'compile'):
            print("Compiling model with torch.compile() for faster inference...")
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled successfully")
    except Exception as e:
        print(f"Warning: torch.compile() failed (may not be available): {e}")
        print("Continuing without compilation - this is fine for older PyTorch versions")
torch.set_grad_enabled(False)

# Note: Warm-up is skipped to allow server to start quickly
# The first move will naturally warm up the model, and the performance impact is minimal

# Create UCI engine with PUCT search (only if model loaded successfully)
# Optimized for 1-minute games: fast, efficient, trusts policy more
# TUNING GUIDE: See MCTS_TUNING_GUIDE.md for detailed parameter tuning instructions
# Key parameters:
#   - sims: Number of MCTS simulations (higher = stronger but slower, default: 150 for 1-min)
#   - c_puct: Exploration constant (lower = trust policy more, default: 0.6 for 1-min)
#     * 0.5-0.6: Very conservative, trusts policy heavily (good for fast games)
#     * 0.7-0.8: Balanced
#     * 1.0-1.2: More exploration, may find hidden tactics but also blunders
engine = None
if model is not None:
    try:
        engine = UciEngine(
            model,
            use_puct=True,
            sims=40,  # Reduced from 120 for bullet/1-minute games (will be adjusted by time management)
            c_puct=0.5,  # Reduced from 0.55 to trust policy more, faster convergence
            device=device,
            opening_book_path=OPENING_BOOK_PATH,
            opening_max_ply=8,
        )
        print("Chess engine initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize UCI engine: {e}")
        import traceback
        traceback.print_exc()
        engine = None
else:
    print("WARNING: Chess engine not initialized - model not loaded")

# FEN-based caching for neural network evaluations
# This avoids re-evaluating the same position multiple times
@lru_cache(maxsize=20000)
def _cached_nn_eval(fen: str):
    """Cache neural network evaluations by FEN string."""
    from chess import Board
    board = Board(fen)
    try:
        move, probs, value = choose_move(board, model, device=device, temperature=0.8, sample=False)
        return probs.cpu() if probs is not None else None, float(value) if value is not None else 0.0
    except Exception as e:
        print(f"Warning: Cached eval failed for FEN {fen[:20]}...: {e}")
        return None, 0.0

def search_with_budget(engine, board, sims_cap, time_ms):
    """Time-budgeted search wrapper that caps wall-clock time."""
    start = time.monotonic()
    
    # Spend ~5% of remaining time, bounded [80ms, 300ms]
    budget_s = max(0.08, min(0.30, (time_ms / 1000.0) * 0.05))
    
    if hasattr(engine, "search_tree") and engine.search_tree is not None and hasattr(engine, 'mcts_config') and engine.mcts_config is not None:
        # Use smaller sims to stay within time budget
        sims_small = max(50, min(sims_cap, 120))
        # Create a temporary search tree for this board
        from .chess_policy.mcts import SearchTree, SearchConfig
        temp_config = SearchConfig(
            n_simulations=sims_small,
            c_puct=engine.mcts_config.c_puct,
            device=engine.device,
            temperature=engine.mcts_config.temperature,
            use_dirichlet_noise=engine.mcts_config.use_dirichlet_noise,
        )
        temp_tree = SearchTree(board, engine.model, temp_config)
        move, _, _ = temp_tree.search()
        
        elapsed = time.monotonic() - start
        # Allow a tiny top-up if we're way under budget (but cap it)
        if elapsed < budget_s * 0.5 and elapsed < 0.15:
            remaining_budget = budget_s - elapsed
            if remaining_budget > 0.05:  # At least 50ms remaining
                topup_sims = min(int(sims_small * 0.5), 60)
                temp_config2 = SearchConfig(
                    n_simulations=topup_sims,
                    c_puct=engine.mcts_config.c_puct,
                    device=engine.device,
                    temperature=engine.mcts_config.temperature,
                    use_dirichlet_noise=engine.mcts_config.use_dirichlet_noise,
                )
                temp_tree2 = SearchTree(board, engine.model, temp_config2)
                move2, _, _ = temp_tree2.search()
                if move2 is not None:
                    move = move2
        
        return move
    
    # Fallback: no search available
    return None

def calculate_game_phase(board):
    """Calculate game phase: 0.0 = endgame, 1.0 = opening."""
    # Simple heuristic: count non-pawn material
    piece_values = {'Q': 9, 'R': 5, 'B': 3, 'N': 3, 'P': 1}
    total_material = 0
    for square in board.piece_map().values():
        if square.symbol().upper() != 'P' and square.symbol().upper() != 'K':
            total_material += piece_values.get(square.symbol().upper(), 0)
    
    # Normalize: full material = 1.0, endgame threshold = 0.3
    max_material = 2 * (9 + 5 + 3 + 3)  # 2 queens, rooks, bishops, knights
    phase = min(1.0, total_material / (max_material * 0.3))
    return phase


def value_to_sims_scale(value: float) -> float:
    """Convert NN value to a simulation scale factor.
    
    Args:
        value: Evaluation from the perspective of the side to move, in [-1, 1].
            - abs(value) near 0 → unclear position → scale ~1.0 (full search)
            - abs(value) near 1 → clearly winning/losing → scale ~0.15 (very reduced search)
            - For losing positions (value < -0.3), use even more aggressive scaling
    
    Returns:
        Scale factor in [0.15, 1.0] for adjusting simulation count (more aggressive for speed).
    """
    abs_value = abs(float(value))
    value_float = float(value)
    
    # More aggressive scaling for losing positions - when losing, search less
    if value_float < -0.3:  # Losing position
        # For losing positions, be very aggressive: scale down more
        if abs_value >= 0.85:
            scale = 0.10  # Very aggressive for clearly losing positions
        elif abs_value >= 0.5:
            scale = 0.20  # Aggressive for moderately losing positions
        else:
            scale = 0.35  # Still reduce for slightly losing positions (like -0.417)
    elif abs_value >= 0.85:  # Very winning/losing (but winning)
        scale = 0.15  # Very aggressive for clearly winning positions
    else:
        # Linear mapping: abs_value 0.0 → scale 1.0, abs_value 0.85 → scale ~0.065 (bullet: shrink harder)
        scale = 1.0 - 1.1 * abs_value
    
    # Clamp to [0.15, 1.0] to ensure reasonable bounds
    return max(0.15, min(1.0, scale))


def format_value_eval(value: float) -> str:
    """Convert a value in [-1, 1] to a human-readable string.
    
    Args:
        value: Evaluation from the perspective of the side to move.
            Range approximately [-1, 1] where 1 = winning, -1 = losing, 0 = equal.
    
    Returns:
        Human-readable string like "Eval: +0.80 (winning for side to move)"
        or "Eval: -0.95 (losing for side to move)"
    """
    if value is None:
        return "Eval: N/A (no value available)"
    
    value_float = float(value)
    
    # Determine evaluation description
    if value_float > 0.5:
        desc = "strongly winning for side to move"
    elif value_float > 0.1:
        desc = "winning for side to move"
    elif value_float > -0.1:
        desc = "equal position"
    elif value_float > -0.5:
        desc = "losing for side to move"
    else:
        desc = "strongly losing for side to move"
    
    # Format with sign
    sign = "+" if value_float >= 0 else ""
    return f"Eval: {sign}{value_float:.3f} ({desc})"


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position
    
    # Measure total move time if debug is enabled
    move_start_time = None
    if DEBUG_MOVE_TIME:
        move_start_time = time.monotonic()

    if VERBOSE:
        print("Cooking move with chess engine...")
    
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")

    # Check if engine is initialized
    if engine is None or model is None:
        print("ERROR: Engine or model not initialized - cannot make move")
        # Fallback: return first legal move
        move = legal_moves[0]
        move_probs = {move: 1.0}
        ctx.logProbabilities(move_probs)
        return move

    # Update engine's board to match current position
    engine.board = ctx.board.copy()
    
    # Ensure _recent_positions exists to avoid AttributeError
    if not hasattr(engine, "_recent_positions"):
        engine._recent_positions = []
    
    # Try to update search tree if it exists and position changed
    # This handles tree reuse when opponent plays a move
    if hasattr(engine, 'search_tree') and engine.search_tree is not None:
        try:
            # Check if we can update the tree to match current position
            # If the tree's root is one move away (opponent's move), update it
            if engine.search_tree.root_state.fen() != ctx.board.fen():
                # Position changed - could be opponent's move
                # Try to find if any child of current root matches the new position
                # For now, we'll recreate if position doesn't match (simple approach)
                # A more sophisticated approach would check if position is reachable from tree
                engine.search_tree = None  # Will be recreated below if needed
        except Exception:
            # If check fails, reset tree
            engine.search_tree = None
    
    # Convert to list for easier access
    legal_move_list = list(legal_moves)
    
    # Calculate current ply for opening book check
    current_ply = len(ctx.board.move_stack)
    
    # OPTIMIZATION: For very early opening (first 4 moves), skip NN evaluation
    # and go straight to opening book for maximum speed
    skip_nn_eval = current_ply < 4
    
    # -------------------------
    # STEP 1: POLICY EVALUATION (skip for very early opening)
    # -------------------------
    policy_move = None
    probs_tensor = None
    value = None
    sorted_moves = []   # NEW: always defined as list
    move_probs = {}     # NEW: initialize here for logging

    # Only run NN evaluation if not in very early opening
    if not skip_nn_eval:
        try:
            # Determine temperature based on game phase
            if current_ply < 12:
                # Opening: use lower temperature for more deterministic play
                temperature = 0.7
            else:
                # Mid/endgame: slightly higher temperature
                temperature = 0.8
            
            # Single forward pass - compute policy once with timing
            nn_start = time.monotonic()
            policy_move, probs_tensor, value = choose_move(
                ctx.board,
                model,
                device=device,
                temperature=temperature,
                sample=False
            )
            nn_time = (time.monotonic() - nn_start) * 1000
            if VERBOSE:
                if nn_time > 10:  # Only print if it's significant
                    print(f"NN forward (root): {nn_time:.1f} ms")
                
                # Log the neural network value evaluation (side to move perspective)
                if value is not None:
                    print(format_value_eval(float(value)))
            
            # Ensure we got a valid move
            if policy_move is None or policy_move not in legal_moves:
                if VERBOSE:
                    print("Warning: choose_move returned invalid move, using first legal move")
                policy_move = legal_move_list[0]
            
            # Convert probabilities tensor to dictionary of Move -> probability for logging
            # Use move_to_index to map moves to indices (more reliable than index_to_move)
            from .chess_policy.move_index import move_to_index
            
            move_probs = {}
            
            if DEBUG_POLICY and probs_tensor is not None:
                # DEBUG MODE: Full extraction of all legal moves (expensive)
                # Iterate through all legal moves and get their probabilities
                failed_moves = []
                for move in legal_move_list:
                    try:
                        # Get the index for this move
                        move_idx = move_to_index(ctx.board, move)
                        if move_idx is not None and 0 <= move_idx < len(probs_tensor):
                            prob = float(probs_tensor[move_idx].item())
                            # Some moves might have multiple indices (underpromotions), take max
                            if move not in move_probs or prob > move_probs[move]:
                                move_probs[move] = prob
                        else:
                            failed_moves.append((move, move_idx))
                    except (ValueError, AttributeError, TypeError) as e:
                        # move_to_index can raise ValueError for some edge cases
                        failed_moves.append((move, f"Exception: {e}"))
                
                # Debug: warn if we failed to extract probabilities for many moves
                if len(failed_moves) > 0:
                    print(f"Warning: Failed to extract probabilities for {len(failed_moves)} moves")
                    if len(failed_moves) <= 5:
                        for mv, reason in failed_moves:
                            print(f"  {mv.uci()}: {reason}")
                
                # Verify we got probabilities for the policy_move
                if policy_move is not None and policy_move not in move_probs:
                    print(f"WARNING: policy_move {policy_move.uci()} not in move_probs! Trying to extract...")
                    try:
                        policy_idx = move_to_index(ctx.board, policy_move)
                        if policy_idx is not None and 0 <= policy_idx < len(probs_tensor):
                            policy_prob = float(probs_tensor[policy_idx].item())
                            move_probs[policy_move] = policy_prob
                            print(f"  Extracted policy_move prob: {policy_prob:.6f}")
                    except Exception as e:
                        print(f"  Failed to extract policy_move prob: {e}")
                
                # The probabilities are already normalized in probs_tensor, but we only extracted legal moves
                # So they should sum to ~1.0 (minus tiny probabilities on illegal moves)
                # We don't need to normalize again - the probabilities are correct as-is
                total_prob = sum(move_probs.values())
                
                # Debug: Check probability distribution
                max_prob = max(move_probs.values()) if move_probs else 0
                
                # Raw tensor diagnostics (only in DEBUG_TENSOR mode)
                if DEBUG_TENSOR and probs_tensor is not None and move_probs:
                    from .chess_policy.encoding import legal_mask_4672
                    legal_mask = legal_mask_4672(ctx.board)
                    legal_probs_raw = [float(probs_tensor[i].item()) for i in range(len(probs_tensor)) if legal_mask[i] > 0.5]
                    if legal_probs_raw:
                        legal_max_raw = max(legal_probs_raw)
                        legal_sum_raw = sum(legal_probs_raw)
                        print(f"Raw tensor check: max_legal={legal_max_raw:.6f}, sum_legal={legal_sum_raw:.6f}, extracted_max={max_prob:.6f}, extracted_count={len(move_probs)}")
                        # If there's a big discrepancy, something is wrong with extraction
                        if abs(legal_max_raw - max_prob) > 0.1:
                            print(f"WARNING: Raw tensor max ({legal_max_raw:.6f}) doesn't match extracted max ({max_prob:.6f})!")
                
                # Debug output to diagnose probability issues (only in DEBUG_TENSOR mode)
                # Print debug info if probabilities are suspiciously uniform/low
                if DEBUG_TENSOR and (max_prob < 0.15 or (max_prob < 0.2 and len(legal_move_list) > 15)):
                    print(f"DEBUG: Probabilities seem low/uniform")
                    print(f"  Position: {ctx.board.fen()[:50]}...")
                    print(f"  Extracted probs: max={max_prob:.4f}, sum={total_prob:.4f}, legal_moves={len(legal_move_list)}")
                    # Check if probs_tensor itself has low values
                    if probs_tensor is not None:
                        probs_max = float(probs_tensor.max().item())
                        probs_sum = float(probs_tensor.sum().item())
                        from .chess_policy.encoding import legal_mask_4672
                        legal_mask = legal_mask_4672(ctx.board)
                        legal_probs_raw = [float(probs_tensor[i].item()) for i in range(len(probs_tensor)) if legal_mask[i] > 0.5]
                        legal_max_raw = max(legal_probs_raw) if legal_probs_raw else 0
                        legal_sum_raw = sum(legal_probs_raw) if legal_probs_raw else 0
                        print(f"  probs_tensor: max={probs_max:.4f}, sum={probs_sum:.4f}, shape={probs_tensor.shape}")
                        print(f"  Legal moves in tensor: max={legal_max_raw:.4f}, sum={legal_sum_raw:.4f}, count={len(legal_probs_raw)}")
                        if legal_sum_raw < 0.5:
                            print(f"  WARNING: Legal moves sum to only {legal_sum_raw:.4f} - model may not be confident or extraction is wrong!")
                        if abs(total_prob - legal_sum_raw) > 0.1:
                            print(f"  WARNING: Extracted probs sum ({total_prob:.4f}) doesn't match legal_sum_raw ({legal_sum_raw:.4f}) - extraction may be incomplete!")
                
                # Print top 5 moves with probabilities (only in DEBUG_POLICY mode)
                # Note: These are extracted probabilities for logging - policy_move is the ground truth
                sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
                if DEBUG_POLICY:
                    top_5 = sorted_moves[:5]
                    print("Top 5 moves from model (for logging):")
                    for i, (move, prob) in enumerate(top_5, 1):
                        marker = "✓" if move == policy_move else " "
                        print(f"  {marker} {i}. {move.uci()}: {prob:.4f} ({prob*100:.2f}%)")
            else:
                # FAST PATH: Only track probability for policy_move (for logs / trust override)
                if probs_tensor is not None and policy_move is not None:
                    try:
                        move_idx = move_to_index(ctx.board, policy_move)
                        if move_idx is not None and 0 <= move_idx < len(probs_tensor):
                            move_probs[policy_move] = float(probs_tensor[move_idx].item())
                    except Exception:
                        pass
                # Create empty sorted_moves for compatibility
                sorted_moves = []
            
                # policy_move from choose_move is the model's actual argmax - this is what we use for selection
            if policy_move is not None and VERBOSE:
                policy_prob = move_probs.get(policy_move, 0.0) if move_probs else 0.0
                print(f"Using policy_move (model's argmax): {policy_move.uci()} (prob={policy_prob:.4f})")
        except Exception as e:
            print(f"Warning: Could not compute probabilities: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: uniform distribution and use first legal move
            move_probs = {move: 1.0 / len(legal_move_list) for move in legal_move_list}
        sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
        # Log probabilities for downstream tooling (handles both success and exception cases)
        try:
            ctx.logProbabilities(move_probs)
        except Exception as log_err:
            print(f"Warning: logProbabilities failed: {log_err}")
        # Ensure policy_move is set to a valid move
        if policy_move is None or policy_move not in legal_moves:
            policy_move = legal_move_list[0]
    
    # ------------------------------
    # PURE POLICY BULLET MODE (skip MCTS entirely for maximum speed)
    # ------------------------------
    # Check if we should use pure policy mode (no MCTS at all)
    use_pure_policy = False
    if PURE_POLICY_BULLET:
        if USE_HYBRID_MODE:
            # Hybrid mode: use pure policy when time is low or game is advanced
            movetime_ms = ctx.timeLeft if ctx.timeLeft and ctx.timeLeft > 0 else 0
            if movetime_ms > 0 and movetime_ms < HYBRID_TIME_THRESHOLD_MS:
                use_pure_policy = True
                if VERBOSE:
                    print(f"Pure policy mode (hybrid): timeLeft={movetime_ms:.0f}ms < {HYBRID_TIME_THRESHOLD_MS}ms")
            elif current_ply > HYBRID_PLY_THRESHOLD:
                use_pure_policy = True
                if VERBOSE:
                    print(f"Pure policy mode (hybrid): ply={current_ply} > {HYBRID_PLY_THRESHOLD}")
        else:
            # Always use pure policy
            use_pure_policy = True
            if VERBOSE:
                print("Pure policy mode (always)")
    
    if use_pure_policy:
        # Just trust the net, no MCTS at all - fastest possible mode
        move = policy_move if policy_move is not None and policy_move in legal_moves else legal_move_list[0]
        if VERBOSE:
            print(f"Pure policy move: {move.uci()}")
        # Log probabilities if available
        if move_probs:
            ctx.logProbabilities(move_probs)
        else:
            ctx.logProbabilities({move: 1.0})
        # Print move time if debug is enabled
        if DEBUG_MOVE_TIME and move_start_time is not None:
            move_time_ms = (time.monotonic() - move_start_time) * 1000
            print(f"[DEBUG_MOVE_TIME] Total move time (pure policy): {move_time_ms:.1f} ms (ply {current_ply})")
        return move
    
    # ------------------------------
    # EARLY TERMINATION CHECK (based on NN value and policy confidence)
    # ------------------------------
    # NEW: Check if position is "decided" based on extreme value magnitude
    # Guardrails: Don't skip search in tactical positions (checks, many forcing moves)
    # NEW: Also check policy confidence - if policy is very confident, skip MCTS
    early_termination = False
    policy_confidence_skip = False
    
    # Compute tactical indicators
    num_check_moves = sum(1 for m in legal_move_list if ctx.board.gives_check(m))
    in_check = ctx.board.is_check()
    
    if VERBOSE:
        print(f"Checks available: {num_check_moves}; in_check={in_check}")
    
    # Policy confidence check: if top policy move has very high probability, skip MCTS
    if move_probs and policy_move is not None:
        top_policy_prob = move_probs.get(policy_move, 0.0)
        # If top move has >40% probability and no tactical complexity, trust policy (aggressive for bullet)
        if top_policy_prob > 0.40 and not in_check and num_check_moves < 2:
            policy_confidence_skip = True
            if VERBOSE:
                print(f"Policy confidence skip: top move has {top_policy_prob*100:.1f}% probability")
    
    # Early termination conditions:
    # 1. Value must be available
    # 2. Must be past minimum ply
    # 3. Value magnitude must be extreme
    # 4. Not in check (tactical)
    # 5. Not too many check moves available (tactical)
    if value is not None and current_ply >= VALUE_EARLY_TERMINATION_MIN_PLY:
        if abs(float(value)) >= VALUE_EARLY_TERMINATION_THRESHOLD:
            if not in_check and num_check_moves < 3:
                early_termination = True
                if VERBOSE:
                    print(f"Early termination triggered (value={float(value):+.3f})")
            elif VERBOSE:
                print("Skipping early termination due to tactical complexity")
    
    # ------------------------------
    # STEP 2: CONFIGURE SEARCH BUDGET
    # ------------------------------
    # Reset stop flag
    engine._stop_flag = False
    
    # Phase-aware simulation calculation
    game_phase = calculate_game_phase(ctx.board)
    
    # Calculate simulations based on time and game phase
    # Default sims from engine config
    sims = engine.sims
    # Set default movetime_ms to avoid UnboundLocalError when ctx.timeLeft <= 0
    movetime_ms = ctx.timeLeft if ctx.timeLeft and ctx.timeLeft > 0 else 0
    
    if movetime_ms > 0:
        # Dynamic time management: adapt simulations based on available time
        # Optimized for 1-minute games: very conservative time usage
        
        # For 1-minute games, be very conservative with time usage
        # Estimate moves remaining (assume ~35-40 moves per game for 1-min)
        # Use 1.5-2% of remaining time per move to ensure we finish the game
        moves_remaining_estimate = max(8, int(movetime_ms / 1200))  # Slightly more conservative
        time_per_move_ms = movetime_ms / max(moves_remaining_estimate, 1)
        
        # Adaptive simulation rate based on time per move (optimized for speed)
        # In 1-min games, prioritize speed and efficiency - increased rates for faster moves
        if time_per_move_ms > 2500:  # >2.5 seconds per move: can search more
            sims_per_sec = 120  # Increased from 100
        elif time_per_move_ms > 1200:  # 1.2-2.5 seconds: moderate search
            sims_per_sec = 100  # Increased from 80
        elif time_per_move_ms > 600:  # 0.6-1.2 seconds: fast search
            sims_per_sec = 80  # Increased from 60
        elif time_per_move_ms > 300:  # 0.3-0.6 seconds: very fast
            sims_per_sec = 65  # Increased from 50
        else:  # <0.3 seconds: critical, minimal search
            sims_per_sec = 45  # Increased from 35, but still minimal
        
        # Use conservative time budget: only use 75% of estimated time per move
        # Reserve 25% for safety and overhead (more conservative for 1-min games)
        time_budget_ms = time_per_move_ms * 0.75
        estimated_sims = max(15, int(time_budget_ms * sims_per_sec / 1000.0))
        
        # Cap simulations based on time remaining (optimized for bullet/1-minute games)
        if movetime_ms > 40000:  # >40 seconds left (early game)
            max_sims = 60  # Reduced from 90 for bullet
        elif movetime_ms > 25000:  # 25-40 seconds
            max_sims = 45  # Reduced from 75 for bullet
        elif movetime_ms > 15000:  # 15-25 seconds
            max_sims = 35  # Reduced from 60 for bullet
        elif movetime_ms > 8000:  # 8-15 seconds
            max_sims = 28  # Reduced from 45 for bullet
        elif movetime_ms > 4000:  # 4-8 seconds
            max_sims = 22  # Reduced from 35 for bullet
        elif movetime_ms > 2000:  # 2-4 seconds
            max_sims = 16  # Reduced from 25 for bullet
        else:  # <2 seconds: critical time
            max_sims = 10  # Reduced from 15 for bullet
        
        sims = min(estimated_sims, max_sims)
        
        # Additional time pressure handling (optimized for bullet/1-minute games)
        if movetime_ms < 15000:  # Less than 15 seconds
            sims = min(sims, 40)  # Reduced from 65 for bullet
        if movetime_ms < 8000:  # Less than 8 seconds
            sims = min(sims, 28)  # Reduced from 45 for bullet
        if movetime_ms < 4000:  # Less than 4 seconds
            sims = min(sims, 20)  # Reduced from 30 for bullet
        if movetime_ms < 2000:  # Less than 2 seconds
            sims = min(sims, 12)  # Reduced from 18 for bullet
        
        # Absolute global cap for bullet/1-minute games
        sims = min(sims, 50)
    else:
        # No time info available - use conservative defaults
        sims = min(sims, 50)  # Reduced from 80 for bullet
    
    # Phase-aware adjustments (optimized for speed while maintaining accuracy)
    # Strategy: Trust policy more, reduce search in all phases for faster moves
    # TUNE: Adjust phase-specific parameters if blunders occur in specific phases
    if game_phase > 0.7:  # Opening
        sims = min(sims, 40)  # Reduced from 50 - opening is usually book moves, prioritize speed
        if hasattr(engine, 'mcts_config') and engine.mcts_config is not None:
            engine.mcts_config.c_puct = 0.8  # Reduced from 1.0 - trust policy more
    elif game_phase < 0.3:  # Endgame
        # In endgame, be very efficient - trust policy heavily
        if movetime_ms > 0:
            if movetime_ms < 8000:  # Less than 8 seconds: time pressure
                sims = min(sims, 40)  # Reduced from 50
                if hasattr(engine, 'mcts_config') and engine.mcts_config is not None:
                    engine.mcts_config.c_puct = 0.4  # Trust policy very heavily in time trouble
            elif movetime_ms < 15000:  # 8-15 seconds: moderate time
                sims = min(sims, 50)  # Reduced from 60
                if hasattr(engine, 'mcts_config') and engine.mcts_config is not None:
                    engine.mcts_config.c_puct = 0.45  # Reduced from 0.5 - trust policy more
            else:  # >15 seconds: can search more
                sims = min(sims, 60)  # Reduced from 75
                if hasattr(engine, 'mcts_config') and engine.mcts_config is not None:
                    engine.mcts_config.c_puct = 0.45  # Reduced from 0.5
        else:
            # No time info - use safe defaults for endgame
            sims = min(sims, 50)  # Reduced from 60
            if hasattr(engine, 'mcts_config') and engine.mcts_config is not None:
                engine.mcts_config.c_puct = 0.45
    else:  # Midgame - prioritize speed, especially in winning positions
        # Midgame: be very fast, trust policy more
        if movetime_ms > 0:
            if movetime_ms > 30000:  # Plenty of time: still prioritize speed
                sims = min(sims, 80)  # Reduced from 110 - faster moves
                if hasattr(engine, 'mcts_config') and engine.mcts_config is not None:
                    engine.mcts_config.c_puct = 0.5  # Trust policy more
            elif movetime_ms > 15000:  # Moderate time: be fast
                sims = min(sims, 55)  # Reduced from 80
                if hasattr(engine, 'mcts_config') and engine.mcts_config is not None:
                    engine.mcts_config.c_puct = 0.5  # Trust policy more
            else:  # Time pressure: minimal search
                sims = min(sims, 40)  # Reduced from 60
                if hasattr(engine, 'mcts_config') and engine.mcts_config is not None:
                    engine.mcts_config.c_puct = 0.45  # Trust policy heavily
        else:
            # No time info - use safe defaults for midgame
            sims = min(sims, 55)  # Reduced from 75
            if hasattr(engine, 'mcts_config') and engine.mcts_config is not None:
                engine.mcts_config.c_puct = 0.5
    
    # ------------------------------
    # VALUE-AWARE SIMS SCALING
    # ------------------------------
    # NEW: Adjust sims based on position clarity (value magnitude)
    # Unclear positions (value ~0) get full search, clear positions (|value| ~1) get reduced search
    # Losing positions get even more aggressive reduction
    if value is not None and not early_termination:
        scale = value_to_sims_scale(float(value))
        sims_before = sims
        value_float = float(value)
        abs_value = abs(float(value))
        
        # More aggressive minimums for losing/winning positions (bullet-optimized)
        if abs_value >= 0.85:
            min_sims = 4  # Very winning/losing - minimal search (reduced from 8 for bullet)
        elif value_float < -0.3:  # Losing position
            min_sims = 4  # Losing - very minimal search (reduced from 8 for bullet)
        elif abs_value >= 0.5:
            min_sims = 6  # Moderately winning/losing (reduced from 10 for bullet)
        else:
            min_sims = 8  # Unclear positions (reduced from 12 for bullet)
        
        sims = max(min_sims, int(sims * scale))
        if VERBOSE:
            print(f"Value-aware sims scaling: value={value_float:+.3f}, scale={scale:.2f}, sims {sims_before} → {sims}")
    
    # ------------------------------
    # STEP 3: SELECTION STRATEGY (using UCI engine logic)
    # ------------------------------
    move = None
    decision_mode = None

    try:
        # Determine if we're in opening phase (same logic as UCI engine)
        # Ensure engine has required attributes
        if not hasattr(engine, 'mcts_config'):
            # This shouldn't happen if engine is properly initialized, but handle it gracefully
            print("Warning: engine.mcts_config not found, MCTS features disabled")
        in_opening = False
        if engine.opening_adaptive_depth:
            # Adaptive: check if position is in opening book
            if engine.opening_book is not None and ctx.board.fen() in engine.opening_book:
                in_opening = True
        else:
            # Fixed depth: check if within max_ply
            in_opening = current_ply < engine.opening_max_ply
        
        # If in opening phase, check for tactical opportunities first
        # This allows the engine to catch blunders even during opening book moves
        if in_opening:
            try:
                tactical_move = engine._check_tactical_opportunity()
            except AttributeError as e:
                # Handle any attribute errors gracefully
                print(f"Warning: Error in _check_tactical_opportunity: {e}")
                tactical_move = None
            if tactical_move is not None:
                # Found a tactical opportunity - use it instead of opening book
                decision_mode = "Tactical opportunity"
                move = tactical_move
                if VERBOSE:
                    print(f"Decision mode: {decision_mode} (opening, ply {current_ply})")
                    print(f"Using tactical move: {move.uci()}")
                # Track position after move
                test_board = ctx.board.copy()
                test_board.push(tactical_move)
                engine._recent_positions.append(test_board.fen())
                if len(engine._recent_positions) > 10:
                    engine._recent_positions.pop(0)
                return move
        
        # Check opening book (if in opening phase and no tactical opportunity found)
        mv = engine._get_opening_move()
        if mv is not None:
            # OPTIMIZATION: For very early opening (ply < 4), use opening book immediately
            # Skip model comparison to save time
            if skip_nn_eval:
                decision_mode = "Opening book (fast)"
                move = mv
                if VERBOSE:
                    print(f"Decision mode: {decision_mode} (ply {current_ply})")
                    print(f"Using opening book move: {move.uci()}")
                
                # Create simple move_probs for logging
                move_probs = {move: 1.0}
                
                # Track position after move
                test_board = ctx.board.copy()
                test_board.push(mv)
                engine._recent_positions.append(test_board.fen())
                if len(engine._recent_positions) > 10:
                    engine._recent_positions.pop(0)
                
                ctx.logProbabilities(move_probs)
                # Print move time if debug is enabled
                if DEBUG_MOVE_TIME and move_start_time is not None:
                    move_time_ms = (time.monotonic() - move_start_time) * 1000
                    print(f"[DEBUG_MOVE_TIME] Total move time (opening book): {move_time_ms:.1f} ms (ply {current_ply})")
                return move
            
            # For later opening moves (ply >= 4), compare with model if available
            # Get opening book move confidence (weight of selected move)
            fen = ctx.board.fen()
            book_moves = engine.opening_book.get(fen, []) if engine.opening_book else []
            book_move_weight = next((w for m, w in book_moves if m == mv), 0.5)
            
            # Find the top model move (highest probability)
            top_model_move = None
            top_model_prob = 0.0
            if move_probs:
                top_model_move, top_model_prob = max(move_probs.items(), key=lambda x: x[1])
            
            # Check if the model is very confident in a different move
            # If top model move has >35% probability and is different from opening book, compare them
            # Increased threshold from 0.3 to 0.35 to reduce unnecessary comparisons
            if top_model_move is not None and top_model_move != mv and top_model_prob > 0.35:
                # Model is confident in a different move - compare with opening book using MCTS
                # Only do MCTS comparison if explicitly enabled (disabled by default for speed)
                if OPENING_COMPARE_WITH_MCTS and engine.use_puct and hasattr(engine, 'mcts_config') and engine.mcts_config is not None:
                    # Quick evaluation of both moves using MCTS - reduced sims for speed
                    from .chess_policy.mcts import SearchTree, SearchConfig
                    quick_config = SearchConfig(
                        n_simulations=35,  # Reduced from 50 for faster comparison
                        c_puct=engine.mcts_config.c_puct,
                        device=engine.device,
                        temperature=0.0,
                        use_dirichlet_noise=False,
                    )
                    test_board_book = ctx.board.copy()
                    test_board_book.push(mv)
                    temp_tree_book = SearchTree(test_board_book, engine.model, quick_config)
                    _, _, value_book_raw = temp_tree_book.search()
                    value_book = -value_book_raw  # Negate: opponent's perspective -> ours
                    
                    test_board_model = ctx.board.copy()
                    test_board_model.push(top_model_move)
                    temp_tree_model = SearchTree(test_board_model, engine.model, quick_config)
                    _, _, value_model_raw = temp_tree_model.search()
                    value_model = -value_model_raw  # Negate: opponent's perspective -> ours
                    
                    # If model move is better or similar (within 0.1), use it
                    # Also if model confidence is very high (>40%), prefer it
                    if value_model > value_book - 0.1 or top_model_prob > 0.4:
                        decision_mode = "Model override (high confidence)"
                        move = top_model_move
                        if VERBOSE:
                            print(f"Decision mode: {decision_mode} (opening, ply {current_ply})")
                            print(f"Model confident ({top_model_prob*100:.1f}%) in {top_model_move.uci()} (value {value_model:.3f}), overriding opening book {mv.uci()} (value {value_book:.3f})")
                        # Track position after move
                        test_board = ctx.board.copy()
                        test_board.push(move)
                        engine._recent_positions.append(test_board.fen())
                        if len(engine._recent_positions) > 10:
                            engine._recent_positions.pop(0)
                        return move
                elif top_model_prob > 0.5:
                    # Very high confidence, use it even without PUCT
                    decision_mode = "Model override (very high confidence)"
                    move = top_model_move
                    if VERBOSE:
                        print(f"Decision mode: {decision_mode} (opening, ply {current_ply})")
                        print(f"Model very confident ({top_model_prob*100:.1f}%) in {top_model_move.uci()}, overriding opening book {mv.uci()}")
                    # Track position after move
                    test_board = ctx.board.copy()
                    test_board.push(move)
                    engine._recent_positions.append(test_board.fen())
                    if len(engine._recent_positions) > 10:
                        engine._recent_positions.pop(0)
                    return move
            
            # If we have MCTS available, do a quick search to see if there's a better move
            # This catches subtle advantages that the simple tactical check might miss
            # Only do quick search if explicitly enabled (disabled by default for speed)
            if OPENING_QUICK_SEARCH and engine.use_puct and hasattr(engine, 'mcts_config') and engine.mcts_config is not None:
                # Quick search with fewer simulations (30-70) to compare with opening book - reduced for speed
                quick_sims = min(70, max(30, engine.sims // 5))  # Reduced from 50-100 and //4 to //5
                from .chess_policy.mcts import SearchTree, SearchConfig
                quick_config = SearchConfig(
                    n_simulations=quick_sims,
                    c_puct=engine.mcts_config.c_puct,
                    device=engine.device,
                    temperature=0.0,
                    use_dirichlet_noise=False,
                )
                quick_tree = SearchTree(ctx.board, engine.model, quick_config)
                search_move, _, _ = quick_tree.search()
                
                if search_move is not None and search_move != mv:
                    # Compare values: get evaluation for both moves
                    # After pushing a move, the board's turn flips, so we need to negate the value
                    # to get it from our perspective
                    test_board_book = ctx.board.copy()
                    test_board_book.push(mv)
                    temp_tree_book = SearchTree(test_board_book, engine.model, quick_config)
                    _, _, value_book_raw = temp_tree_book.search()
                    value_book = -value_book_raw  # Negate: opponent's perspective -> ours
                    
                    test_board_search = ctx.board.copy()
                    test_board_search.push(search_move)
                    temp_tree_search = SearchTree(test_board_search, engine.model, quick_config)
                    _, _, value_search_raw = temp_tree_search.search()
                    value_search = -value_search_raw  # Negate: opponent's perspective -> ours
                    
                    # Adaptive threshold: higher for early opening and high-confidence book moves
                    # Early opening (ply 0-4): trust book more (threshold 0.25-0.3)
                    # Mid opening (ply 4+): more flexible (threshold 0.15-0.2)
                    # High confidence book move (>0.8): trust more (increase threshold)
                    base_threshold = 0.20 if current_ply < 4 else 0.18
                    if book_move_weight > 0.8:
                        threshold = base_threshold + 0.1  # Trust high-confidence moves more
                    else:
                        threshold = base_threshold
                    
                    # If search move is significantly better, use it
                    # Value is from our perspective, so higher is better
                    if value_search > value_book + threshold:
                        mv = search_move
                        decision_mode = "Opening book (PUCT override)"
                        if VERBOSE:
                            print(f"Decision mode: {decision_mode} (opening, ply {current_ply})")
                            print(f"PUCT found better move: {search_move.uci()} (value {value_search:.3f} vs book {value_book:.3f})")
                    else:
                        decision_mode = "Opening book"
                        if VERBOSE:
                            print(f"Decision mode: {decision_mode} (opening, ply {current_ply})")
                            print(f"Using opening book move: {mv.uci()} (value {value_book:.3f} vs PUCT {value_search:.3f})")
                else:
                    decision_mode = "Opening book"
                    if VERBOSE:
                        print(f"Decision mode: {decision_mode} (opening, ply {current_ply})")
                        print(f"Using opening book move: {mv.uci()}")
            
            move = mv
            # Track position after opening move
            test_board = ctx.board.copy()
            test_board.push(move)
            engine._recent_positions.append(test_board.fen())
            if len(engine._recent_positions) > 10:
                engine._recent_positions.pop(0)
            return move
        
        # Use model/MCTS for non-opening positions
        # NEW: Force greedy policy in critical situations (critical time or losing badly)
        # Check for critical time (very low on clock)
        critical_time = movetime_ms > 0 and movetime_ms < CRITICAL_TIME_SKIP_MCTS_MS
        # Check for losing badly (very negative value)
        losing_badly = value is not None and float(value) < LOSING_BADLY_THRESHOLD
        
        # Force greedy policy if critical time or losing badly
        force_greedy = critical_time or losing_badly
        if force_greedy:
            if critical_time:
                decision_mode = "Greedy policy (critical time)"
                if VERBOSE:
                    print(f"Decision mode: {decision_mode} (movetime_ms={movetime_ms:.0f}ms, ply {current_ply})")
            elif losing_badly:
                decision_mode = "Greedy policy (losing badly)"
                if VERBOSE:
                    print(f"Decision mode: {decision_mode} (value={float(value):+.3f}, ply {current_ply})")
            
            # If we already did NN eval, just use it (avoids redundant choose_move call)
            if not skip_nn_eval and policy_move is not None and policy_move in legal_moves:
                move = policy_move
                if VERBOSE:
                    print(f"Using cached policy_move: {move.uci()}")
            else:
                # Only in very early opening (skip_nn_eval) do we actually need a fresh call
                try:
                    mv, _, _ = choose_move(ctx.board, model, device=device, temperature=0.8, sample=False)
                    move = mv
                    if VERBOSE:
                        print(f"Using greedy policy move: {move.uci()}")
                except Exception as e:
                    # Fallback if choose_move fails
                    print(f"info string Error in greedy selection: {e}", file=sys.stderr)
                    # Fallback to policy_move or first legal move
                    if policy_move is not None and policy_move in legal_moves:
                        move = policy_move
                        print(f"Fallback to policy_move: {move.uci()}")
                    else:
                        move = legal_move_list[0]
                        print(f"Fallback to first legal move: {move.uci()}")
        # NEW: Skip PUCT if early termination is triggered (extreme value magnitude)
        # NEW: Skip MCTS if policy is very confident (policy_confidence_skip)
        # NEW: Skip MCTS if low on time (use greedy policy for speed)
        elif not early_termination and not policy_confidence_skip:
            skip_mcts_for_time = movetime_ms > 0 and movetime_ms < LOW_TIME_SKIP_MCTS_MS
            if not skip_mcts_for_time and engine.use_puct and hasattr(engine, 'mcts_config') and engine.mcts_config is not None:
                decision_mode = "MCTS search"
                print(f"Decision mode: {decision_mode} (ply {current_ply}, phase={game_phase:.2f})")
            
            # Initialize or update search tree
            from .chess_policy.mcts import SearchTree
            if not hasattr(engine, 'search_tree') or engine.search_tree is None:
                # No existing tree - create new one
                engine.search_tree = SearchTree(ctx.board, engine.model, engine.mcts_config)
            else:
                # Check if tree matches current position
                try:
                    if engine.search_tree.root_state.fen() != ctx.board.fen():
                        # Position changed - try to reuse tree if we can update it
                        # For now, create new tree (could be improved to check if position is one move away)
                        engine.search_tree = SearchTree(ctx.board, engine.model, engine.mcts_config)
                except Exception:
                    # If comparison fails, create new tree
                    engine.search_tree = SearchTree(ctx.board, engine.model, engine.mcts_config)
            
            # Update config with current sims
            engine.mcts_config.n_simulations = sims
            
            # Search for best move
            search_start = time.monotonic()
            mv, visit_dist, root_value = engine.search_tree.search(max_simulations_override=sims)
            search_time = (time.monotonic() - search_start) * 1000
            if VERBOSE:
                print(f"Search: {search_time:.1f} ms (sims={sims})")
            
            # Policy trust check: if top policy move has much higher probability than MCTS choice,
            # and MCTS choice has low policy probability, trust the policy instead
            # TUNE: Adjust these thresholds if you see blunders:
            #   - Lower top_policy_prob threshold (0.10-0.12) = more aggressive override
            #   - Raise mcts_policy_prob threshold (0.06-0.08) = catch more bad MCTS choices
            #   - Lower multiplier (2.5-2.8) = override when difference is smaller
            if mv is not None and policy_move is not None:
                mcts_policy_prob = move_probs.get(mv, 0.0) if move_probs else 0.0
                top_policy_prob = move_probs.get(policy_move, 0.0) if move_probs else 0.0
                
                # TUNE: Optimized for 1-min games - more aggressive override
                # Current thresholds: top > 12%, MCTS < 6%, top is 2.5x higher
                # More aggressive than before to catch blunders faster in quick games
                # Make even more aggressive: top > 0.10, MCTS < 0.08, multiplier > 2.0
                # Make less aggressive: top > 0.20, MCTS < 0.04, multiplier > 4.0
                if top_policy_prob > 0.12 and mcts_policy_prob < 0.06 and top_policy_prob > mcts_policy_prob * 2.5:
                    # Calculate ratio safely (avoid division by zero)
                    if mcts_policy_prob > 0:
                        ratio = top_policy_prob / mcts_policy_prob
                        ratio_str = f"{ratio:.1f}x higher"
                    else:
                        ratio_str = "infinitely higher (MCTS prob=0)"
                    if VERBOSE:
                        print(f"Policy trust override: MCTS chose {mv.uci()} (prob={mcts_policy_prob:.4f}), "
                              f"but policy top move {policy_move.uci()} has prob={top_policy_prob:.4f} "
                              f"({ratio_str}). Using policy move.")
                    mv = policy_move
                    decision_mode = "Policy trust override"
            
            # If move would lead to a position we've seen recently, try to avoid it
            if mv is not None:
                test_board = ctx.board.copy()
                test_board.push(mv)
                test_fen = test_board.fen()
                # If this position appeared in last 3 moves, try a different move
                if test_fen in engine._recent_positions[-3:]:
                    print("Warning: Best move leads to recent repetition, re-searching with more sims...")
                    # Re-search with more simulations to potentially get different move
                    mv, _, _ = engine.search_tree.search(max_simulations_override=sims * 2)
            
            # Log root value prediction (from current player's perspective)
            if VERBOSE and hasattr(engine.search_tree, 'root') and engine.search_tree.root.visit_count > 0:
                v = engine.search_tree.root.q_value
                print(f"info string root_value {v:.3f}")
            
            move = mv
            
            # Update search tree for next move (tree reuse)
            # This allows the next search to reuse the subtree under the played move
            if move is not None and hasattr(engine, 'search_tree') and engine.search_tree is not None:
                try:
                    engine.search_tree.update_root(move)
                except Exception as e:
                    # If update fails, tree will be recreated on next search
                    print(f"Warning: Failed to update search tree: {e}")
                    engine.search_tree = None
        else:
            # Use greedy policy selection (fast, no search)
            # NEW: Handle early termination and policy confidence cases with special logging
            if early_termination or policy_confidence_skip:
                if early_termination:
                    decision_mode = "Greedy policy (early termination)"
                    if VERBOSE:
                        print(f"Decision mode: {decision_mode} (value={float(value):+.3f}, ply {current_ply})")
                else:
                    decision_mode = "Greedy policy (high confidence)"
                    if VERBOSE:
                        top_prob = move_probs.get(policy_move, 0.0) if move_probs else 0.0
                        print(f"Decision mode: {decision_mode} (top prob={top_prob*100:.1f}%, ply {current_ply})")
                # Prefer sorted_moves[0][0] if available, otherwise fallback to policy_move
                if sorted_moves and len(sorted_moves) > 0:
                    move = sorted_moves[0][0]
                    if VERBOSE:
                        print(f"Using top policy move from sorted_moves: {move.uci()}")
                elif policy_move is not None and policy_move in legal_moves:
                    move = policy_move
                    if VERBOSE:
                        print(f"Using policy_move: {move.uci()}")
                else:
                    move = legal_move_list[0]
                    if VERBOSE:
                        print(f"Fallback to first legal move: {move.uci()}")
            elif skip_mcts_for_time:
                decision_mode = "Greedy policy (low time)"
                if VERBOSE:
                    print(f"Decision mode: {decision_mode} (movetime_ms={movetime_ms}, ply {current_ply})")
                # If we already did NN eval, just use it (avoids redundant choose_move call)
                if not skip_nn_eval and policy_move is not None and policy_move in legal_moves:
                    move = policy_move
                    if VERBOSE:
                        print(f"Using cached policy_move: {move.uci()}")
                else:
                    try:
                        mv, _, _ = choose_move(ctx.board, model, device=device, temperature=0.8, sample=False)
                        move = mv
                    except Exception as e:
                        # Fallback if choose_move fails
                        print(f"info string Error in greedy selection: {e}", file=sys.stderr)
                        move = next(iter(ctx.board.legal_moves), None)
            else:
                decision_mode = "Greedy policy"
                if VERBOSE:
                    print(f"Decision mode: {decision_mode} (ply {current_ply})")
                # If we already did NN eval, just use it (avoids redundant choose_move call)
                if not skip_nn_eval and policy_move is not None and policy_move in legal_moves:
                    move = policy_move
                    if VERBOSE:
                        print(f"Using cached policy_move: {move.uci()}")
                else:
                    try:
                        mv, _, _ = choose_move(ctx.board, model, device=device, temperature=0.8, sample=False)
                        move = mv
                    except Exception as e:
                        # Fallback if choose_move fails
                        print(f"info string Error in greedy selection: {e}", file=sys.stderr)
                        move = next(iter(ctx.board.legal_moves), None)
        
        # Fallback if selected move is invalid
        if move is None or move not in legal_moves:
            print("Warning: Selected move invalid, falling back to policy_move / first legal")
            if policy_move is not None and policy_move in legal_moves:
                move = policy_move
            else:
                move = legal_move_list[0]
        
        # Last resort: pick first legal move
        if move is None or move not in legal_moves:
            print("Warning: Using first legal move as last resort")
            move = legal_move_list[0]
        
        # Track position after move to detect repetition
        if move is not None:
            test_board = ctx.board.copy()
            test_board.push(move)
            engine._recent_positions.append(test_board.fen())
            # Keep only last 10 positions
            if len(engine._recent_positions) > 10:
                engine._recent_positions.pop(0)
        
        # Final logging
        if VERBOSE:
            print(f"Final decision mode: {decision_mode}")
            print(f"Selected move: {move.uci()}")
            
            # Log position assessment using NN value
            if value is not None:
                print("Position assessment:", format_value_eval(float(value)))
            
            # Log probability for debugging (from our extracted move_probs, which is best-effort)
            if move_probs and move in move_probs:
                move_prob = move_probs[move]
                print(f"Move probability (from extracted probs): {move_prob:.4f} ({move_prob*100:.2f}%)")
        
        # Print move time if debug is enabled
        if DEBUG_MOVE_TIME and move_start_time is not None:
            move_time_ms = (time.monotonic() - move_start_time) * 1000
            print(f"[DEBUG_MOVE_TIME] Total move time: {move_time_ms:.1f} ms (ply {current_ply})")
        
        return move
        
    except Exception as e:
        print(f"Error in move selection: {e}, falling back to policy move")
        import traceback
        traceback.print_exc()
        # Fallback to the policy move we already computed
        move = policy_move if policy_move is not None and policy_move in legal_moves else legal_move_list[0]
        print(f"Selected move (exception fallback): {move.uci()}")
        
        # Print move time if debug is enabled (even on exception)
        if DEBUG_MOVE_TIME and move_start_time is not None:
            move_time_ms = (time.monotonic() - move_start_time) * 1000
            current_ply = len(ctx.board.move_stack) if hasattr(ctx, 'board') else 0
            print(f"[DEBUG_MOVE_TIME] Total move time (exception): {move_time_ms:.1f} ms (ply {current_ply})")
        
        return move


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    print("Resetting chess engine for new game...")
    if engine is not None:
        engine.ucinewgame()  # This resets the board and clears caches
    else:
        print("Warning: engine is None in reset_func; skipping ucinewgame()")
    # Clear the FEN cache to avoid memory buildup
    _cached_nn_eval.cache_clear()
    print("Chess engine reset complete")
