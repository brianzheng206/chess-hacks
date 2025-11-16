from .utils import chess_manager, GameContext
from chess import Move, Board
import sys
import os
import io
import contextlib
import time
from functools import lru_cache
import numpy as np
import chess

# Note: This code uses Python 3.10+ type hint syntax (int | None, tuple[...] | None).
# For Python 3.9 compatibility, either:
#   - Use Optional[int] and Optional[Tuple[...]] from typing, or
#   - Add: from __future__ import annotations

# Compatibility helper for python-chess transposition_key
# Newer python-chess versions have transposition_key as a property (int)
# Older versions may have it as a method or attribute
def _get_transposition_key(board: Board) -> int | None:
    """Get transposition key from board, handling both property and method cases.
    
    Returns:
        int if transposition_key is available, None otherwise.
    """
    if not hasattr(board, 'transposition_key'):
        return None
    
    try:
        key = board.transposition_key
        # If it's callable, it's a method (older python-chess)
        if callable(key):
            key = key()
        # Convert to int (should already be int, but ensure it)
        return int(key)
    except (TypeError, AttributeError, ValueError):
        # Fallback: try _transposition_key attribute (some versions)
        try:
            if hasattr(board, '_transposition_key'):
                return int(board._transposition_key)
        except (AttributeError, ValueError):
            pass
    return None

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

# Early termination constants for value-based decision skipping - AGGRESSIVELY OPTIMIZED FOR SPEED
VALUE_EARLY_TERMINATION_THRESHOLD = 0.45  # abs(value) above this → skip PUCT (lowered from 0.60 for maximum speed)
VALUE_EARLY_TERMINATION_MIN_PLY = 0       # allow immediate termination

# Policy confidence skip: if enabled, skip MCTS when policy is very confident
# Currently disabled for accuracy - with ~50 sims budget, the extra search is worth it
# To re-enable: set ENABLE_POLICY_CONFIDENCE_SKIP = True and adjust POLICY_CONFIDENCE_THRESHOLD
ENABLE_POLICY_CONFIDENCE_SKIP = False  # Disabled for better accuracy
POLICY_CONFIDENCE_THRESHOLD = 0.55  # If re-enabled, use 0.55 instead of 0.40 for safety

# Debug flags - set to True only when debugging (significantly impacts performance)
DEBUG_POLICY = False  # Enable verbose policy extraction and diagnostics
DEBUG_TENSOR = False  # Enable raw tensor diagnostics
DEBUG_MOVE_TIME = False  # Measure and print move time

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

# Time management flags - AGGRESSIVELY OPTIMIZED FOR SPEED
ULTRA_FAST_MODE_THRESHOLD_MS = 10000  # Ultra-fast mode: < 10 seconds, use first legal move if no cache (maximum speed)
INSTANT_MODE_THRESHOLD_MS = 20000  # Instant mode: skip everything, use fastest possible move (20 seconds - optimized for 10s target)
LOW_TIME_SKIP_MCTS_MS = 15000  # Skip MCTS entirely when time drops below this (15 seconds - optimized for 10s target)
CRITICAL_TIME_SKIP_MCTS_MS = 12000  # Force greedy policy when time is critically low (12 seconds - optimized for 10s target)
LOSING_BADLY_THRESHOLD = -0.6  # Force greedy policy when losing very badly (value < -0.6, very aggressive)

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
            sims=20,  # Reduced from 30 for maximum speed (will be adjusted by time management)
            c_puct=0.2,  # Reduced from 0.3 to trust policy heavily, very fast convergence (aggressive speed)
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

# Global cache for neural network evaluations
# This avoids re-evaluating the same position multiple times (transpositions, repetitions)
# Uses transposition_key (property or method) if available (fast & collision-resistant), falls back to FEN
_nn_eval_cache: dict = {}
_nn_eval_cache_maxsize = 20000

def _position_key_from_state(state) -> str:
    """Get cache key for a position from any state object (Board, GameState, etc.).
    
    Uses transposition_key if available (fast & collision-resistant), else FEN.
    Works directly with the state object without constructing new Board objects.
    
    Args:
        state: Any object with transposition_key (property or method) and/or fen() method.
    
    Returns:
        String key with prefix "tt:" for transposition keys or "fen:" for FEN strings.
        This avoids collisions between transposition keys and FEN strings.
    """
    # Try transposition_key first (faster)
    # Handle both property and method cases directly (mirror _get_transposition_key logic)
    if hasattr(state, 'transposition_key'):
        try:
            key = state.transposition_key
            # If it's callable, it's a method (older python-chess)
            if callable(key):
                key = key()
            # Convert to int (should already be int, but ensure it)
            tt_key = int(key)
            return f"tt:{tt_key}"
        except (TypeError, AttributeError, ValueError):
            # Fallback: try _transposition_key attribute (some versions)
            try:
                if hasattr(state, '_transposition_key'):
                    tt_key = int(state._transposition_key)
                    return f"tt:{tt_key}"
            except (AttributeError, ValueError):
                pass
    
    # Fallback to FEN
    if hasattr(state, 'fen'):
        try:
            return f"fen:{state.fen()}"
        except Exception:
            pass
    
    # Last resort: try to convert to string (shouldn't happen in practice)
    return f"fen:{str(state)}"

def _get_position_key(board: Board) -> str:
    """Get cache key for a Board object. Wrapper around _position_key_from_state for backward compatibility."""
    return _position_key_from_state(board)

def _search_tree_matches_board(search_tree, board: Board) -> bool:
    """Check if a search tree's root state matches the given board position.
    
    Uses transposition_key if available for faster comparison, falls back to FEN comparison.
    
    Args:
        search_tree: SearchTree instance with root_state attribute.
        board: Board position to compare against.
        
    Returns:
        True if the tree's root state matches the board position, False otherwise.
    """
    try:
        # Use transposition_key if available for faster comparison
        root_tt_key = _get_transposition_key(search_tree.root_state)
        board_tt_key = _get_transposition_key(board)
        if root_tt_key is not None and board_tt_key is not None:
            return root_tt_key == board_tt_key
        
        # Fallback to FEN comparison
        return search_tree.root_state.fen() == board.fen()
    except Exception:
        return False

def _cached_nn_eval(board: Board) -> tuple[torch.Tensor, float] | None:
    """Get cached neural network evaluation for a position, or None if not cached.
    
    Returns:
        Tuple of (policy_logits, value) if cached, None otherwise.
        policy_logits: [POLICY_SIZE] tensor on CPU
        value: float in [-1, 1]
    """
    cache_key = _get_position_key(board)
    return _nn_eval_cache.get(cache_key)

def _cache_nn_eval(board: Board, policy_logits: torch.Tensor, value: float) -> None:
    """Cache neural network evaluation for a position.
    
    Args:
        board: Chess board position
        policy_logits: Policy logits tensor [POLICY_SIZE] (will be moved to CPU)
        value: Value prediction in [-1, 1]
    
    Note:
        Eviction is FIFO (First In First Out), not true LRU. Since CPython dicts are
        insertion-ordered, we remove the oldest entry. Access doesn't bump recency.
        This is effectively a ring buffer of recent positions, which is fine for chess
        where we want to keep recent positions cached. For true LRU, use collections.OrderedDict
        or functools.lru_cache, but FIFO is usually sufficient.
    """
    cache_key = _get_position_key(board)
    # Ensure logits are on CPU for caching
    if policy_logits.device.type != 'cpu':
        policy_logits = policy_logits.cpu()
    
    # FIFO eviction: if cache is full, remove oldest entry (first in insertion order)
    # This is effectively a ring buffer of recent positions, not true LRU
    if len(_nn_eval_cache) >= _nn_eval_cache_maxsize:
        # Remove first (oldest) entry
        oldest_key = next(iter(_nn_eval_cache))
        del _nn_eval_cache[oldest_key]
    
    _nn_eval_cache[cache_key] = (policy_logits, float(value))

def _clear_nn_eval_cache() -> None:
    """Clear the neural network evaluation cache."""
    global _nn_eval_cache
    _nn_eval_cache.clear()

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
            - abs(value) near 1 → clearly winning/losing → scale ~0.10 (very reduced search)
            - For losing positions (value < -0.3), use even more aggressive scaling
    
    Returns:
        Scale factor in [0.10, 1.0] for adjusting simulation count (aggressively optimized for speed).
    """
    abs_value = abs(float(value))
    value_float = float(value)
    
    # Very aggressive scaling for maximum speed - reduce search significantly for clear positions
    if value_float < -0.3:  # Losing position
        # For losing positions, reduce aggressively
        if abs_value >= 0.85:
            scale = 0.10  # Very aggressive for clearly losing positions
        elif abs_value >= 0.5:
            scale = 0.20  # Aggressive for moderately losing positions
        else:
            scale = 0.35  # Moderate for slightly losing positions
    elif abs_value >= 0.85:  # Very winning/losing (but winning)
        scale = 0.15  # Aggressive for clearly winning positions
    else:
        # Linear mapping: abs_value 0.0 → scale 1.0, abs_value 0.85 → scale ~0.15
        # More aggressive than before for maximum speed
        scale = 1.0 - 1.0 * abs_value
    
    # Clamp to [0.10, 1.0] to ensure reasonable bounds (lowered minimum from 0.25)
    return max(0.10, min(1.0, scale))


# Piece values for tactical evaluation
PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def find_obvious_tactic(board: Board, legal_moves) -> Move | None:
    """Find obvious winning captures (simple material gain heuristic).
    
    This is a cheap, fast check for obvious tactical wins like:
    - Winning a rook/queen for free
    - Winning a minor piece for free (minor for pawn, rook for minor, etc.)
    - Any capture with net gain >= 2 pawns
    
    Checks if the capturing piece would be immediately recaptured to avoid
    "win a rook, instantly lose the queen" type disasters.
    
    Args:
        board: Current chess board position
        legal_moves: List of legal moves to check
        
    Returns:
        Best winning capture move if found, None otherwise
    """
    best_move = None
    best_gain = 0  # in pawns
    
    for mv in legal_moves:
        if not board.is_capture(mv):
            continue
        
        piece = board.piece_at(mv.from_square)
        captured = board.piece_at(mv.to_square)
        if piece is None or captured is None:
            continue
        
        # Calculate material gain: captured piece value - our piece value
        gain = PIECE_VALUE.get(captured.piece_type, 0) - PIECE_VALUE.get(piece.piece_type, 0)
        
        # Only care about big-ish swings (gain >= 2: minor for pawn, rook for minor, etc.)
        if gain < 2:
            continue
        
        # Check if the destination square is defended by the opponent after the capture
        # This avoids "win a rook, instantly lose the queen" type disasters
        board.push(mv)
        try:
            # Is our capturing piece hanging on the new square?
            attackers = board.attackers(not board.turn, mv.to_square)
            if attackers:
                # Capturing piece would be recaptured immediately; skip this as "obvious tactic"
                continue
        finally:
            board.pop()
        
        # Found a good capture that won't be immediately recaptured
        if gain > best_gain:
            best_gain = gain
            best_move = mv
    
    return best_move


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

    # Get time early to check for ultra-fast mode
    movetime_ms_early = ctx.timeLeft if ctx.timeLeft and ctx.timeLeft > 0 else 0
    
    # OPTIMIZATION: Skip tactical checks in ultra-fast mode (< 10 seconds) for maximum speed
    skip_tactical_for_time = movetime_ms_early > 0 and movetime_ms_early < ULTRA_FAST_MODE_THRESHOLD_MS
    
    # 0. Immediate tactical wins: never miss mate-in-one
    # This is the single highest-impact, lowest-cost accuracy boost
    # Cost: O(#legal_moves) with one push/pop each – very cheap
    # Effect: engine will never miss a mate in one, regardless of what the net/MCTS think
    # OPTIMIZATION: Skip in ultra-fast mode for maximum speed
    if not skip_tactical_for_time:
        board = ctx.board
        for mv in legal_moves:
            board.push(mv)
            if board.is_checkmate():
                board.pop()
                if VERBOSE:
                    print(f"Forced mate in 1 found: {mv.uci()}")
                ctx.logProbabilities({mv: 1.0})
                return mv
            board.pop()

    # 1. Simple tactical scan: find obvious winning captures (all phases)
    # This catches "missed free rook/queen" type positions cheaply
    # Cost: O(#legal_moves) - very cheap
    # Effect: engine will never miss obvious winning captures
    # OPTIMIZATION: Skip tactical scan in early post-opening and ultra-fast mode for maximum speed
    if not skip_tactical_for_time:
        current_ply_for_tactical = len(ctx.board.move_stack)
        if not (current_ply_for_tactical >= 8 and current_ply_for_tactical < 18):
            tactical_mv = find_obvious_tactic(ctx.board, legal_moves)
            if tactical_mv is not None:
                if VERBOSE:
                    print(f"Taking obvious winning capture: {tactical_mv.uci()}")
                ctx.logProbabilities({tactical_mv: 1.0})
                return tactical_mv

    # Check if engine is initialized
    if engine is None or model is None:
        print("ERROR: Engine or model not initialized - cannot make move")
        # Fallback: return first legal move
        move = legal_moves[0]
        move_probs = {move: 1.0}
        ctx.logProbabilities(move_probs)
        return move
    
    # NOTE: ctx.timeLeft comes from the platform (ChessHacks) via GameContext, not our own timer
    movetime_ms = movetime_ms_early  # Use the early check we did above
    
    # ULTRA-FAST MODE: When time is < 10 seconds, use absolute fastest path
    # Skip ALL expensive operations: NN eval, MCTS, opening book, tactical checks
    # Just use cached eval if available, otherwise use first legal move
    if movetime_ms > 0 and movetime_ms < ULTRA_FAST_MODE_THRESHOLD_MS:
        if VERBOSE:
            print(f"ULTRA-FAST MODE: timeLeft={movetime_ms:.0f}ms < {ULTRA_FAST_MODE_THRESHOLD_MS}ms - using fastest possible move")
        
        # Try cached evaluation first (fastest possible)
        cached_result = _cached_nn_eval(ctx.board)
        if cached_result is not None:
            # Cache hit - use it instantly
            root_policy_logits, root_value = cached_result
            from .chess_policy.encoding import legal_mask_4672
            from .chess_policy.infer import mask_logits, probs_from_logits
            from .chess_policy.move_index import index_to_move, POLICY_SIZE
            
            logits = root_policy_logits.to(device)
            legal = torch.from_numpy(legal_mask_4672(ctx.board)).to(logits.device)
            masked = mask_logits(logits, legal)
            probs = probs_from_logits(masked, temperature=0.0)  # Deterministic
            
            idx = int(torch.argmax(probs).item())
            move = index_to_move(ctx.board, idx) if 0 <= idx < POLICY_SIZE else None
            
            if move is None or move not in legal_moves:
                # Fallback: try sorted moves (but limit iterations for speed)
                order = torch.argsort(probs, descending=True).tolist()[:10]  # Only check top 10
                for i in order:
                    if 0 <= i < POLICY_SIZE:
                        mv_try = index_to_move(ctx.board, int(i))
                        if mv_try is not None and mv_try in legal_moves:
                            move = mv_try
                            break
                if move is None or move not in legal_moves:
                    move = legal_moves[0]
            
            move_probs = {move: 1.0}
            ctx.logProbabilities(move_probs)
            if VERBOSE:
                print(f"Ultra-fast move (cached): {move.uci()}")
            return move
        
        # No cache - use first legal move (fastest possible, no NN eval)
        move = legal_moves[0]
        move_probs = {move: 1.0}
        ctx.logProbabilities(move_probs)
        if VERBOSE:
            print(f"Ultra-fast move (first legal, no cache): {move.uci()}")
        return move
    
    # INSTANT MODE: When time is low (< 20 seconds), make moves instantly
    # Skip all expensive operations: MCTS, opening book comparisons, etc.
    # Just use cached NN eval if available, or do a quick NN eval, then pick top move
    if movetime_ms > 0 and movetime_ms < INSTANT_MODE_THRESHOLD_MS:
        if VERBOSE:
            print(f"INSTANT MODE: timeLeft={movetime_ms:.0f}ms < {INSTANT_MODE_THRESHOLD_MS}ms - making instant move")
        
        # Try to get cached evaluation first (fastest)
        cached_result = _cached_nn_eval(ctx.board)
        if cached_result is not None:
            # Cache hit - use it instantly
            root_policy_logits, root_value = cached_result
            from .chess_policy.encoding import legal_mask_4672
            from .chess_policy.infer import mask_logits, probs_from_logits
            from .chess_policy.move_index import index_to_move, POLICY_SIZE
            
            logits = root_policy_logits.to(device)
            legal = torch.from_numpy(legal_mask_4672(ctx.board)).to(logits.device)
            masked = mask_logits(logits, legal)
            probs = probs_from_logits(masked, temperature=0.0)  # Deterministic
            
            idx = int(torch.argmax(probs).item())
            move = index_to_move(ctx.board, idx) if 0 <= idx < POLICY_SIZE else None
            
            if move is None or move not in legal_moves:
                # Fallback: try sorted moves
                order = torch.argsort(probs, descending=True).tolist()
                for i in order:
                    if 0 <= i < POLICY_SIZE:
                        mv_try = index_to_move(ctx.board, int(i))
                        if mv_try is not None and mv_try in legal_moves:
                            move = mv_try
                            break
                if move is None or move not in legal_moves:
                    move = legal_moves[0]
            
            move_probs = {move: 1.0}
            ctx.logProbabilities(move_probs)
            if VERBOSE:
                print(f"Instant move (cached): {move.uci()}")
            return move
        
        # No cache - do a quick NN eval (still fast, just one forward pass)
        try:
            from .chess_policy.encoding import board_to_tensor, legal_mask_4672
            from .chess_policy.infer import unpack_policy, mask_logits, probs_from_logits
            from .chess_policy.move_index import index_to_move, POLICY_SIZE
            
            # Quick NN forward pass
            x_np = board_to_tensor(ctx.board)
            c_encoded = x_np.shape[0]
            try:
                c_model = int(getattr(model, "in_channels", c_encoded))
            except Exception:
                c_model = c_encoded
            if c_encoded > c_model:
                x_np = x_np[:c_model]
            elif c_encoded < c_model:
                pad = np.zeros((c_model - c_encoded, x_np.shape[1], x_np.shape[2]), dtype=x_np.dtype)
                x_np = np.concatenate([x_np, pad], axis=0)
            
            x = torch.from_numpy(x_np).unsqueeze(0).to(device)
            model.eval()
            with torch.inference_mode():
                output = model(x)
                logits_b, v_pred = unpack_policy(output)
                logits = logits_b[0] if logits_b.dim() == 2 and logits_b.size(0) == 1 else logits_b
            
            # Get top move instantly
            legal = torch.from_numpy(legal_mask_4672(ctx.board)).to(logits.device)
            masked = mask_logits(logits, legal)
            probs = probs_from_logits(masked, temperature=0.0)  # Deterministic
            
            idx = int(torch.argmax(probs).item())
            move = index_to_move(ctx.board, idx) if 0 <= idx < POLICY_SIZE else None
            
            if move is None or move not in legal_moves:
                order = torch.argsort(probs, descending=True).tolist()
                for i in order:
                    if 0 <= i < POLICY_SIZE:
                        mv_try = index_to_move(ctx.board, int(i))
                        if mv_try is not None and mv_try in legal_moves:
                            move = mv_try
                            break
                if move is None or move not in legal_moves:
                    move = legal_moves[0]
            
            # Cache the evaluation for future instant moves (if we see this position again)
            value = 0.0
            if v_pred is not None:
                v = v_pred
                if v.dim() == 2 and v.size(0) == 1:
                    v = v[0]
                value = float(v.squeeze(-1).cpu().item())
            _cache_nn_eval(ctx.board, logits.detach().cpu(), value)
            
            move_probs = {move: 1.0}
            ctx.logProbabilities(move_probs)
            if VERBOSE:
                print(f"Instant move (quick NN): {move.uci()}")
            return move
        except Exception as e:
            # If even NN eval fails, just return first legal move
            if VERBOSE:
                print(f"Instant mode fallback: {e}")
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
    # OPTIMIZATION: Aggressively reuse tree by checking if opponent's move leads to a position in the tree
    if hasattr(engine, 'search_tree') and engine.search_tree is not None:
        try:
            tree_matches = _search_tree_matches_board(engine.search_tree, ctx.board)
            
            if not tree_matches:
                # Position changed - could be opponent's move
                # Try aggressive tree reuse: check if current position is reachable from the tree
                if not engine.search_tree.update_root_to_position(ctx.board):
                    # Position not in tree - will be handled below when creating/updating tree
                    # Don't set to None here, let the code below handle it with precomputed root evaluation
                    pass
        except Exception:
            # If check fails, tree will be recreated below if needed
            pass
    
    # Convert to list for easier access
    legal_move_list = list(legal_moves)
    
    # Calculate current ply for opening book check
    current_ply = len(ctx.board.move_stack)
    
    # OPTIMIZATION: For very early opening (first 4 moves), skip NN evaluation
    # and go straight to opening book for maximum speed
    # Also skip NN eval for early post-opening (ply 8-18) to use pure policy for maximum speed
    # OPTIMIZATION: Skip expensive tactical checks in early post-opening for speed
    skip_nn_eval = current_ply < 4 or (current_ply >= 8 and current_ply < 18)
    skip_tactical_checks = current_ply >= 8 and current_ply < 18  # Skip tactical scan in early post-opening
    
    # -------------------------
    # STEP 1: POLICY EVALUATION (skip for very early opening)
    # -------------------------
    policy_move = None
    probs_tensor = None
    value = None
    root_policy_logits = None  # Store raw logits for SearchTree to avoid redundant NN evaluation
    root_value = None
    sorted_moves = []   # NEW: always defined as list
    move_probs = {}     # NEW: initialize here for logging

    # Only run NN evaluation if not in very early opening or early post-opening
    # Early post-opening (ply 8-12): use pure policy for speed, skip MCTS
    if not skip_nn_eval:
        try:
            # Determine temperature based on game phase
            if current_ply < 12:
                # Opening: use lower temperature for more deterministic play
                temperature = 0.7
            else:
                # Mid/endgame: slightly higher temperature
                temperature = 0.8
            
            # OPTIMIZATION: Check cache first to avoid redundant NN forward pass for repeated positions
            # This catches repetitions, transpositions, and "shuffling" in time trouble
            nn_start = time.monotonic()
            from .chess_policy.encoding import board_to_tensor, legal_mask_4672
            from .chess_policy.infer import unpack_policy, mask_logits, probs_from_logits
            from .chess_policy.move_index import index_to_move, POLICY_SIZE
            
            # Check cache first
            cached_result = _cached_nn_eval(ctx.board)
            if cached_result is not None:
                # Cache hit! Reconstruct policy_move and probs_tensor from cached logits
                root_policy_logits, root_value = cached_result
                value = root_value
                
                # Reconstruct probs_tensor and policy_move from cached logits (cheap operations)
                # Move logits to device for processing
                logits = root_policy_logits.to(device)
                
                # Generate legal mask and compute probabilities for policy move selection
                legal = torch.from_numpy(legal_mask_4672(ctx.board)).to(logits.device)
                masked = mask_logits(logits, legal)
                probs = probs_from_logits(masked, temperature=temperature)
                probs_tensor = probs.detach().cpu()
                
                # Select policy move (argmax)
                idx = int(torch.argmax(probs).item())
                policy_move = index_to_move(ctx.board, idx) if 0 <= idx < POLICY_SIZE else None
                
                # Fallback if move is invalid
                if policy_move is None or policy_move not in legal_moves:
                    order = torch.argsort(probs, descending=True).tolist()
                    for i in order:
                        if 0 <= i < POLICY_SIZE:
                            mv_try = index_to_move(ctx.board, int(i))
                            if mv_try is not None and mv_try in legal_moves:
                                policy_move = mv_try
                                break
                    if policy_move is None or policy_move not in legal_moves:
                        policy_move = legal_move_list[0]
                
                if VERBOSE:
                    print(f"Root eval: cache hit (reconstructed from cached logits)")
            else:
                # Cache miss: do full forward pass
                # Encode board and run model once
                x_np = board_to_tensor(ctx.board)
                c_encoded = x_np.shape[0]
                try:
                    c_model = int(getattr(model, "in_channels", c_encoded))
                except Exception:
                    c_model = c_encoded
                if c_encoded > c_model:
                    x_np = x_np[:c_model]
                elif c_encoded < c_model:
                    pad = np.zeros((c_model - c_encoded, x_np.shape[1], x_np.shape[2]), dtype=x_np.dtype)
                    x_np = np.concatenate([x_np, pad], axis=0)
                
                x = torch.from_numpy(x_np).unsqueeze(0).to(device)
                model.eval()
                with torch.inference_mode():
                    output = model(x)
                    logits_b, v_pred = unpack_policy(output)
                    logits = logits_b[0] if logits_b.dim() == 2 and logits_b.size(0) == 1 else logits_b
                    
                    # Extract value
                    value = 0.0
                    if v_pred is not None:
                        v = v_pred
                        if v.dim() == 2 and v.size(0) == 1:
                            v = v[0]
                        value = float(v.squeeze(-1).cpu().item())
                
                # Generate legal mask and compute probabilities for policy move selection
                legal = torch.from_numpy(legal_mask_4672(ctx.board)).to(logits.device)
                masked = mask_logits(logits, legal)
                probs = probs_from_logits(masked, temperature=temperature)
                probs_tensor = probs.detach().cpu()
                
                # Select policy move (argmax)
                idx = int(torch.argmax(probs).item())
                policy_move = index_to_move(ctx.board, idx) if 0 <= idx < POLICY_SIZE else None
                
                # Fallback if move is invalid
                if policy_move is None or policy_move not in legal_moves:
                    order = torch.argsort(probs, descending=True).tolist()
                    for i in order:
                        if 0 <= i < POLICY_SIZE:
                            mv_try = index_to_move(ctx.board, int(i))
                            if mv_try is not None and mv_try in legal_moves:
                                policy_move = mv_try
                                break
                    if policy_move is None or policy_move not in legal_moves:
                        policy_move = legal_move_list[0]
                
                # Store raw logits for passing to SearchTree (avoid redundant NN evaluation)
                root_policy_logits = logits.detach().cpu()  # Store on CPU, will move to device in SearchTree if needed
                root_value = value
                
                # Cache this evaluation for future use (transpositions, repetitions)
                _cache_nn_eval(ctx.board, root_policy_logits, root_value)
            
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
            # NOTE: ctx.timeLeft comes from the platform (ChessHacks) via GameContext, not our own timer
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
    
    # Compute tactical indicators (checks, captures, promotions)
    # Tactical positions require full search - don't skip MCTS in these positions
    num_check_moves = sum(1 for m in legal_move_list if ctx.board.gives_check(m))
    in_check = ctx.board.is_check()
    num_captures = sum(1 for m in legal_move_list if ctx.board.is_capture(m))
    num_promotions = sum(1 for m in legal_move_list if m.promotion is not None)
    
    # Position is tactically heavy if there are checks, many captures, or promotions
    tactical_heavy = in_check or num_check_moves > 1 or num_captures > 1 or num_promotions > 0
    
    if VERBOSE:
        print(f"Tactical indicators: checks={num_check_moves}, in_check={in_check}, "
              f"captures={num_captures}, promotions={num_promotions}, tactical_heavy={tactical_heavy}")
    
    # Policy confidence check: if top policy move has very high probability, skip MCTS
    # BUT: never skip in tactical positions where the net might be overconfident
    # Currently disabled for accuracy - with ~50 sims budget, the extra search is worth it
    if ENABLE_POLICY_CONFIDENCE_SKIP and move_probs and policy_move is not None:
        top_policy_prob = move_probs.get(policy_move, 0.0)
        # If top move has high probability (>55% for safety) and no tactical complexity, trust policy
        if top_policy_prob > POLICY_CONFIDENCE_THRESHOLD and not tactical_heavy:
            policy_confidence_skip = True
            if VERBOSE:
                print(f"Policy confidence skip: top move has {top_policy_prob*100:.1f}% probability")
        elif tactical_heavy and VERBOSE:
            print(f"Skipping policy confidence skip due to tactical complexity (checks/captures/promotions)")
    
    # Early termination conditions:
    # 1. Value must be available
    # 2. Must be past minimum ply
    # 3. Value magnitude must be extreme
    # 4. Not in a tactically heavy position (checks, captures, promotions)
    if value is not None and current_ply >= VALUE_EARLY_TERMINATION_MIN_PLY:
        if abs(float(value)) >= VALUE_EARLY_TERMINATION_THRESHOLD:
            if not tactical_heavy:
                early_termination = True
                if VERBOSE:
                    print(f"Early termination triggered (value={float(value):+.3f})")
            elif VERBOSE:
                print("Skipping early termination due to tactical complexity (checks/captures/promotions)")
    
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
    # NOTE: ctx.timeLeft comes from the platform (ChessHacks) via GameContext, not our own timer
    movetime_ms = ctx.timeLeft if ctx.timeLeft and ctx.timeLeft > 0 else 0
    
    if movetime_ms > 0:
        # Dynamic time management: adapt simulations based on available time
        # Optimized for 1-minute games: very conservative time usage
        
        # For 1-minute games, be very conservative with time usage
        # Estimate moves remaining (assume ~35-40 moves per game for 1-min)
        # Use 1.5-2% of remaining time per move to ensure we finish the game
        moves_remaining_estimate = max(8, int(movetime_ms / 1200))  # Slightly more conservative
        time_per_move_ms = movetime_ms / max(moves_remaining_estimate, 1)
        
        # Adaptive simulation rate - AGGRESSIVELY OPTIMIZED FOR SPEED (very reduced rates)
        if time_per_move_ms > 2500:  # >2.5 seconds per move: can search more
            sims_per_sec = 60  # Reduced from 80 for maximum speed
        elif time_per_move_ms > 1200:  # 1.2-2.5 seconds: moderate search
            sims_per_sec = 50  # Reduced from 65 for maximum speed
        elif time_per_move_ms > 600:  # 0.6-1.2 seconds: fast search
            sims_per_sec = 40  # Reduced from 50 for maximum speed
        elif time_per_move_ms > 300:  # 0.3-0.6 seconds: very fast
            sims_per_sec = 30  # Reduced from 40 for maximum speed
        else:  # <0.3 seconds: critical, minimal search
            sims_per_sec = 20  # Reduced from 28 for maximum speed
        
        # Use very aggressive time budget for speed - AGGRESSIVELY OPTIMIZED FOR SPEED
        # Use only 30% of estimated time per move to ensure maximum speed
        time_budget_ms = time_per_move_ms * 0.30  # Reduced from 0.40 for maximum speed
        estimated_sims = max(3, int(time_budget_ms * sims_per_sec / 1000.0))  # Reduced min from 6 to 3
        
        # Cap simulations based on time remaining - AGGRESSIVELY OPTIMIZED FOR SPEED
        if movetime_ms > 40000:  # >40 seconds left (early game)
            max_sims = 20  # Reduced from 28 for maximum speed
        elif movetime_ms > 25000:  # 25-40 seconds
            max_sims = 16  # Reduced from 22 for maximum speed
        elif movetime_ms > 15000:  # 15-25 seconds
            max_sims = 12  # Reduced from 18 for maximum speed
        elif movetime_ms > 8000:  # 8-15 seconds
            max_sims = 10  # Reduced from 14 for maximum speed
        elif movetime_ms > 4000:  # 4-8 seconds
            max_sims = 7  # Reduced from 10 for maximum speed
        elif movetime_ms > 2000:  # 2-4 seconds
            max_sims = 5  # Reduced from 8 for maximum speed
        else:  # <2 seconds: critical time
            max_sims = 3  # Reduced from 4 for maximum speed
        
        sims = min(estimated_sims, max_sims)
        
        # Additional time pressure handling - AGGRESSIVELY OPTIMIZED FOR SPEED
        if movetime_ms < 15000:  # Less than 15 seconds
            sims = min(sims, 15)  # Reduced from 20 for maximum speed
        if movetime_ms < 8000:  # Less than 8 seconds
            sims = min(sims, 10)  # Reduced from 14 for maximum speed
        if movetime_ms < 4000:  # Less than 4 seconds
            sims = min(sims, 7)  # Reduced from 10 for maximum speed
        if movetime_ms < 2000:  # Less than 2 seconds
            sims = min(sims, 4)  # Reduced from 6 for maximum speed
        
        # Absolute global cap - AGGRESSIVELY OPTIMIZED FOR SPEED
        sims = min(sims, 18)  # Reduced from 25 for maximum speed
    else:
        # No time info available - use conservative defaults - AGGRESSIVELY OPTIMIZED FOR SPEED
        sims = min(sims, 18)  # Reduced from 25 for maximum speed
    
    # Phase-aware adjustments - AGGRESSIVELY OPTIMIZED FOR SPEED
    # Strategy: Trust policy heavily, minimize search in all phases for maximum speed
    # OPTIMIZATION: Early post-opening (ply 8-20) gets even more aggressive treatment
    early_post_opening = current_ply >= 8 and current_ply < 20
    if early_post_opening:
        # Right after opening book ends, use minimal search or skip entirely
        sims = min(sims, 5)  # Very minimal search for early post-opening (reduced from 8)
        if hasattr(engine, 'mcts_config') and engine.mcts_config is not None:
            engine.mcts_config.c_puct = 0.10  # Very low - trust policy extremely heavily
    elif game_phase > 0.7:  # Opening
        sims = min(sims, 15)  # Reduced from 22 for maximum speed
        if hasattr(engine, 'mcts_config') and engine.mcts_config is not None:
            engine.mcts_config.c_puct = 0.15  # Reduced from 0.25 - trust policy very heavily for speed
    elif game_phase < 0.3:  # Endgame
        # In endgame, be very efficient - trust policy heavily
        if movetime_ms > 0:
            if movetime_ms < 8000:  # Less than 8 seconds: time pressure
                sims = min(sims, 12)  # Reduced from 18 for maximum speed
                if hasattr(engine, 'mcts_config') and engine.mcts_config is not None:
                    engine.mcts_config.c_puct = 0.15  # Trust policy very heavily for speed
            elif movetime_ms < 15000:  # 8-15 seconds: moderate time
                sims = min(sims, 18)  # Reduced from 25 for maximum speed
                if hasattr(engine, 'mcts_config') and engine.mcts_config is not None:
                    engine.mcts_config.c_puct = 0.15  # Trust policy more for speed
            else:  # >15 seconds: can search more
                sims = min(sims, 20)  # Reduced from 30 for maximum speed
                if hasattr(engine, 'mcts_config') and engine.mcts_config is not None:
                    engine.mcts_config.c_puct = 0.15  # Trust policy more for speed
        else:
            # No time info - use safe defaults for endgame
            sims = min(sims, 18)  # Reduced from 25 for maximum speed
            if hasattr(engine, 'mcts_config') and engine.mcts_config is not None:
                engine.mcts_config.c_puct = 0.15
    else:  # Midgame - prioritize speed (AGGRESSIVE FOR PLY 10-14)
        # Midgame: be very fast, trust policy more - AGGRESSIVELY REDUCED FOR SPEED
        if movetime_ms > 0:
            if movetime_ms > 30000:  # Plenty of time: still prioritize speed
                sims = min(sims, 20)  # Reduced from 28 for maximum speed (middlegame optimization)
                if hasattr(engine, 'mcts_config') and engine.mcts_config is not None:
                    engine.mcts_config.c_puct = 0.15  # Reduced from 0.25 - trust policy very heavily
            elif movetime_ms > 15000:  # Moderate time: be fast
                sims = min(sims, 15)  # Reduced from 22 for maximum speed (middlegame optimization)
                if hasattr(engine, 'mcts_config') and engine.mcts_config is not None:
                    engine.mcts_config.c_puct = 0.15  # Reduced from 0.25 - trust policy very heavily
            else:  # Time pressure: minimal search
                sims = min(sims, 12)  # Reduced from 18 for maximum speed (middlegame optimization)
                if hasattr(engine, 'mcts_config') and engine.mcts_config is not None:
                    engine.mcts_config.c_puct = 0.15  # Reduced from 0.25 - trust policy very heavily
        else:
            # No time info - use safe defaults for midgame
            sims = min(sims, 15)  # Reduced from 22 for maximum speed (middlegame optimization)
            if hasattr(engine, 'mcts_config') and engine.mcts_config is not None:
                engine.mcts_config.c_puct = 0.15  # Reduced from 0.25
    
    # ------------------------------
    # VALUE-AWARE SIMS SCALING
    # ------------------------------
    # NEW: Adjust sims based on position clarity (value magnitude)
    # Unclear positions (value ~0) get full search, clear positions (|value| ~1) get reduced search
    # Losing positions get even more aggressive reduction
    # BUT: Don't shrink sims in tactical chaos – we actually need search there
    if value is not None and not early_termination and not tactical_heavy:
        scale = value_to_sims_scale(float(value))
        sims_before = sims
        value_float = float(value)
        abs_value = abs(value_float)
        
        # Reduced minimums for speed - AGGRESSIVELY OPTIMIZED FOR SPEED
        if abs_value >= 0.85:
            min_sims = 1  # Very winning/losing - minimal search for maximum speed
        elif value_float < -0.3:  # Losing position
            min_sims = 1  # Losing - minimal search for maximum speed
        elif abs_value >= 0.5:
            min_sims = 3  # Moderately winning/losing - very reduced for maximum speed
        else:
            min_sims = 6  # Unclear positions - reduced for maximum speed
        
        sims = max(min_sims, int(sims * scale))
        if VERBOSE:
            print(f"Value-aware sims scaling: value={value_float:+.3f}, scale={scale:.2f}, sims {sims_before} → {sims}")
    elif value is not None and tactical_heavy:
        # Reduced bump for tactical chaos - AGGRESSIVELY OPTIMIZED FOR SPEED
        sims_before = sims
        sims = max(sims, 8)  # Reduced from 12 for maximum speed
        if VERBOSE and sims > sims_before:
            print(f"Tactical position: increased sims from {sims_before} to {sims} (tactical_heavy=True)")
    
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
                ctx.board.push(tactical_move)
                try:
                    engine._recent_positions.append(ctx.board.fen())
                    if len(engine._recent_positions) > 10:
                        engine._recent_positions.pop(0)
                finally:
                    ctx.board.pop()
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
                ctx.board.push(mv)
                try:
                    engine._recent_positions.append(ctx.board.fen())
                    if len(engine._recent_positions) > 10:
                        engine._recent_positions.pop(0)
                finally:
                    ctx.board.pop()
                
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
                        n_simulations=25,  # Reduced from 35 for speed
                        c_puct=engine.mcts_config.c_puct,
                        device=engine.device,
                        temperature=0.0,
                        use_dirichlet_noise=False,
                    )
                    # Push move, copy board in new position, then pop
                    ctx.board.push(mv)
                    test_board_book = ctx.board.copy()
                    ctx.board.pop()
                    temp_tree_book = SearchTree(test_board_book, engine.model, quick_config)
                    _, _, value_book_raw = temp_tree_book.search()
                    value_book = -value_book_raw  # Negate: opponent's perspective -> ours
                    
                    ctx.board.push(top_model_move)
                    test_board_model = ctx.board.copy()
                    ctx.board.pop()
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
                        ctx.board.push(move)
                        try:
                            engine._recent_positions.append(ctx.board.fen())
                            if len(engine._recent_positions) > 10:
                                engine._recent_positions.pop(0)
                        finally:
                            ctx.board.pop()
                        return move
                elif top_model_prob > 0.5:
                    # Very high confidence, use it even without PUCT
                    decision_mode = "Model override (very high confidence)"
                    move = top_model_move
                    if VERBOSE:
                        print(f"Decision mode: {decision_mode} (opening, ply {current_ply})")
                        print(f"Model very confident ({top_model_prob*100:.1f}%) in {top_model_move.uci()}, overriding opening book {mv.uci()}")
                    # Track position after move
                    ctx.board.push(move)
                    try:
                        engine._recent_positions.append(ctx.board.fen())
                        if len(engine._recent_positions) > 10:
                            engine._recent_positions.pop(0)
                    finally:
                        ctx.board.pop()
                    return move
            
            # If we have MCTS available, do a quick search to see if there's a better move
            # This catches subtle advantages that the simple tactical check might miss
            # Only do quick search if explicitly enabled (disabled by default for speed)
            if OPENING_QUICK_SEARCH and engine.use_puct and hasattr(engine, 'mcts_config') and engine.mcts_config is not None:
                # Quick search with fewer simulations - OPTIMIZED FOR SPEED
                quick_sims = min(50, max(20, engine.sims // 6))  # Reduced from 30-70 and //5 to //6 for speed
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
                    # Push move, copy board in new position, then pop
                    ctx.board.push(mv)
                    test_board_book = ctx.board.copy()
                    ctx.board.pop()
                    temp_tree_book = SearchTree(test_board_book, engine.model, quick_config)
                    _, _, value_book_raw = temp_tree_book.search()
                    value_book = -value_book_raw  # Negate: opponent's perspective -> ours
                    
                    ctx.board.push(search_move)
                    test_board_search = ctx.board.copy()
                    ctx.board.pop()
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
            ctx.board.push(move)
            try:
                engine._recent_positions.append(ctx.board.fen())
                if len(engine._recent_positions) > 10:
                    engine._recent_positions.pop(0)
            finally:
                ctx.board.pop()
            return move
        
        # Use model/MCTS for non-opening positions
        # OPTIMIZATION: Right after opening (ply 8-20), use pure policy for maximum speed
        # This avoids expensive MCTS right after opening book ends
        early_post_opening = current_ply >= 8 and current_ply < 20
        
        # NEW: Force greedy policy in critical situations (critical time or losing badly)
        # BUT: never skip search in tactical positions where tactics matter
        # Check for critical time (very low on clock)
        critical_time = movetime_ms > 0 and movetime_ms < CRITICAL_TIME_SKIP_MCTS_MS
        # Check for losing badly (very negative value)
        losing_badly = value is not None and float(value) < LOSING_BADLY_THRESHOLD
        
        # Still search in tactical positions even when losing badly - there might be tricks/perpetuals
        # The value head might undervalue tactical opportunities
        if losing_badly and tactical_heavy:
            losing_badly = False  # still search in tactical chaos
            if VERBOSE:
                print(f"Skipping 'losing badly' greedy policy due to tactical complexity "
                      f"(value={float(value):+.3f}, checks/captures/promotions)")
        
        # Force greedy policy if critical time or losing badly, BUT not in tactical positions
        force_greedy = (critical_time or losing_badly) and not tactical_heavy
        if (critical_time or losing_badly) and tactical_heavy and VERBOSE:
            print("Skipping greedy policy due to tactical complexity (checks/captures/promotions)")
        if force_greedy:
            if critical_time:
                decision_mode = "Greedy policy (critical time)"
                if VERBOSE:
                    print(f"Decision mode: {decision_mode} (movetime_ms={movetime_ms:.0f}ms, ply {current_ply})")
            elif losing_badly:
                decision_mode = "Greedy policy (losing badly)"
                if VERBOSE:
                    print(f"Decision mode: {decision_mode} (value={float(value):+.3f}, ply {current_ply})")
            
            # OPTIMIZATION: Always prefer policy_move if available (avoids redundant choose_move call)
            # policy_move is valid if we did NN eval in this function call OR if it was set from a previous call
            if policy_move is not None and policy_move in legal_moves:
                move = policy_move
                if VERBOSE:
                    print(f"Using cached policy_move: {move.uci()}")
            elif not skip_nn_eval:
                # We did NN eval but policy_move is invalid - this shouldn't happen, but handle gracefully
                # Fallback to first legal move (we already have probs_tensor if needed)
                move = legal_move_list[0]
                if VERBOSE:
                    print(f"Warning: policy_move invalid, using first legal move: {move.uci()}")
            else:
                # Only call choose_move if we skipped NN eval earlier (very early opening or early post-opening)
                # OPTIMIZATION: For early post-opening, use a very fast heuristic instead of full NN eval
                if current_ply >= 8 and current_ply < 18:
                    # Early post-opening: use simple heuristic (first reasonable move) for maximum speed
                    # Skip expensive NN forward pass
                    move = legal_move_list[0]  # Use first legal move for maximum speed
                    if VERBOSE:
                        print(f"Using fast heuristic move (early post-opening): {move.uci()}")
                else:
                    try:
                        mv, _, _ = choose_move(ctx.board, model, device=device, temperature=0.8, sample=False)
                        move = mv
                        if VERBOSE:
                            print(f"Using greedy policy move: {move.uci()}")
                    except Exception as e:
                        # Fallback if choose_move fails
                        print(f"info string Error in greedy selection: {e}", file=sys.stderr)
                        # Fallback to first legal move
                        move = legal_move_list[0]
                        print(f"Fallback to first legal move: {move.uci()}")
        # NEW: Skip PUCT if early termination is triggered (extreme value magnitude)
        # NEW: Skip MCTS if policy is very confident (policy_confidence_skip)
        # NEW: Skip MCTS if low on time (use greedy policy for speed)
        # NEW: Skip MCTS in early post-opening (ply 8-15) for maximum speed
        # BUT: never skip in tactical positions where tactics matter
        elif not early_termination and not policy_confidence_skip:
            skip_mcts_for_time = movetime_ms > 0 and movetime_ms < LOW_TIME_SKIP_MCTS_MS and not tactical_heavy
            skip_mcts_early_post_opening = early_post_opening and not tactical_heavy
            if movetime_ms > 0 and movetime_ms < LOW_TIME_SKIP_MCTS_MS and tactical_heavy and VERBOSE:
                print("Skipping time-based MCTS skip due to tactical complexity (checks/captures/promotions)")
            if early_post_opening and tactical_heavy and VERBOSE:
                print("Skipping early post-opening MCTS skip due to tactical complexity (checks/captures/promotions)")
            if not skip_mcts_for_time and not skip_mcts_early_post_opening and engine.use_puct and hasattr(engine, 'mcts_config') and engine.mcts_config is not None:
                decision_mode = "MCTS search"
                if VERBOSE:
                    print(f"Decision mode: {decision_mode} (ply {current_ply}, phase={game_phase:.2f})")
            
            # Initialize or update search tree
            # OPTIMIZATION: Pass precomputed root evaluation and position cache to avoid redundant NN forward pass
            from .chess_policy.mcts import SearchTree
            root_policy_logits_for_tree = root_policy_logits if not skip_nn_eval else None
            root_value_for_tree = root_value if not skip_nn_eval else None
            
            # Pass cache functions for transposition/repetition caching
            # OPTIMIZATION: Use shared key function to avoid creating Board objects
            def cache_get(state):
                """Get cached evaluation for a position."""
                try:
                    cache_key = _position_key_from_state(state)
                    return _nn_eval_cache.get(cache_key)
                except Exception:
                    return None
            
            def cache_set(state, logits, value):
                """Cache evaluation for a position.
                
                Uses FIFO eviction (ring buffer of recent positions), not true LRU.
                """
                try:
                    cache_key = _position_key_from_state(state)
                    # Ensure logits are on CPU for caching
                    if logits.device.type != 'cpu':
                        logits = logits.cpu()
                    
                    # FIFO eviction: if cache is full, remove oldest entry (first in insertion order)
                    if len(_nn_eval_cache) >= _nn_eval_cache_maxsize:
                        oldest_key = next(iter(_nn_eval_cache))
                        del _nn_eval_cache[oldest_key]
                    
                    _nn_eval_cache[cache_key] = (logits, float(value))
                except Exception:
                    pass
            
            if not hasattr(engine, 'search_tree') or engine.search_tree is None:
                # No existing tree - create new one with precomputed root evaluation and cache
                engine.search_tree = SearchTree(
                    ctx.board, 
                    engine.model, 
                    engine.mcts_config,
                    root_policy_logits=root_policy_logits_for_tree,
                    root_value=root_value_for_tree,
                    position_cache_get=cache_get,
                    position_cache_set=cache_set,
                )
            else:
                # Check if tree matches current position
                try:
                    tree_matches = _search_tree_matches_board(engine.search_tree, ctx.board)
                    
                    if not tree_matches:
                        # Position changed - try aggressive tree reuse
                        # Check if current position is reachable from the tree (opponent's move scenario)
                        if engine.search_tree.update_root_to_position(ctx.board):
                            # Successfully re-rooted tree to match current position
                            if VERBOSE:
                                print(f"Tree reuse: re-rooted to opponent's move position (aggressive reuse)")
                        else:
                            # Current position not in tree - create new tree
                            engine.search_tree = SearchTree(
                                ctx.board, 
                                engine.model, 
                                engine.mcts_config,
                                root_policy_logits=root_policy_logits_for_tree,
                                root_value=root_value_for_tree,
                                position_cache_get=cache_get,
                                position_cache_set=cache_set,
                            )
                except Exception:
                    # If comparison fails, try to reuse tree, otherwise create new one
                    try:
                        if not engine.search_tree.update_root_to_position(ctx.board):
                            # Couldn't reuse, create new tree
                            engine.search_tree = SearchTree(
                                ctx.board, 
                                engine.model, 
                                engine.mcts_config,
                                root_policy_logits=root_policy_logits_for_tree,
                                root_value=root_value_for_tree,
                                position_cache_get=cache_get,
                                position_cache_set=cache_set,
                            )
                    except Exception:
                        # If reuse attempt fails, create new tree
                        engine.search_tree = SearchTree(
                            ctx.board, 
                            engine.model, 
                            engine.mcts_config,
                            root_policy_logits=root_policy_logits_for_tree,
                            root_value=root_value_for_tree,
                            position_cache_get=cache_get,
                            position_cache_set=cache_set,
                        )
            
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
                
                # TUNE: Optimized for accuracy - more aggressive override to catch blunders
                # Current thresholds: top > 10%, MCTS < 8%, top is 2.0x higher
                # More aggressive to catch blunders where MCTS chooses a bad move
                # Make even more aggressive: top > 0.08, MCTS < 0.10, multiplier > 1.8
                # Make less aggressive: top > 0.15, MCTS < 0.05, multiplier > 3.0
                if top_policy_prob > 0.10 and mcts_policy_prob < 0.08 and top_policy_prob > mcts_policy_prob * 2.0:
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
                ctx.board.push(mv)
                try:
                    test_fen = ctx.board.fen()
                    # If this position appeared in last 3 moves, try a different move
                    if test_fen in engine._recent_positions[-3:]:
                        print("Warning: Best move leads to recent repetition, re-searching with more sims...")
                        # Re-search with more simulations to potentially get different move
                        mv, _, _ = engine.search_tree.search(max_simulations_override=sims * 2)
                finally:
                    ctx.board.pop()
            
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
            elif skip_mcts_for_time or skip_mcts_early_post_opening:
                if skip_mcts_early_post_opening:
                    decision_mode = "Greedy policy (early post-opening)"
                    if VERBOSE:
                        print(f"Decision mode: {decision_mode} (ply {current_ply}, skipping MCTS for speed)")
                else:
                    decision_mode = "Greedy policy (low time)"
                    if VERBOSE:
                        print(f"Decision mode: {decision_mode} (movetime_ms={movetime_ms}, ply {current_ply})")
                # OPTIMIZATION: Always prefer policy_move if available (avoids redundant choose_move call)
                if policy_move is not None and policy_move in legal_moves:
                    move = policy_move
                    if VERBOSE:
                        print(f"Using cached policy_move: {move.uci()}")
                elif not skip_nn_eval:
                    # We did NN eval but policy_move is invalid - fallback to first legal move
                    move = legal_move_list[0]
                    if VERBOSE:
                        print(f"Warning: policy_move invalid, using first legal move: {move.uci()}")
                else:
                    # Only call choose_move if we skipped NN eval earlier
                    # OPTIMIZATION: For early post-opening, use fast heuristic instead
                    if current_ply >= 8 and current_ply < 18:
                        move = legal_move_list[0]  # Use first legal move for maximum speed
                        if VERBOSE:
                            print(f"Using fast heuristic move (early post-opening, low time): {move.uci()}")
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
                # OPTIMIZATION: Always prefer policy_move if available (avoids redundant choose_move call)
                if policy_move is not None and policy_move in legal_moves:
                    move = policy_move
                    if VERBOSE:
                        print(f"Using cached policy_move: {move.uci()}")
                elif not skip_nn_eval:
                    # We did NN eval but policy_move is invalid - fallback to first legal move
                    move = legal_move_list[0]
                    if VERBOSE:
                        print(f"Warning: policy_move invalid, using first legal move: {move.uci()}")
                else:
                    # Only call choose_move if we skipped NN eval earlier
                    # OPTIMIZATION: For early post-opening, use fast heuristic instead
                    if current_ply >= 8 and current_ply < 18:
                        move = legal_move_list[0]  # Use first legal move for maximum speed
                        if VERBOSE:
                            print(f"Using fast heuristic move (early post-opening, greedy): {move.uci()}")
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
            ctx.board.push(move)
            try:
                engine._recent_positions.append(ctx.board.fen())
                # Keep only last 10 positions
                if len(engine._recent_positions) > 10:
                    engine._recent_positions.pop(0)
            finally:
                ctx.board.pop()
        
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
        
        # Final tactical sanity check: if we somehow missed a mate in 1 or huge free capture,
        # override the chosen move. This acts as a "guard rail" around the whole decision logic.
        mate_mv = None
        board = ctx.board
        for mv in legal_moves:
            board.push(mv)
            if board.is_checkmate():
                board.pop()
                mate_mv = mv
                break
            board.pop()
        
        if mate_mv is not None and mate_mv != move:
            if VERBOSE:
                print(f"Overriding move {move.uci()} with mate-in-1 {mate_mv.uci()}")
            move = mate_mv
        else:
            tactical_mv = find_obvious_tactic(ctx.board, legal_moves)
            if tactical_mv is not None and tactical_mv != move:
                if VERBOSE:
                    print(f"Overriding move {move.uci()} with obvious winning capture {tactical_mv.uci()}")
                move = tactical_mv
        
        return move
        
    except Exception as e:
        print(f"Error in move selection: {e}, falling back to policy move")
        import traceback
        traceback.print_exc()
        # Fallback to the policy move we already computed
        move = policy_move if policy_move is not None and policy_move in legal_moves else legal_move_list[0]
        print(f"Selected move (exception fallback): {move.uci()}")
        
        # Ensure legal_moves is available (might not be if exception occurred early)
        if 'legal_moves' not in locals() or legal_moves is None:
            legal_moves = list(ctx.board.generate_legal_moves())
            if not legal_moves:
                ctx.logProbabilities({})
                raise ValueError("No legal moves available (i probably lost didn't i)")
        
        # Final tactical sanity check: if we somehow missed a mate in 1 or huge free capture,
        # override the chosen move. This acts as a "guard rail" around the whole decision logic.
        mate_mv = None
        board = ctx.board
        for mv in legal_moves:
            board.push(mv)
            if board.is_checkmate():
                board.pop()
                mate_mv = mv
                break
            board.pop()
        
        if mate_mv is not None and mate_mv != move:
            if VERBOSE:
                print(f"Overriding move {move.uci()} with mate-in-1 {mate_mv.uci()}")
            move = mate_mv
        else:
            tactical_mv = find_obvious_tactic(ctx.board, legal_moves)
            if tactical_mv is not None and tactical_mv != move:
                if VERBOSE:
                    print(f"Overriding move {move.uci()} with obvious winning capture {tactical_mv.uci()}")
                move = tactical_mv
        
        # CRITICAL: Always ensure logProbabilities is called before returning
        # This ensures move_probs is set even in exception cases
        try:
            # Try to use existing move_probs if available, otherwise create a simple one
            if 'move_probs' in locals() and move_probs and isinstance(move_probs, dict) and move in move_probs:
                ctx.logProbabilities(move_probs)
            else:
                # Fallback: create a simple probability dict for the selected move
                ctx.logProbabilities({move: 1.0})
        except Exception as log_err:
            print(f"Warning: logProbabilities failed in exception handler: {log_err}")
            # Last resort: try one more time with a simple dict
            try:
                ctx.logProbabilities({move: 1.0})
            except:
                pass  # If this fails, we've done our best
        
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
    # Clear the NN evaluation cache to avoid memory buildup
    _clear_nn_eval_cache()
    print("Chess engine reset complete")
