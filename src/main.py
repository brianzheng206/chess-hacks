from .utils import chess_manager, GameContext
from chess import Move, Board
import sys
import os
import io
import contextlib
import time
from functools import lru_cache

# Compatibility shim for python-chess transposition_key
# The chess-engine expects board.transposition_key() as a method
# but some python-chess versions use _transposition_key as an attribute
if not hasattr(Board, 'transposition_key'):
    def transposition_key(self):
        """Compatibility method for transposition_key."""
        return self._transposition_key
    Board.transposition_key = transposition_key

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
# Use stockfish_949.pt from chess-engine directory
MODEL_PATH = "/home/brianzheng/chess-engine/stockfish_949.pt"
# Opening book enabled
OPENING_BOOK_PATH = str(REPO_ROOT / "opening_book.pkl") if (REPO_ROOT / "opening_book.pkl").exists() else None

print("Loading chess engine model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load stockfish_949.pt with backward compatibility for policy_head.4 -> policy_head.3
# This model uses the old architecture format
try:
    state = torch.load(MODEL_PATH, map_location=device)
    meta = state.get("meta", {})
    state_dict = state["model"]
    
    # Transform old checkpoint: policy_head.4 -> policy_head.3
    if "policy_head.4.weight" in state_dict and "policy_head.3.weight" not in state_dict:
        state_dict = dict(state_dict)
        state_dict["policy_head.3.weight"] = state_dict.pop("policy_head.4.weight")
        state_dict["policy_head.3.bias"] = state_dict.pop("policy_head.4.bias")
        state["model"] = state_dict
        print("Applied backward compatibility: transformed policy_head.4 -> policy_head.3")
    
    # Use exact architecture from metadata (no defaults!)
    arch = meta["arch"]
    if arch == "PolicyValueResNet":
        from chess_policy.model import PolicyValueResNet
        in_ch = meta["in_channels"]  # Must match exactly
        width = meta["width"]  # Must match exactly
        blocks = meta["n_blocks"]  # Must match exactly
        model = PolicyValueResNet(in_channels=in_ch, width=width, n_blocks=blocks, dropout=0.0)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    # Load with strict=True - fail if there's any mismatch
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys:
        raise RuntimeError(f"Missing keys: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected keys: {incompatible.unexpected_keys}")
    
    print("Model loaded successfully (strict=True, architecture matches checkpoint)")
    # Verify model loaded correctly by checking a test inference
    import chess
    from chess_policy.infer import choose_move
    from chess_policy.move_index import move_to_index
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
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
    raise

# Performance optimizations: set model to eval mode and disable gradients
model.eval()
torch.set_grad_enabled(False)

# Note: Warm-up is skipped to allow server to start quickly
# The first move will naturally warm up the model, and the performance impact is minimal

# Create UCI engine with PUCT search
engine = UciEngine(
    model,
    use_puct=True,
    sims=120,  # Reduced default simulations for faster moves
    c_puct=1.2,  # PUCT exploration constant
    device=device,
    opening_book_path=OPENING_BOOK_PATH,
    opening_max_ply=8,
)
print("Chess engine initialized")

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
    
    if hasattr(engine, "search") and engine.search is not None:
        # Use smaller sims to stay within time budget
        # The search method doesn't support time_budget parameter, so we use sims
        sims_small = max(50, min(sims_cap, 120))
        move = engine.search.search(board, simulations=sims_small)
        
        elapsed = time.monotonic() - start
        # Allow a tiny top-up if we're way under budget (but cap it)
        if elapsed < budget_s * 0.5 and elapsed < 0.15:
            remaining_budget = budget_s - elapsed
            if remaining_budget > 0.05:  # At least 50ms remaining
                topup_sims = min(int(sims_small * 0.5), 60)
                move2 = engine.search.search(board, simulations=topup_sims)
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
    
    # -------------------------
    # STEP 1: POLICY EVALUATION
    # -------------------------
    policy_move = None
    probs_tensor = None
    value = None
    sorted_moves = []   # NEW: always defined as list
    move_probs = {}     # NEW: initialize here for logging

    # Determine temperature based on game phase
    if current_ply < 12:
        # Opening: use lower temperature for more deterministic play
        temperature = 0.7
    else:
        # Mid/endgame: slightly higher temperature
        temperature = 0.8
    
    # Single forward pass - compute policy once with timing
    nn_start = time.monotonic()
    try:
        policy_move, probs_tensor, value = choose_move(
            ctx.board,
            model,
            device=device,
            temperature=temperature,
            sample=False
        )
        nn_time = (time.monotonic() - nn_start) * 1000
        if nn_time > 10:  # Only print if it's significant
            print(f"NN forward (root): {nn_time:.1f} ms")
        
        # Ensure we got a valid move
        if policy_move is None or policy_move not in legal_moves:
            print("Warning: choose_move returned invalid move, using first legal move")
            policy_move = legal_move_list[0]
        
        # Convert probabilities tensor to dictionary of Move -> probability for logging
        # Use move_to_index to map moves to indices (more reliable than index_to_move)
        from chess_policy.move_index import move_to_index
        
        move_probs = {}
        
        # Get probabilities for legal moves only
        # The probs_tensor is already normalized over all 4672 indices
        # We extract probabilities by mapping each legal move to its index
        if probs_tensor is not None:
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
        
        # Always check the raw tensor to see what the model actually output
        if probs_tensor is not None:
            from chess_policy.encoding import legal_mask_4672
            legal_mask = legal_mask_4672(ctx.board)
            legal_probs_raw = [float(probs_tensor[i].item()) for i in range(len(probs_tensor)) if legal_mask[i] > 0.5]
            if legal_probs_raw:
                legal_max_raw = max(legal_probs_raw)
                legal_sum_raw = sum(legal_probs_raw)
                print(f"Raw tensor check: max_legal={legal_max_raw:.6f}, sum_legal={legal_sum_raw:.6f}, extracted_max={max_prob:.6f}, extracted_count={len(move_probs)}")
                # If there's a big discrepancy, something is wrong with extraction
                if abs(legal_max_raw - max_prob) > 0.1:
                    print(f"WARNING: Raw tensor max ({legal_max_raw:.6f}) doesn't match extracted max ({max_prob:.6f})!")
        
        # Debug output to diagnose probability issues
        # Always print debug info if probabilities are suspiciously uniform/low
        if max_prob < 0.15 or (max_prob < 0.2 and len(legal_move_list) > 15):
            print(f"DEBUG: Probabilities seem low/uniform")
            print(f"  Position: {ctx.board.fen()[:50]}...")
            print(f"  Extracted probs: max={max_prob:.4f}, sum={total_prob:.4f}, legal_moves={len(legal_move_list)}")
            # Check if probs_tensor itself has low values
            if probs_tensor is not None:
                probs_max = float(probs_tensor.max().item())
                probs_sum = float(probs_tensor.sum().item())
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
        
        # Print top 5 moves with probabilities (for logging/debugging only)
        # Note: These are extracted probabilities for logging - policy_move is the ground truth
        sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
        top_5 = sorted_moves[:5]
        print("Top 5 moves from model (for logging):")
        for i, (move, prob) in enumerate(top_5, 1):
            marker = "✓" if move == policy_move else " "
            print(f"  {marker} {i}. {move.uci()}: {prob:.4f} ({prob*100:.2f}%)")
        
        # policy_move from choose_move is the model's actual argmax - this is what we use for selection
        if policy_move is not None:
            policy_prob = move_probs.get(policy_move, 0.0) if move_probs else 0.0
            print(f"Using policy_move (model's argmax): {policy_move.uci()} (prob={policy_prob:.4f})")

        # Log probabilities for downstream tooling
        try:
            ctx.logProbabilities(move_probs)
        except Exception as log_err:
            print(f"Warning: logProbabilities failed: {log_err}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"Warning: Could not compute probabilities: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: uniform distribution and use first legal move
        move_probs = {move: 1.0 / len(legal_move_list) for move in legal_move_list}
        sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
        try:
            ctx.logProbabilities(move_probs)
        except Exception as log_err:
            print(f"Warning: logProbabilities failed after exception: {log_err}")
        # Ensure policy_move is set to a valid move
        if policy_move is None or policy_move not in legal_moves:
            policy_move = legal_move_list[0]
    
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
    if ctx.timeLeft > 0:
        # Map time to simulations: ~100 sims per second heuristic (reduced from 400)
        movetime_ms = ctx.timeLeft
        estimated_sims = max(50, int(movetime_ms * 0.1))  # Much more conservative
        sims = min(estimated_sims, 250)  # Reduced cap from 800 to 250
    
    # Phase-aware adjustments
    if game_phase > 0.7:  # Opening
        sims = min(sims, 120)
        if hasattr(engine, 'search') and engine.search is not None:
            engine.search.c_puct = 1.4
    elif game_phase < 0.3:  # Endgame
        sims = min(sims, 100)
        if hasattr(engine, 'search') and engine.search is not None:
            engine.search.c_puct = 0.9
    else:  # Midgame
        sims = min(sims, 150)
        if hasattr(engine, 'search') and engine.search is not None:
            engine.search.c_puct = 1.2
    
    # ------------------------------
    # STEP 3: SELECTION STRATEGY (using UCI engine logic)
    # ------------------------------
    move = None
    decision_mode = None

    try:
        # Determine if we're in opening phase (same logic as UCI engine)
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
            tactical_move = engine._check_tactical_opportunity()
            if tactical_move is not None:
                # Found a tactical opportunity - use it instead of opening book
                decision_mode = "Tactical opportunity"
                move = tactical_move
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
            # If top model move has >30% probability and is different from opening book, compare them
            if top_model_move is not None and top_model_move != mv and top_model_prob > 0.3:
                # Model is confident in a different move - compare with opening book using PUCT
                if engine.use_puct and engine.search is not None:
                    # Quick evaluation of both moves
                    test_board_book = ctx.board.copy()
                    test_board_book.push(mv)
                    value_book = -engine.search._rollout_value(test_board_book)
                    
                    test_board_model = ctx.board.copy()
                    test_board_model.push(top_model_move)
                    value_model = -engine.search._rollout_value(test_board_model)
                    
                    # If model move is better or similar (within 0.1), use it
                    # Also if model confidence is very high (>40%), prefer it
                    if value_model > value_book - 0.1 or top_model_prob > 0.4:
                        decision_mode = "Model override (high confidence)"
                        move = top_model_move
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
                    print(f"Decision mode: {decision_mode} (opening, ply {current_ply})")
                    print(f"Model very confident ({top_model_prob*100:.1f}%) in {top_model_move.uci()}, overriding opening book {mv.uci()}")
                    # Track position after move
                    test_board = ctx.board.copy()
                    test_board.push(move)
                    engine._recent_positions.append(test_board.fen())
                    if len(engine._recent_positions) > 10:
                        engine._recent_positions.pop(0)
                    return move
            
            # If we have PUCT available, do a quick search to see if there's a better move
            # This catches subtle advantages that the simple tactical check might miss
            if engine.use_puct and engine.search is not None:
                # Quick search with fewer simulations (50-100) to compare with opening book
                quick_sims = min(100, max(50, engine.sims // 4))
                # Use time budget if available
                time_budget_s = None
                if ctx.timeLeft > 0:
                    time_budget_s = (ctx.timeLeft / 1000.0) * 0.1  # Use 10% of time for quick search
                search_move = engine.search.search(ctx.board, simulations=quick_sims, time_budget_s=time_budget_s)
                
                if search_move is not None and search_move != mv:
                    # Compare values: get evaluation for both moves
                    # After pushing a move, the board's turn flips, so we need to negate the value
                    # to get it from our perspective
                    test_board_book = ctx.board.copy()
                    test_board_book.push(mv)
                    value_book = -engine.search._rollout_value(test_board_book)  # Negate: opponent's perspective -> ours
                    
                    test_board_search = ctx.board.copy()
                    test_board_search.push(search_move)
                    value_search = -engine.search._rollout_value(test_board_search)  # Negate: opponent's perspective -> ours
                    
                    # Adaptive threshold: higher for early opening and high-confidence book moves
                    # Early opening (ply 0-4): trust book more (threshold 0.25-0.3)
                    # Mid opening (ply 4+): more flexible (threshold 0.15-0.2)
                    # High confidence book move (>0.8): trust more (increase threshold)
                    base_threshold = 0.25 if current_ply < 4 else 0.18
                    if book_move_weight > 0.8:
                        threshold = base_threshold + 0.1  # Trust high-confidence moves more
                    else:
                        threshold = base_threshold
                    
                    # If search move is significantly better, use it
                    # Value is from our perspective, so higher is better
                    if value_search > value_book + threshold:
                        mv = search_move
                        decision_mode = "Opening book (PUCT override)"
                        print(f"Decision mode: {decision_mode} (opening, ply {current_ply})")
                        print(f"PUCT found better move: {search_move.uci()} (value {value_search:.3f} vs book {value_book:.3f})")
                    else:
                        decision_mode = "Opening book"
                        print(f"Decision mode: {decision_mode} (opening, ply {current_ply})")
                        print(f"Using opening book move: {mv.uci()} (value {value_book:.3f} vs PUCT {value_search:.3f})")
                else:
                    decision_mode = "Opening book"
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
        
        # Use model/PUCT for non-opening positions
        if engine.use_puct and engine.search is not None:
            decision_mode = "PUCT search"
            print(f"Decision mode: {decision_mode} (ply {current_ply}, phase={game_phase:.2f})")
            
            # Parse time budget if available
            time_budget_s = None
            if ctx.timeLeft > 0:
                time_budget_s = (ctx.timeLeft / 1000.0) * 0.8  # Use 80% of remaining time
            
            # Search for best move
            search_start = time.monotonic()
            mv = engine.search.search(ctx.board, simulations=sims, time_budget_s=time_budget_s)
            search_time = (time.monotonic() - search_start) * 1000
            print(f"Search: {search_time:.1f} ms (sims={sims})")
            
            try:
                cs = engine.search.cache_stats()
                print(
                    f"PUCT cache size={cs['size']} "
                    f"eval_cache={cs.get('eval_cache_size', 0)} "
                    f"hits={cs['hits']} misses={cs['misses']} "
                    f"hit_rate={cs['hit_rate']:.3f}"
                )
            except Exception:
                pass
            
            # If move would lead to a position we've seen recently, try to avoid it
            if mv is not None:
                test_board = ctx.board.copy()
                test_board.push(mv)
                test_fen = test_board.fen()
                # If this position appeared in last 3 moves, try a different move
                if test_fen in engine._recent_positions[-3:]:
                    print("Warning: Best move leads to recent repetition, re-searching with more sims...")
                    # Re-search with more simulations to potentially get different move
                    mv = engine.search.search(ctx.board, simulations=sims * 2, time_budget_s=time_budget_s)
            
            # Log root value prediction (from current player's perspective)
            try:
                v = engine.search._rollout_value(ctx.board)
                print(f"info string root_value {v:.3f}")
            except Exception:
                pass
            
            move = mv
        else:
            # Use greedy policy selection (fast, no search)
            decision_mode = "Greedy policy"
            print(f"Decision mode: {decision_mode} (ply {current_ply})")
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
        print(f"Final decision mode: {decision_mode}")
        print(f"Selected move: {move.uci()}")
        
        # Log probability for debugging (from our extracted move_probs, which is best-effort)
        if move_probs and move in move_probs:
            move_prob = move_probs[move]
            print(f"Move probability (from extracted probs): {move_prob:.4f} ({move_prob*100:.2f}%)")
        
        return move
        
    except Exception as e:
        print(f"Error in move selection: {e}, falling back to policy move")
        import traceback
        traceback.print_exc()
        # Fallback to the policy move we already computed
        move = policy_move if policy_move is not None and policy_move in legal_moves else legal_move_list[0]
        print(f"Selected move (exception fallback): {move.uci()}")  # NEW
        return move


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    print("Resetting chess engine for new game...")
    engine.ucinewgame()  # This resets the board and clears caches
    # Clear the FEN cache to avoid memory buildup
    _cached_nn_eval.cache_clear()
    print("Chess engine reset complete")
