from __future__ import annotations

import pickle
import sys
import time
from typing import Optional, Dict, List, Tuple

import chess
import random
import torch

from .model import TinyPolicyResNet
from .mcts import SearchTree, SearchConfig, mcts_search
from .infer import choose_move
from .encoding import board_to_tensor
import torch.nn as nn


class UciEngine:
    """Minimal UCI wrapper around a policy+MCTS move generator.

    Uses the new MCTS implementation (mcts.py) with tree reuse for efficient search.
    Supports opening book, tactical opportunity detection, and repetition avoidance.

    Supported commands: uci, isready, ucinewgame, position, go, stop, quit.
    """

    def __init__(self, model: nn.Module, use_puct: bool = True, sims: int = 200, c_puct: float = 1.2, device: Optional[str] = None, opening_book_path: Optional[str] = None, opening_max_ply: int = 8, opening_temperature: float = 0.3, opening_adaptive_depth: bool = True, opening_trust_book: bool = True):
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # Move model to device immediately
        self.model = model.to(self.device)
        self.model.eval()
        self.board = chess.Board()
        self.use_puct = use_puct
        self.sims = sims
        self.c_puct = c_puct
        self.use_nn_value = True  # engine option
        # Create MCTS search config
        self.mcts_config = SearchConfig(
            n_simulations=sims,
            c_puct=c_puct,
            device=device,
            temperature=0.1,  # Low temperature for deterministic play
            use_dirichlet_noise=False,  # No noise for evaluation
            fast_mode=False,
        )
        # Use SearchTree for tree reuse across moves
        self.search_tree: Optional[SearchTree] = None
        self._stop_flag = False
        # Track recent positions (FEN strings) for repetition detection in higher-level clients
        self._recent_positions: List[str] = []
        
        # Opening book support
        self.opening_book: Optional[Dict[str, List[Tuple[chess.Move, float]]]] = None
        self.opening_max_ply = opening_max_ply
        self.opening_temperature = opening_temperature  # Controls variety: 0.0 = deterministic, higher = more random
        self.opening_adaptive_depth = opening_adaptive_depth  # If True, continue using book until position not found
        self.opening_trust_book = opening_trust_book  # If True, trust opening book completely in early opening (ply 0-4)
        if opening_book_path:
            try:
                with open(opening_book_path, 'rb') as f:
                    self.opening_book = pickle.load(f)
                print(f"info string Loaded opening book with {len(self.opening_book):,} positions", file=sys.stderr)
            except Exception as e:
                print(f"info string Failed to load opening book: {e}", file=sys.stderr)
                self.opening_book = None

    def uci(self):
        print("id name chess-policy-puct-lite")
        print("id author codex-cli")
        # Engine options
        print("option name UseNNValue type check default true")
        print("uciok")

    def isready(self):
        print("readyok")

    def ucinewgame(self):
        self.board = chess.Board()
        # Clear recent positions history
        self._recent_positions.clear()
        # Reset search tree (create new one for new game)
        self.search_tree = None

    def position(self, args):
        if not args:
            return
        if args[0] == "startpos":
            self.board = chess.Board()
            moves_idx = 1
        elif args[0] == "fen":
            fen = " ".join(args[1:7])
            self.board = chess.Board(fen=fen)
            moves_idx = 7
        else:
            return
        if moves_idx < len(args) and args[moves_idx] == "moves":
            # Update search tree as moves are played (for tree reuse)
            for mv_uci in args[moves_idx + 1 :]:
                try:
                    mv = self.board.parse_uci(mv_uci)
                    # If we have a search tree, update it to follow the move
                    if self.search_tree is not None:
                        self.search_tree.update_root(mv)
                    self.board.push(mv)
                except Exception:
                    pass
            # If search tree doesn't match current position, reset it
            if self.search_tree is not None:
                # Compare FEN strings to check if positions match
                try:
                    if self.search_tree.root_state.fen() != self.board.fen():
                        self.search_tree = None
                except Exception:
                    # If comparison fails, reset tree
                    self.search_tree = None

    def _sims_from_args(self, args) -> int:
        # Defaults
        sims = self.sims
        # Parse tokens
        i = 0
        movetime_ms = None
        while i < len(args):
            tok = args[i]
            if tok == "movetime" and i + 1 < len(args):
                try:
                    movetime_ms = int(args[i + 1])
                except ValueError:
                    pass
                i += 2
                continue
            if tok in ("nodes", "sims") and i + 1 < len(args):
                try:
                    sims = int(args[i + 1])
                except ValueError:
                    pass
                i += 2
                continue
            # Ignore other tokens for minimal implementation
            i += 1

        # If movetime specified, map to sims with a simple heuristic
        if movetime_ms is not None and sims == self.sims:
            sims = max(50, int(movetime_ms * 0.4))  # ~400 sims per second heuristic
        return sims

    def _check_tactical_opportunity(self) -> Optional[chess.Move]:
        """Quickly check for obvious tactical opportunities (checkmate, hanging pieces, etc.).
        
        Returns a move if there's a clear tactical advantage, None otherwise.
        This is used to override opening book moves when the opponent blunders.
        In the opening, this is more sensitive to catch blunders like hanging pieces.
        """
        # Check for checkmate in one
        for move in self.board.legal_moves:
            test_board = self.board.copy()
            test_board.push(move)
            if test_board.is_checkmate():
                return move
        
        # Check for captures that win significant material
        # Look for captures of undefended pieces or winning trades
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                       chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
        
        # Determine if we're in opening phase (for more sensitive blunder detection)
        current_ply = len(self.board.move_stack)
        in_opening = current_ply < 10  # Consider first 10 plies as opening
        
        best_capture = None
        best_capture_value = 0
        
        for move in self.board.legal_moves:
            if not self.board.is_capture(move):
                continue
            
            # Get captured piece value
            captured_square = move.to_square
            captured_piece = self.board.piece_at(captured_square)
            if captured_piece is None:
                continue
            
            captured_value = piece_values.get(captured_piece.piece_type, 0)
            
            # Check if the captured piece is defended (on the current board, before the move)
            is_defended = self.board.is_attacked_by(not self.board.turn, captured_square)
            
            # Get our piece value (if we're capturing with a piece)
            our_piece = self.board.piece_at(move.from_square)
            our_value = piece_values.get(our_piece.piece_type, 0) if our_piece else 0
            
            # Check for pawn promotion (adds significant value)
            promotion_bonus = 0
            if move.promotion is not None:
                promotion_bonus = piece_values.get(move.promotion, 0) - piece_values.get(chess.PAWN, 0)
            
            # Calculate net material gain
            if not is_defended:
                # Hanging piece - pure gain (plus promotion bonus if applicable)
                net_gain = captured_value + promotion_bonus
            else:
                # Defended piece - check if we win the trade
                # Our piece might be recaptured, so net is captured - our piece + promotion bonus
                net_gain = captured_value - our_value + promotion_bonus
            
            # Tactical opportunity thresholds:
            # - In opening: catch any hanging piece (net_gain > 0) or winning trade (net_gain >= 1)
            #   This catches blunders like hanging pawns, pieces, etc.
            #   Also catch any capture of a piece worth 3+ even if we trade evenly (net_gain >= 0)
            # - Later: require more significant gains (net_gain >= 3) or major piece wins
            if in_opening:
                # More sensitive in opening: catch any hanging piece or good trade
                # Accept if:
                # 1. Net gain > 0 (hanging piece or winning trade)
                # 2. Capturing a minor piece (3+) with even trade (net_gain >= 0)
                # 3. Any capture of major piece (5+) even if we lose a minor piece
                if (net_gain > 0) or (captured_value >= 3 and net_gain >= 0) or (captured_value >= 5):
                    if net_gain > best_capture_value:
                        best_capture = move
                        best_capture_value = net_gain
            else:
                # Later game: require more significant material gain
                if (net_gain >= 3) or (captured_value >= 5 and net_gain > 0):
                    if net_gain > best_capture_value:
                        best_capture = move
                        best_capture_value = net_gain
        
        return best_capture

    def _get_opening_move(self) -> Optional[chess.Move]:
        """Get move from opening book if available.
        
        Uses weighted random selection for variety while maintaining strength.
        Early opening (ply 0-4): More deterministic
        Mid opening (ply 4+): More variety
        """
        if self.opening_book is None:
            return None
        
        # Calculate current ply (number of moves made so far)
        # ply = 0 at start, 1 after first move, 2 after second move, etc.
        current_ply = len(self.board.move_stack)
        
        # Adaptive depth: if enabled, continue using book until position not found
        # Otherwise, use fixed max_ply limit
        if not self.opening_adaptive_depth and current_ply >= self.opening_max_ply:
            return None
        
        fen = self.board.fen()
        if fen not in self.opening_book:
            # If adaptive depth enabled and we're past max_ply, this is expected
            # If we're before max_ply, position should be in book (might indicate issue)
            return None
        
        # Get weighted moves from opening book (already sorted by weight, descending)
        moves = self.opening_book[fen]
        if not moves:
            return None
        
        # Filter to legal moves only
        legal_moves = [(m, w) for m, w in moves if m in self.board.legal_moves]
        if not legal_moves:
            return None
        
        # Take top N moves for selection (e.g., top 5)
        top_n = min(5, len(legal_moves))
        candidates = legal_moves[:top_n]
        
        # Early opening: more deterministic (lower effective temperature)
        # Mid opening: more variety (higher effective temperature)
        # Very early opening (ply 0-5): use top move deterministically for consistency
        # This ensures opening_trust_book works correctly and prevents MCTS from overpowering
        if current_ply <= 5:
            # In very early opening, prefer the top move, but filter out obviously weak moves
            # Filter out moves that are generally considered weak in opening (e.g., f5, g5, h5, b5, a5)
            # These are often gambits or weak moves that shouldn't be played automatically
            weak_moves = {'f7f5', 'f2f4', 'g7g5', 'g2g4', 'h7h5', 'h2h4', 'b7b5', 'b2b4', 'a7a5', 'a2a4'}
            
            # If top move is weak and there's a better alternative, prefer the alternative
            top_move = candidates[0][0]
            if top_move.uci() in weak_moves and len(candidates) > 1:
                # Check if second move is not weak and has reasonable weight
                second_move, second_weight = candidates[1]
                if second_move.uci() not in weak_moves and second_weight > 0.4:
                    # Use second move if it's significantly better (within 20% of top weight)
                    top_weight = candidates[0][1]
                    if second_weight >= top_weight * 0.8:
                        return second_move
            
            # Otherwise, use top move
            return top_move
        elif current_ply < 8:
            effective_temp = self.opening_temperature * 0.5  # More deterministic early
        else:
            effective_temp = self.opening_temperature * 1.5  # More variety later
        
        # Deterministic selection if temperature is very low or only one candidate
        if effective_temp < 0.1 or len(candidates) == 1:
            return candidates[0][0]
        
        # Weighted random selection: P(move) ∝ weight^(1/temperature)
        # Higher temperature = more uniform distribution
        # Lower temperature = more concentrated on top moves
        weights = [w ** (1.0 / effective_temp) for _, w in candidates]
        total_weight = sum(weights)
        if total_weight == 0:
            return candidates[0][0]  # Fallback to top move
        
        probs = [w / total_weight for w in weights]
        
        # Select move based on weighted probabilities
        selected_idx = random.choices(range(len(candidates)), weights=probs, k=1)[0]
        return candidates[selected_idx][0]

    def go(self, args):
        # Check if we're in opening phase
        current_ply = len(self.board.move_stack)
        in_opening = False
        if self.opening_adaptive_depth:
            # Adaptive: check if position is in opening book
            if self.opening_book is not None and self.board.fen() in self.opening_book:
                in_opening = True
        else:
            # Fixed depth: check if within max_ply
            in_opening = current_ply < self.opening_max_ply
        
        # If in opening phase, check for tactical opportunities first
        # This allows the engine to catch blunders even during opening book moves
        if in_opening:
            tactical_move = self._check_tactical_opportunity()
            if tactical_move is not None:
                # Found a tactical opportunity - use it instead of opening book
                print(f"bestmove {tactical_move.uci()}")
                # Track position after move
                test_board = self.board.copy()
                test_board.push(tactical_move)
                self._recent_positions.append(test_board.fen())
                if len(self._recent_positions) > 10:
                    self._recent_positions.pop(0)
                return
        
        # Check opening book (if in opening phase and no tactical opportunity found)
        mv = self._get_opening_move()
        if mv is not None:
            # Get opening book move confidence (weight of selected move)
            fen = self.board.fen()
            book_moves = self.opening_book.get(fen, [])
            book_move_weight = next((w for m, w in book_moves if m == mv), 0.5)
            
            # If opening_trust_book is enabled, trust the opening book completely in early opening
            # This ensures consistent opening play without MCTS overriding
            # Extended to ply < 6 to prevent MCTS from overpowering the opening book
            if self.opening_trust_book and current_ply < 6:
                # Just use the opening book move without MCTS comparison
                print(f"bestmove {mv.uci()}")
                test_board = self.board.copy()
                test_board.push(mv)
                self._recent_positions.append(test_board.fen())
                if len(self._recent_positions) > 10:
                    self._recent_positions.pop(0)
                return
            
            # For later opening (ply 4+), do MCTS comparison but with very high threshold
            # This catches subtle advantages while still respecting the opening book
            # Only do comparison if opening_trust_book is False or we're past early opening
            # If opening_trust_book is True, we extend the "trust book" phase to ply < 6
            # to prevent MCTS from overpowering the opening book
            if self.use_puct and (not self.opening_trust_book or current_ply >= 6):
                # Quick search with fewer simulations (50-100) to compare with opening book
                quick_sims = min(100, max(50, self.sims // 4))
                quick_config = SearchConfig(
                    n_simulations=quick_sims,
                    c_puct=self.c_puct,
                    device=self.device,
                    temperature=0.1,
                    use_dirichlet_noise=False,
                    fast_mode=True,  # Use fast mode for quick comparison
                )
                search_move, _, _ = mcts_search(self.board, self.model, quick_config)
                
                if search_move is not None and search_move != mv:
                    # Compare values by doing quick searches from both positions
                    # and comparing the root Q-values
                    test_board_book = self.board.copy()
                    test_board_book.push(mv)
                    book_tree = SearchTree(test_board_book, self.model, quick_config)
                    _, _, value_book_raw = book_tree.search()  # Run search to get evaluation
                    value_book = -value_book_raw  # Negate: opponent's POV -> ours
                    
                    test_board_search = self.board.copy()
                    test_board_search.push(search_move)
                    search_tree = SearchTree(test_board_search, self.model, quick_config)
                    _, _, value_search_raw = search_tree.search()  # Run search to get evaluation
                    value_search = -value_search_raw  # Negate: opponent's POV -> ours
                    
                    # Very high threshold for opening to trust book more
                    # Opening (ply 6-8): trust book more (threshold 0.5-0.6)
                    # Mid opening (ply 9+): more flexible (threshold 0.4-0.45)
                    # High confidence book move (>0.8): trust even more (increase threshold)
                    base_threshold = 0.5 if current_ply < 9 else 0.4
                    if book_move_weight > 0.8:
                        threshold = base_threshold + 0.2  # Trust high-confidence moves much more
                    else:
                        threshold = base_threshold
                    
                    # Only override if search move is SIGNIFICANTLY better
                    # Value is from our perspective, so higher is better
                    if value_search > value_book + threshold:
                        mv = search_move
            
            print(f"bestmove {mv.uci()}")
            # Track position after opening move
            test_board = self.board.copy()
            test_board.push(mv)
            self._recent_positions.append(test_board.fen())
            if len(self._recent_positions) > 10:
                self._recent_positions.pop(0)
            return
        
        # Use model/MCTS for non-opening positions
        if self.use_puct:
            sims = self._sims_from_args(args)
            # Parse movetime if present to set time budget
            available_time_ms = None
            i = 0
            while i < len(args):
                if args[i] == "movetime" and i + 1 < len(args):
                    try:
                        available_time_ms = float(args[i + 1])
                    except ValueError:
                        pass
                    break
                i += 1
            
            # Update MCTS config with current simulation count
            self.mcts_config.n_simulations = sims
            
            # Get or create search tree for tree reuse
            if self.search_tree is None:
                self.search_tree = SearchTree(self.board, self.model, self.mcts_config)
            else:
                # Check if search tree matches current position
                try:
                    if self.search_tree.root_state.fen() != self.board.fen():
                        self.search_tree = SearchTree(self.board, self.model, self.mcts_config)
                except Exception:
                    # If comparison fails, create new tree
                    self.search_tree = SearchTree(self.board, self.model, self.mcts_config)
            
            # Run search
            mv, visit_dist, root_value = self.search_tree.search(
                available_time_ms=available_time_ms,
                max_simulations_override=sims
            )
            
            # If move would lead to a position we've seen recently, try to avoid it
            if mv is not None:
                test_board = self.board.copy()
                test_board.push(mv)
                test_fen = test_board.fen()
                # If this position appeared in last 3 moves, try a different move
                if test_fen in self._recent_positions[-3:]:
                    # Re-search with more simulations to potentially get different move
                    mv, _, _ = self.search_tree.search(max_simulations_override=sims * 2)
            
            # Log root value prediction (from current player's perspective)
            try:
                # Get Q-value from root node as approximate value
                if hasattr(self.search_tree, 'root') and self.search_tree.root.visit_count > 0:
                    root_value = self.search_tree.root.q_value
                    print(f"info string root_value {root_value:.3f}")
            except Exception:
                pass
        else:
            # Use greedy policy selection (fast, no search)
            try:
                mv, _, _ = choose_move(self.board, self.model, device=self.device, temperature=0.8, sample=False)
            except Exception as e:
                # Fallback if choose_move fails
                print(f"info string Error in greedy selection: {e}", file=sys.stderr)
                mv = next(iter(self.board.legal_moves), None)
        if mv is None:
            print("bestmove 0000")
        else:
            print(f"bestmove {mv.uci()}")
            # Track position after move to detect repetition
            test_board = self.board.copy()
            test_board.push(mv)
            self._recent_positions.append(test_board.fen())
            # Keep only last 10 positions
            if len(self._recent_positions) > 10:
                self._recent_positions.pop(0)

    def stop(self):
        # Minimal: no async search; marker kept for API compatibility
        self._stop_flag = True

    def loop(self):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cmd, args = parts[0], parts[1:]
            if cmd == "uci":
                self.uci()
            elif cmd == "isready":
                self.isready()
            elif cmd == "ucinewgame":
                self.ucinewgame()
            elif cmd == "position":
                self.position(args)
            elif cmd == "go":
                self._stop_flag = False
                self.go(args)
            elif cmd == "stop":
                self.stop()
            elif cmd == "quit":
                break
            elif cmd == "setoption":
                # setoption name <Name> value <Value>
                try:
                    name_idx = args.index("name") + 1 if "name" in args else 1
                except ValueError:
                    name_idx = 1
                try:
                    value_idx = args.index("value") + 1 if "value" in args else None
                except ValueError:
                    value_idx = None
                opt_name = None
                opt_value = None
                if name_idx < len(args):
                    opt_name = args[name_idx]
                if value_idx is not None and value_idx < len(args):
                    opt_value = args[value_idx]
                if opt_name == "UseNNValue" and opt_value is not None:
                    val = str(opt_value).lower() in ("true", "1", "yes", "on")
                    self.use_nn_value = val
                    # MCTS always uses NN value, so this option is kept for compatibility
                    # but doesn't affect behavior (MCTS doesn't support material-only evaluation)
                    # Reset search tree to apply any changes
                    self.search_tree = None
