"""Monte Carlo Tree Search with PUCT for chess policy networks.

This module implements MCTS with PUCT (Polynomial Upper Confidence bounds for Trees)
for use with neural network chess models. The search uses the model's policy and value
predictions to guide exploration and selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Tuple, TYPE_CHECKING, Optional, Dict, Any
import math
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Try to import chess module for tactical move detection
try:
    import chess
    CHESS_AVAILABLE = True
except ImportError:
    CHESS_AVAILABLE = False
    chess = None


def is_tactical_move(board: Any, move: Any) -> bool:
    """Returns True if the move is a check or a capture.
    
    Can be extended later to include promotions or big material swings.
    
    Args:
        board: A chess board object (e.g., chess.Board) that supports:
            - is_capture(move) -> bool
            - push(move) -> None
            - is_check() -> bool
            - pop() -> None
        move: A chess move object (e.g., chess.Move).
    
    Returns:
        True if the move is a capture or gives check, False otherwise.
    
    Note: This function temporarily mutates the board (push/pop) to check for check.
    If you need to preserve the board state, pass a copy.
    """
    if not CHESS_AVAILABLE:
        return False
    
    try:
        # Check if move is a capture
        if hasattr(board, 'is_capture') and board.is_capture(move):
            return True
        
        # Check detection: need to test move on the board
        # We push the move, check if it gives check, then pop to restore state
        if hasattr(board, 'push') and hasattr(board, 'is_check') and hasattr(board, 'pop'):
            board.push(move)
            gives_check = board.is_check()
            board.pop()
            return gives_check
    except (AttributeError, TypeError, ValueError):
        # If board doesn't support these operations, return False
        pass
    
    return False

if TYPE_CHECKING:
    # Type hints for chess move and board objects
    # In practice, these will be chess.Move and chess.Board from python-chess
    AnyMoveType = object  # chess.Move
    GameState = object  # chess.Board
else:
    # At runtime, use Any for flexibility
    AnyMoveType = Any


@dataclass
class SearchConfig:
    """Configuration parameters for MCTS search.
    
    Blunder-focused defaults for catching mistakes and trusting the engine:
        - n_simulations: 600 (balanced search depth)
        - c_puct: 0.8 (trust policy more, less exploration)
        - tactical_bonus: 0.02 (look deeper along forcing lines)
        - min_policy_moves: 12 (keep more weird gambits alive)
        - use_dirichlet_noise: False (no noise for play, only for training/self-play)
        - temperature: 0.0 (argmax on visit counts, deterministic selection)
    
    Attributes:
        n_simulations: Number of MCTS simulations to run from the root.
            More simulations = stronger but slower. Default: 600 for balanced play.
        c_puct: PUCT exploration constant. Controls exploration vs trusting the policy.
            Lower values (0.8-1.0) = trust policy more, less exploration.
            Higher values (1.5-2.0) = more exploration, may need more simulations.
            Default: 0.8 to trust the policy more and catch blunders.
            Tuning: Try 0.6 if you want to trust the Stockfish-trained policy even more.
        dirichlet_alpha: Alpha parameter for Dirichlet noise applied to root priors.
            Used to add exploration variety. Typical range: 0.1-0.5. Default: 0.3.
        dirichlet_frac: Fraction of Dirichlet noise to mix with root priors.
            Range: [0, 1]. 0.25 means 25% noise, 75% prior. Default: 0.25.
        max_depth: Maximum search depth (in plies). Prevents infinite loops.
            Set to None for no limit. Default: 200.
        temperature: Temperature for move selection from visit counts.
            temperature = 0.0 means we pick the argmax visit count at root (no sampling).
            Higher values (0.5-1.0) = more random sampling.
            Used only for final move selection, not during search.
            Default: 0.0 for deterministic play.
        use_dirichlet_noise: Whether to apply Dirichlet noise to root node priors.
            Should be False for normal play; True only for training/self-play.
            When False, root priors are left as the network gave them.
            Default: False.
        device: PyTorch device for model inference (cuda/cpu).
            Default: cuda if available, else cpu.
        min_policy_moves: Minimum number of legal moves to keep in policy distribution.
            Even if model assigns tiny priors, at least this many top moves are preserved.
            Helps keep unusual gambits (like Scholar's mate) alive long enough for evaluation.
            Default: 12 to keep more weird gambits alive.
        epsilon_prior: Minimum prior probability floor for legal moves.
            Legal moves never get exactly zero probability, ensuring they can be explored.
            Default: 1e-4.
        tactical_bonus: Small bonus added to PUCT score for tactical moves (checks/captures).
            Helps search favor tactical lines slightly deeper while remaining PUCT-guided.
            Default: 0.02 to look deeper along forcing lines.
            Tuning: Try 0.03 for extra eagerness to look at forcing lines.
        min_simulations: Minimum number of simulations to run (used in fast_mode or time-limited search).
            Default: 50.
        max_simulations: Maximum number of simulations to run (used for time-limited search).
            Default: 1000.
        fast_mode: If True, use fewer simulations and optionally reduced depth for speed.
            Useful in quiet/obvious positions where full strength isn't needed.
            Default: False.
    
    Tuning Guide:
        Speed vs Strength:
            - More simulations (n_simulations) = stronger but slower
            - Higher c_puct (1.5-2.0) = more exploration, may need more simulations
            - Lower c_puct (0.8-1.0) = more exploitation, faster convergence, trust policy more
            - fast_mode = True: Use min_simulations, good for obvious moves
        
        Training vs Evaluation:
            - Training: use_dirichlet_noise=True, higher temperature (0.5-1.0)
            - Evaluation/Play: use_dirichlet_noise=False, temperature=0.0 (argmax)
        
        Typical Configurations:
            - Fast/Blitz: n_simulations=50-100, fast_mode=True, c_puct=1.0
            - Standard: n_simulations=400-600, c_puct=1.0, temperature=0.0
            - Strong: n_simulations=800-1600, c_puct=1.0, temperature=0.0
    """
    n_simulations: int = 600
    c_puct: float = 0.8
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    max_depth: int | None = 200
    temperature: float = 0.0
    use_dirichlet_noise: bool = False
    device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu"
    min_policy_moves: int = 12
    epsilon_prior: float = 1e-4
    tactical_bonus: float = 0.02
    min_simulations: int = 50
    max_simulations: int = 1000
    fast_mode: bool = False


class GameState(Protocol):
    """Protocol defining the interface expected from a chess board/state object.
    
    The MCTS implementation expects the game state to provide these methods/attributes:
    
    Methods:
        generate_legal_moves() -> Iterable[AnyMoveType]: Returns all legal moves.
        is_game_over() -> bool: True if the game has ended (checkmate, stalemate, etc.).
        result() -> str: Game result ("1-0", "0-1", "1/2-1/2") if game over.
        push(move: AnyMoveType) -> None: Apply a move to the board (mutates state).
        pop() -> AnyMoveType: Undo the last move, return it (mutates state).
        copy() -> GameState: Return a deep copy of the current state.
    
    Attributes:
        turn: The side to move (e.g., chess.WHITE or chess.BLACK).
    
    This protocol is modeled after python-chess.Board but kept generic for flexibility.
    """
    def generate_legal_moves(self) -> object: ...
    def is_game_over(self) -> bool: ...
    def result(self) -> str: ...
    def push(self, move: object) -> None: ...
    def pop(self) -> object: ...
    def copy(self) -> GameState: ...
    turn: object


class SearchNode:
    """A node in the MCTS search tree.
    
    Each node represents a game state and stores PUCT statistics:
    - Prior probability P(s,a) from the neural network
    - Visit count N(s,a) - number of times this move was explored
    - Value sum W(s,a) - cumulative value from all visits
    - Q-value Q(s,a) = W(s,a) / N(s,a) - average value estimate
    
    Attributes:
        parent: Parent node in the search tree. None for root node.
        children: Dictionary mapping moves to child SearchNode instances.
        prior: Prior probability P(s,a) for this move from parent state.
        visit_count: Number of times this node has been visited (N(s,a)).
        value_sum: Sum of all value estimates from visits (W(s,a)).
        state: The game state at this node. May be None if using lightweight representation.
        is_expanded: Whether this node has been expanded (children created).
        is_terminal: Whether this node represents a terminal game state.
        terminal_value: If terminal, the exact outcome from current player's POV in [-1, 0, 1].
            -1 = loss, 0 = draw, +1 = win.
        to_play: The side to move at this node (+1 for white, -1 for black, or similar encoding).
    
    Properties:
        q_value: Average value estimate Q(s,a) = value_sum / visit_count, or 0.0 if unvisited.
    
    Thread-safety:
        Currently NOT thread-safe. For parallel MCTS, you would need:
        - Locks or atomic operations on visit_count and value_sum
        - Virtual loss mechanism to prevent multiple threads from exploring the same path
        - Thread-safe children dictionary access
        - See BatchEvaluator docstring for more details on parallel MCTS requirements.
    """
    
    def __init__(
        self,
        parent: Optional[SearchNode] = None,
        prior: float = 0.0,
        state: Optional[GameState] = None,
        to_play: int = 1,
    ):
        """Initialize a search node.
        
        Args:
            parent: Parent node in the tree. None for root.
            prior: Prior probability P(s,a) for this move.
            state: Game state at this node.
            to_play: Side to move (+1 for white, -1 for black).
        """
        self.parent = parent
        self.children: Dict[AnyMoveType, SearchNode] = {}
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.state = state
        self.is_expanded = False
        self.is_terminal = False
        self.terminal_value: Optional[float] = None
        self.to_play = to_play
    
    def clear(self) -> None:
        """Clear all children and reset node state (useful for tree reuse)."""
        self.children.clear()
        self.is_expanded = False
        self.visit_count = 0
        self.value_sum = 0.0
    
    @property
    def q_value(self) -> float:
        """Return the average value estimate Q(s,a) = value_sum / visit_count.
        
        Returns 0.0 if the node has never been visited (visit_count == 0).
        """
        if self.visit_count > 0:
            return self.value_sum / self.visit_count
        return 0.0
    
    def expand(
        self,
        priors: Dict[AnyMoveType, float],
        state: GameState,
        to_play: int,
        is_terminal: bool,
        terminal_value: Optional[float],
    ) -> None:
        """Expand this node by creating child nodes for each legal move.
        
        Creates a child SearchNode for each move in priors with the corresponding
        prior probability. Sets expansion and terminal flags.
        
        Args:
            priors: Dictionary mapping legal moves to their prior probabilities P(s,a).
            state: The game state at this node (used for child node creation).
            to_play: The side to move at this node (+1 for white, -1 for black).
            is_terminal: Whether this state is terminal (game over).
            terminal_value: If terminal, the outcome from current player's POV in [-1, 0, 1].
        """
        self.state = state
        self.to_play = to_play
        self.is_terminal = is_terminal
        self.terminal_value = terminal_value
        
        if is_terminal:
            # Terminal nodes have no children
            self.is_expanded = True
            return
        
        # Create child nodes for each legal move
        for move, prior_prob in priors.items():
            child_state = state.copy()
            child_state.push(move)
            # Flip to_play for child (opponent's turn)
            child_to_play = -to_play
            child_node = SearchNode(
                parent=self,
                prior=prior_prob,
                state=child_state,
                to_play=child_to_play,
            )
            self.children[move] = child_node
        
        self.is_expanded = True
    
    def is_leaf(self) -> bool:
        """Check if this node is a leaf (not expanded or terminal).
        
        Returns:
            True if the node is not expanded or is terminal, False otherwise.
        """
        return not self.is_expanded or self.is_terminal


class SearchTree:
    """Wrapper class for MCTS search tree that supports tree reuse across moves.
    
    Maintains the root node and root state, allowing the tree to be updated
    when a move is played instead of starting from scratch. This reuses
    previous search information, making subsequent searches faster.
    
    Example:
        tree = SearchTree(root_state, model, config)
        move, dist = tree.search()  # First search
        tree.update_root(move)  # Move root to child after playing move
        move2, dist2 = tree.search()  # Second search reuses previous tree
    """
    
    def __init__(
        self,
        root_state: GameState,
        model: nn.Module,
        config: SearchConfig,
    ):
        """Initialize a search tree.
        
        Args:
            root_state: Initial game state (root position).
            model: Neural network model for position evaluation.
            config: SearchConfig with search parameters.
        """
        self.model = model
        self.config = config
        
        # Track previous root value for blunder detection
        # This enables automatic "double sims on eval spike" move-to-move
        self.prev_root_value: float | None = None
        
        # Reusable batch evaluator for cross-search caching
        # This allows cache hits across multiple search calls (e.g., similar positions)
        device = torch.device(config.device) if isinstance(config.device, str) else config.device
        # Better batch size heuristic: larger batches for GPU
        if device.type == "cuda":
            batch_size = min(64, max(8, config.n_simulations // 2))  # GPU: prefer larger batches
        else:
            batch_size = min(16, max(4, config.n_simulations // 4))  # CPU: smaller is okay
        self.batch_evaluator = BatchEvaluator(model, device, batch_size=batch_size)
        
        # Determine to_play from root_state
        to_play = 1  # Default to white
        if hasattr(root_state, 'turn'):
            try:
                import chess as chess_module
                if root_state.turn == chess_module.BLACK:
                    to_play = -1
                elif root_state.turn == chess_module.WHITE:
                    to_play = 1
            except (ImportError, AttributeError):
                if isinstance(root_state.turn, bool):
                    to_play = 1 if root_state.turn else -1
                elif isinstance(root_state.turn, int):
                    to_play = root_state.turn if root_state.turn in (1, -1) else 1
        
        self.root = SearchNode(parent=None, prior=1.0, state=root_state, to_play=to_play)
        self.root_state = root_state.copy() if hasattr(root_state, 'copy') else root_state
    
    def update_root(self, played_move: AnyMoveType) -> None:
        """Update the root to the child node corresponding to the played move.
        
        After a move is played on the board, this moves the search tree root
        to the corresponding child node. If that child doesn't exist (e.g., the
        move wasn't searched), creates a fresh root node.
        
        This enables tree reuse: instead of discarding the entire search tree
        after each move, we keep the subtree under the played move, which
        contains valuable search information.
        
        Args:
            played_move: The move that was played on the board.
        """
        # Check if the played move exists as a child of current root
        if played_move in self.root.children:
            # Move root to the child node
            new_root = self.root.children[played_move]
            # Update root state by pushing the move
            if hasattr(self.root_state, 'push'):
                self.root_state.push(played_move)
            else:
                # Fallback: try to create new state
                try:
                    self.root_state = self.root_state.copy()
                    if hasattr(self.root_state, 'push'):
                        self.root_state.push(played_move)
                except Exception:
                    # If we can't update state, create fresh root
                    self._create_fresh_root()
                    return
            
            # Clear parent reference and make this the new root
            new_root.parent = None
            new_root.prior = 1.0  # Root has no prior
            self.root = new_root
        else:
            # Move not in tree: create fresh root
            self._create_fresh_root()
    
    def _create_fresh_root(self) -> None:
        """Create a fresh root node (used when move not in tree)."""
        to_play = 1
        if hasattr(self.root_state, 'turn'):
            try:
                import chess as chess_module
                if self.root_state.turn == chess_module.BLACK:
                    to_play = -1
                elif self.root_state.turn == chess_module.WHITE:
                    to_play = 1
            except (ImportError, AttributeError):
                if isinstance(self.root_state.turn, bool):
                    to_play = 1 if self.root_state.turn else -1
                elif isinstance(self.root_state.turn, int):
                    to_play = self.root_state.turn if self.root_state.turn in (1, -1) else 1
        
        self.root = SearchNode(parent=None, prior=1.0, state=self.root_state, to_play=to_play)
    
    def search(
        self,
        available_time_ms: Optional[float] = None,
        max_simulations_override: Optional[int] = None,
    ) -> Tuple[AnyMoveType, torch.Tensor, float]:
        """Run MCTS search from the current root.
        
        Automatically tracks prev_root_value for blunder detection, enabling
        "double sims on eval spike" to work move-to-move without external tracking.
        Uses the shared batch evaluator for cross-search caching.
        
        Args:
            available_time_ms: Optional time budget in milliseconds. If provided,
                runs simulations until time is exhausted (up to max_simulations).
                Not fully implemented - structure is here for future time manager.
            max_simulations_override: Optional override for number of simulations.
                If provided, overrides config.n_simulations.
        
        Returns:
            Tuple of (chosen_move, visit_distribution, root_value) as returned by mcts_search.
        """
        move, dist, current_root_value = mcts_search(
            self.root_state,
            self.model,
            self.config,
            root_node=self.root,
            available_time_ms=available_time_ms,
            max_simulations_override=max_simulations_override,
            prev_root_value=self.prev_root_value,
            batch_evaluator=self.batch_evaluator,  # Reuse batch evaluator for cross-search caching
        )
        # Store current root value for next search (blunder detection)
        self.prev_root_value = current_root_value
        return move, dist, current_root_value


def select_child(node: SearchNode, c_puct: float, config: Optional[SearchConfig] = None) -> Tuple[AnyMoveType, SearchNode]:
    """Select a child node using the PUCT (Polynomial Upper Confidence bounds for Trees) formula.
    
    PUCT balances exploitation (Q-value) with exploration (prior probability) to select
    the most promising child node. The formula encourages exploring moves with high
    prior probability while also exploiting moves with high value estimates.
    
    Tactical Bonus:
        A small bonus (0.02) is added to the PUCT score for tactical moves (checks/captures).
        This heuristic slightly biases search toward checks/captures. It helps the engine
        see mate threats and punish blunders in fewer plies. The bonus is small relative
        to typical Q+U magnitudes so it doesn't override the network and PUCT, it just
        nudges the search to look at forcing moves a bit more.
    
    Args:
        node: The parent SearchNode to select a child from. Must not be terminal.
        c_puct: PUCT exploration constant. Higher values encourage more exploration.
            Typical range: 1.0-2.0.
    
    Returns:
        A tuple of (move, child_node) for the child with the highest PUCT score.
        If multiple children have the same score, ties are broken deterministically
        by move ordering (first move in iteration order).
    
    Raises:
        AssertionError: If node is terminal (terminal nodes have no children to select).
    
    PUCT Formula:
        For each child:
            N = parent.visit_count
            Q = child.q_value (average value estimate)
            P = child.prior (prior probability from neural network)
            n = child.visit_count
            
            u = c_puct * P * sqrt(max(1, N)) / (1 + n)
            score = Q + u + tactical_bonus (if move is tactical)
        
        Select child with maximum score.
    
    The formula ensures that:
    - Moves with high Q-values (exploitation) are preferred
    - Moves with high priors (exploration) are also considered
    - Less-visited moves get a boost (exploration bonus)
    - The exploration bonus decreases as visits increase
    - Tactical moves (checks/captures) get a small bonus to help find short refutations
    """
    assert not node.is_terminal, "Cannot select child from terminal node"
    assert len(node.children) > 0, "Cannot select child from node with no children"
    
    N = node.visit_count
    sqrt_N = math.sqrt(max(1, N))
    
    best_move = None
    best_child = None
    best_score = float('-inf')
    
    # Iterate through children and calculate PUCT score for each
    # Use sorted order for deterministic tie-breaking
    for move, child in sorted(node.children.items(), key=lambda x: str(x[0])):
        Q = child.q_value
        P = child.prior
        n = child.visit_count
        
        # PUCT formula: u = c_puct * P * sqrt(N) / (1 + n)
        u = c_puct * P * sqrt_N / (1 + n)
        score = Q + u
        
        # Add small tactical bonus for checks and captures
        # This heuristic slightly biases search toward checks/captures.
        # It helps the engine see mate threats and punish blunders in fewer plies.
        # The bonus is small relative to typical Q+U magnitudes so it doesn't
        # override the network and PUCT, it just nudges the search to look at forcing moves a bit more.
        tactical_bonus = 0.0
        if config is not None:
            tactical_bonus = getattr(config, "tactical_bonus", 0.0)
        if tactical_bonus and node.state is not None and is_tactical_move(node.state, move):
            score += tactical_bonus
        
        # Select child with maximum score
        # Use > (not >=) so ties are broken deterministically by iteration order
        if score > best_score:
            best_score = score
            best_move = move
            best_child = child
    
    assert best_move is not None and best_child is not None, "Failed to select child"
    return best_move, best_child


class BatchEvaluator:
    """Batch evaluator for efficient neural network inference during MCTS.
    
    Collects states that need evaluation and processes them in batches for better GPU utilization.
    This is similar to the batch inference approach used in AlphaZero training.
    
    Thread-safety:
        - This class uses locks to protect shared state (pending_states, results_cache, stats).
        - Currently, MCTS search is single-threaded, so these locks are primarily future-proofing.
        - Locks are cheap and don't hurt performance in single-threaded mode.
    
    Parallel MCTS (future work):
        For true parallel MCTS with multiple threads running run_simulation() concurrently, you would need:
        
        1. Tree-safe parallel MCTS:
           - Virtual loss: When a worker picks a path, temporarily increment visit_count and subtract
             a "virtual loss" from value_sum to discourage other workers from the same path.
           - Locks or atomic operations on:
             * visit_count updates
             * value_sum updates  
             * children creation/expansion
           - Pattern: Each thread locks nodes as it traverses, adds virtual loss before evaluation,
             then removes virtual loss and adds actual value after backup.
        
        2. Ray actor integration (optional, for distributed evaluation):
           - Ray actor would own the model instance on GPU
           - BatchEvaluator would encode boards locally, send to actor.evaluate_batch.remote(),
             wait for results, then update cache
           - Note: For single GPU setups, Ray is often overkill - single-process batching is usually
             sufficient and simpler.
        
        This class is ready for parallel evaluation, but SearchNode tree operations are not yet
        thread-safe. See open-source AlphaZero/LC0 implementations for reference.
    """
    
    def __init__(self, model: nn.Module, device: torch.device | str, batch_size: int = 32):
        """Initialize batch evaluator.
        
        Args:
            model: Neural network model for evaluation.
            device: PyTorch device for inference.
            batch_size: Maximum batch size for evaluation (default: 32).
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        
        # Thread-safe data structures (locks are cheap, provide safety for future multi-threading)
        self._lock = threading.Lock()  # Lock for thread-safe access to shared state
        self.pending_states: list[Tuple[GameState, object]] = []  # (state, callback_data)
        self.results_cache: Dict[str, Tuple[torch.Tensor, float]] = {}  # FEN -> (logits, value)
        self.stats_batched = 0  # Number of states evaluated in batches
        self.stats_single = 0  # Number of states evaluated individually
        self.stats_cached = 0  # Number of cache hits
        
    def add_state(self, state: GameState, callback_data: object = None) -> None:
        """Add a state to the evaluation queue.
        
        Thread-safe: Uses lock to prevent race conditions.
        
        Args:
            state: Game state to evaluate.
            callback_data: Optional data to associate with this state (e.g., node reference).
        """
        # Use FEN as cache key
        fen = state.fen() if hasattr(state, 'fen') else str(state)
        with self._lock:
            if fen not in self.results_cache:
                self.pending_states.append((state, callback_data))
    
    def evaluate_batch(self) -> None:
        """Evaluate all pending states in a batch and cache results.
        
        Thread-safe: Uses lock to prevent race conditions during evaluation.
        Optimized for GPU: minimizes CPU-GPU transfers and keeps tensors on GPU longer.
        """
        # Thread-safe: acquire lock and copy pending states
        with self._lock:
            if not self.pending_states:
                return
            # Copy pending states to avoid holding lock during evaluation
            states_to_evaluate = self.pending_states.copy()
            self.pending_states.clear()
        
        from .encoding import board_to_tensor, NUM_FEATURE_PLANES
        from .infer import unpack_policy
        
        # Process in batches (using copied list, no lock needed during evaluation)
        for batch_start in range(0, len(states_to_evaluate), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(states_to_evaluate))
            batch_states = states_to_evaluate[batch_start:batch_end]
            
            # Encode all states in batch (CPU encoding, but we'll move to GPU in one go)
            batch_tensors = []
            batch_fens = []
            
            # Get model's expected channel count once
            try:
                c_model = int(getattr(self.model, "in_channels", None) or NUM_FEATURE_PLANES)
            except Exception:
                c_model = NUM_FEATURE_PLANES
            
            for state, _ in batch_states:
                # Encode board to tensor [C, 8, 8]
                x_np = board_to_tensor(state)
                c_encoded = x_np.shape[0]
                
                # Align channels to model's expected input
                if c_encoded > c_model:
                    x_np = x_np[:c_model]
                elif c_encoded < c_model:
                    pad = np.zeros((c_model - c_encoded, x_np.shape[1], x_np.shape[2]), dtype=x_np.dtype)
                    x_np = np.concatenate([x_np, pad], axis=0)
                
                batch_tensors.append(x_np)
                fen = state.fen() if hasattr(state, 'fen') else str(state)
                batch_fens.append(fen)
            
            # Stack into batch tensor [B, C, 8, 8] and move to GPU in one operation
            batch_x = np.stack(batch_tensors, axis=0)
            # Use non_blocking=True for faster CPU->GPU transfer
            x = torch.from_numpy(batch_x).to(self.device, non_blocking=True)
            
            # Run model on batch (all on GPU)
            self.model.eval()
            with torch.no_grad():
                output = self.model(x)
                logits_batch, v_pred_batch = unpack_policy(output)
                
                # Extract values - keep on GPU as long as possible
                if v_pred_batch is not None:
                    if v_pred_batch.dim() == 2:
                        values_tensor = v_pred_batch.squeeze(-1)  # Keep on GPU
                    else:
                        values_tensor = v_pred_batch
                else:
                    values_tensor = torch.zeros(len(batch_states), device=self.device)
                
                # Process results - move to CPU only when necessary for caching
                # Batch the CPU transfer for better efficiency
                values_cpu = values_tensor.cpu().numpy()
                logits_cpu = logits_batch.cpu()  # Move entire batch to CPU at once
                
                # Cache results for each state (now using CPU tensors)
                # Thread-safe: acquire lock to update cache
                with self._lock:
                    for i, fen in enumerate(batch_fens):
                        logits = logits_cpu[i]
                        value = float(values_cpu[i])
                        self.results_cache[fen] = (logits, value)
                    
                    # Update stats
                    self.stats_batched += len(batch_states)
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about batch evaluation.
        
        Thread-safe: Uses lock to prevent race conditions.
        """
        with self._lock:
            return {
                'batched': self.stats_batched,
                'single': self.stats_single,
                'cached': self.stats_cached,
                'total': self.stats_batched + self.stats_single + self.stats_cached
            }
    
    def get_result(self, state: GameState) -> Optional[Tuple[torch.Tensor, float]]:
        """Get cached result for a state.
        
        Thread-safe: Uses lock to prevent race conditions.
        
        Args:
            state: Game state to look up.
            
        Returns:
            Tuple of (logits, value) if cached, None otherwise.
        """
        fen = state.fen() if hasattr(state, 'fen') else str(state)
        with self._lock:
            return self.results_cache.get(fen)
    
    def get_result_and_mark_cached(self, state: GameState) -> Optional[Tuple[torch.Tensor, float]]:
        """Get cached result for a state and increment cache hit stats if found.
        
        Thread-safe: Uses lock to prevent race conditions.
        
        Args:
            state: Game state to look up.
            
        Returns:
            Tuple of (logits, value) if cached, None otherwise.
        """
        fen = state.fen() if hasattr(state, 'fen') else str(state)
        with self._lock:
            res = self.results_cache.get(fen)
            if res is not None:
                self.stats_cached += 1
            return res
    
    def should_eval_now(self) -> bool:
        """Check if batch should be evaluated now based on pending states.
        
        Thread-safe: Uses lock to prevent race conditions.
        
        Returns:
            True if batch is full or has at least 2 states (for better GPU utilization), False otherwise.
            Lower threshold (2 instead of 4) to batch more aggressively and reduce single evaluations.
        """
        with self._lock:
            pending_count = len(self.pending_states)
            return pending_count >= self.batch_size or pending_count >= 2
    
    def _cache_result(self, state: GameState, logits: torch.Tensor, value: float, is_single: bool = False) -> None:
        """Cache a result for a state.
        
        Thread-safe: Uses lock to prevent race conditions.
        
        Args:
            state: Game state to cache.
            logits: Policy logits tensor.
            value: Value prediction.
            is_single: If True, increment single evaluation stats.
        """
        fen = state.fen() if hasattr(state, 'fen') else str(state)
        with self._lock:
            self.results_cache[fen] = (logits, value)
            if is_single:
                self.stats_single += 1
    
    def clear_cache(self) -> None:
        """Clear the results cache.
        
        Thread-safe: Uses lock to prevent race conditions.
        """
        with self._lock:
            self.results_cache.clear()
            self.pending_states.clear()


def evaluate_state_with_model(
    state: GameState,
    model: nn.Module,
    device: torch.device | str,
    batch_evaluator: Optional[BatchEvaluator] = None,
) -> Tuple[torch.Tensor, float]:
    """Evaluate a game state using the neural network model.
    
    Encodes the state into the model's input format, runs the model, and returns
    policy logits and value prediction. Can use batch evaluator for efficiency.
    
    Args:
        state: Game state to evaluate (must implement GameState protocol).
        model: Neural network model (PolicyValueResNet or similar).
            Expected to output (policy_logits, value) where:
            - policy_logits: [B, POLICY_SIZE] = [B, 4672] tensor
            - value: [B, 1] tensor in range [-1, 1]
        device: PyTorch device for model inference.
        batch_evaluator: Optional BatchEvaluator for batch inference. If provided,
            the state will be added to the batch queue instead of evaluated immediately.
    
    Returns:
        A tuple of:
        - policy_logits: 1D tensor of shape [POLICY_SIZE] = [4672] containing
          raw logits for all possible moves.
        - value: float in range [-1, 1] representing the position value from
          the perspective of the side to move.
    
    The encoding handles channel alignment automatically to support models
    trained with different numbers of input channels (e.g., 18 vs 25).
    """
    # If batch evaluator is provided, use it for batching
    if batch_evaluator is not None:
        # Check cache and mark as cached if found (thread-safe)
        cached_result = batch_evaluator.get_result_and_mark_cached(state)
        if cached_result is not None:
            return cached_result
        
        # Add to batch queue (thread-safe)
        batch_evaluator.add_state(state)
        
        # Check if we should evaluate now (thread-safe)
        if batch_evaluator.should_eval_now():
            batch_evaluator.evaluate_batch()
            # Get result after batch evaluation (thread-safe, but don't double-count cache hit)
            cached_result = batch_evaluator.get_result(state)
            if cached_result is not None:
                return cached_result
        
        # If we only have 1 state and batch isn't full, we still need to evaluate it
        # But try to wait a bit - the periodic evaluation should catch it
        # For now, fall through to single eval only as last resort
    
    from .encoding import board_to_tensor, NUM_FEATURE_PLANES
    from .infer import unpack_policy
    
    # Encode board to tensor [C, 8, 8]
    x_np = board_to_tensor(state)
    c_encoded = x_np.shape[0]
    
    # Align channels to model's expected input
    try:
        c_model = int(getattr(model, "in_channels", c_encoded))
    except Exception:
        c_model = c_encoded
    
    if c_encoded > c_model:
        x_np = x_np[:c_model]
    elif c_encoded < c_model:
        pad = np.zeros((c_model - c_encoded, x_np.shape[1], x_np.shape[2]), dtype=x_np.dtype)
        x_np = np.concatenate([x_np, pad], axis=0)
    
    # Convert to torch tensor and add batch dimension [1, C, 8, 8]
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)
    
    # Run model
    model.eval()
    with torch.no_grad():
        output = model(x)
        logits_b, v_pred = unpack_policy(output)
        
        # Remove batch dimension from logits: [1, POLICY_SIZE] -> [POLICY_SIZE]
        if logits_b.dim() == 2 and logits_b.size(0) == 1:
            logits = logits_b[0]
        else:
            logits = logits_b
        
        # Extract value: [1, 1] -> float
        value = 0.0
        if v_pred is not None:
            v = v_pred
            if v.dim() == 2 and v.size(0) == 1:
                v = v[0]
            value = float(v.squeeze(-1).cpu().item())
    
    # Cache result if batch evaluator is provided (thread-safe)
    if batch_evaluator is not None:
        batch_evaluator._cache_result(state, logits.cpu(), value, is_single=True)
    
    return logits.cpu(), value


def expand_node(
    node: SearchNode,
    state: GameState,
    model: nn.Module,
    config: SearchConfig,
    batch_evaluator: Optional[BatchEvaluator] = None,
) -> float:
    """Expand a search node by evaluating it with the neural network.
    
    STRONG TERMINAL HANDLING:
        - Checks for game over BEFORE calling the neural network
        - Uses exact terminal values (+1/-1/0) for checkmates and draws
        - Never trusts the network at terminal nodes
        - Returns immediately for terminal positions (no children created)
    
    For non-terminal positions:
        - Evaluates the state with the model
        - Masks illegal moves and computes priors via softmax
        - Expands the node with child nodes
    
    Args:
        node: SearchNode to expand. Should be a leaf node (not yet expanded).
        state: Game state at this node (must implement GameState protocol).
        model: Neural network model for evaluation (only called for non-terminal positions).
        config: SearchConfig with device and other parameters.
    
    Returns:
        float: The value to be backed up in the tree:
        - If terminal: exact outcome from current player's POV (+1, 0, or -1)
        - If non-terminal: neural network value prediction in [-1, 1]
    
    Terminal value encoding (from current player's POV, state.turn):
        +1.0: Current player has won (checkmate delivered)
        -1.0: Current player is mated (checkmated)
        0.0: Stalemate or draw (repetition, 50-move rule, insufficient material, etc.)
    
    Note: Checkmates and draws are treated with exact +1/-1/0 values. We do not
    trust the network at terminal nodes to ensure correct evaluation of winning/losing positions.
    """
    from .encoding import legal_mask_4672
    from .move_index import POLICY_SIZE, move_to_index, index_to_move
    
    # STRONG TERMINAL HANDLING: Check for game over BEFORE calling the neural net
    # We use exact terminal values (+1/-1/0) for checkmates and draws, and never trust
    # the network at terminal nodes. This ensures correct evaluation of winning/losing positions.
    if state.is_game_over():
        # Import chess module for turn comparison (python-chess)
        try:
            import chess as chess_module
        except ImportError:
            chess_module = None
        
        result = state.result()  # "1-0", "0-1", or "1/2-1/2"
        
        # Compute terminal value from POV of side to move at this node
        # The value must be from the perspective of state.turn (the player to move)
        if result == "1-0":
            # White won: +1.0 if it's white's turn, -1.0 if it's black's turn
            if chess_module is not None:
                terminal_value = 1.0 if state.turn == chess_module.WHITE else -1.0
            else:
                # Fallback: assume True = white, False = black
                terminal_value = 1.0 if state.turn else -1.0
        elif result == "0-1":
            # Black won: +1.0 if it's black's turn, -1.0 if it's white's turn
            if chess_module is not None:
                terminal_value = 1.0 if state.turn == chess_module.BLACK else -1.0
            else:
                # Fallback: assume False = black, True = white
                terminal_value = 1.0 if not state.turn else -1.0
        else:
            # Draw (stalemate, repetition, 50-move rule, insufficient material, etc.)
            # Terminal value: 0.0 (draw) from any player's POV
            terminal_value = 0.0
        
        # Set terminal flags and return exact terminal value
        # We do NOT call the neural network for terminal positions
        node.is_terminal = True
        node.terminal_value = float(terminal_value)
        node.is_expanded = True
        node.state = state
        
        # No children in terminal positions - return immediately
        return node.terminal_value
    
    # Non-terminal: evaluate with model
    # Only reach here if state.is_game_over() is False
    # The neural network value is from the POV of state.turn (the side to move) in [-1, 1]
    device = torch.device(config.device) if isinstance(config.device, str) else config.device
    policy_logits, value = evaluate_state_with_model(state, model, device, batch_evaluator)
    
    # Get legal moves and build mask
    legal_moves = list(state.generate_legal_moves())
    legal_mask = legal_mask_4672(state)
    legal_mask_tensor = torch.from_numpy(legal_mask).to(policy_logits.device)
    
    # Mask illegal moves: set logits for illegal moves to -inf
    masked_logits = torch.where(
        legal_mask_tensor > 0.5,
        policy_logits,
        torch.full_like(policy_logits, float('-inf'))
    )
    
    # Softmax to get probabilities over legal moves
    probs = F.softmax(masked_logits, dim=0)
    
    # Build priors dictionary: Dict[Move, float]
    # Keep track of move-index pairs for sorting
    move_probs: list[Tuple[AnyMoveType, float]] = []
    for move in legal_moves:
        try:
            idx = move_to_index(state, move)
            if idx is not None and 0 <= idx < len(probs):
                prob = float(probs[idx].item())
                # If multiple indices map to same move (e.g., underpromotions), take max
                # Check if we already have this move with a higher probability
                existing_prob = next((p for m, p in move_probs if m == move), None)
                if existing_prob is None or prob > existing_prob:
                    # Remove old entry if exists
                    move_probs = [(m, p) for m, p in move_probs if m != move]
                    move_probs.append((move, prob))
        except (ValueError, AttributeError, TypeError):
            # Skip moves that can't be indexed
            continue
    
    # Enforce minimum policy moves: keep at least top K moves by prior
    # This helps keep unusual gambits (like Scholar's mate) alive long enough
    # to be evaluated by the tree, even if the policy head initially assigns them low probability
    K = config.min_policy_moves
    if len(move_probs) > K:
        # Sort by probability (descending) and keep top K
        move_probs.sort(key=lambda x: x[1], reverse=True)
        top_K_moves = {move for move, _ in move_probs[:K]}
        # Keep all moves that are in top K, plus any others that are above epsilon
        epsilon = config.epsilon_prior
        move_probs = [
            (move, prob) for move, prob in move_probs
            if move in top_K_moves or prob >= epsilon
        ]
    
    # Apply epsilon_prior floor: legal moves never get exactly zero probability
    # This ensures all legal moves can be explored, even if model assigns tiny priors
    epsilon = config.epsilon_prior
    priors: Dict[AnyMoveType, float] = {
        move: max(prob, epsilon) for move, prob in move_probs
    }
    
    # Normalize priors to sum to 1.0
    total_prior = sum(priors.values())
    if total_prior > 0:
        priors = {move: prob / total_prior for move, prob in priors.items()}
    else:
        # Fallback: uniform distribution over legal moves
        uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0.0
        priors = {move: uniform_prob for move in legal_moves}
    
    # Determine to_play (assuming chess.Board-like interface)
    # For python-chess: chess.WHITE = True, chess.BLACK = False
    # Convert to +1/-1 encoding
    to_play = 1  # Default to white
    if hasattr(state, 'turn'):
        # Try to import chess module for comparison (if using python-chess)
        try:
            import chess as chess_module
            if state.turn == chess_module.BLACK:
                to_play = -1
            elif state.turn == chess_module.WHITE:
                to_play = 1
        except (ImportError, AttributeError):
            # Fallback: if it's a boolean, True = white = +1, False = black = -1
            if isinstance(state.turn, bool):
                to_play = 1 if state.turn else -1
            # If it's an integer or other type, assume 1 = white, -1 = black
            elif isinstance(state.turn, int):
                to_play = state.turn if state.turn in (1, -1) else 1
    
    # Expand the node
    node.expand(priors, state, to_play, is_terminal=False, terminal_value=None)
    
    return value


def run_simulation(
    root: SearchNode,
    root_state: GameState,
    model: nn.Module,
    config: SearchConfig,
    batch_evaluator: Optional[BatchEvaluator] = None,
) -> None:
    """Run a single MCTS simulation starting at root.
    
    Performs selection (descending to a leaf using PUCT), expansion (evaluating
    the leaf with the model), and backup (propagating the value back up the tree
    with sign flips).
    
    Args:
        root: Root node of the search tree (should already be expanded).
        root_state: Game state at the root node.
        model: Neural network model for position evaluation.
        config: SearchConfig with search parameters (c_puct, max_depth, etc.).
    
    The simulation:
    1. Selects a path from root to leaf using PUCT formula
    2. Expands the leaf (if not terminal) using the model
    3. Backs up the value along the path, flipping sign at each level
    
    Value backup:
        Values are stored from the perspective of the player to move at each node.
        When backing up from child to parent, the sign is flipped because the
        value is from the child's perspective but needs to be from the parent's.
    """
    # Maintain path of nodes from root to leaf (for backup)
    path: list[SearchNode] = []
    
    # Track state for expansion (we need the state at the leaf)
    state = root_state.copy() if hasattr(root_state, 'copy') else root_state
    depth = 0
    node = root
    
    # Selection phase: descend from root to leaf using PUCT
    while node.is_expanded and not node.is_terminal:
        # Check depth limit
        if config.max_depth is not None and depth >= config.max_depth:
            break
        
        # Select child using PUCT
        move, child = select_child(node, config.c_puct, config)
        
        # Add current node to path (before moving to child)
        path.append(node)
        
        # Push move on state to get to child's state
        if hasattr(state, 'push'):
            state.push(move)
        else:
            # Fallback: try to create new state by copying and applying move
            try:
                new_state = state.copy() if hasattr(state, 'copy') else state
                if hasattr(new_state, 'push'):
                    new_state.push(move)
                state = new_state
            except Exception:
                # If we can't push the move, we've reached a leaf
                break
        
        # Move to child
        node = child
        depth += 1
    
    # Current node is the leaf, state is the leaf's state
    leaf_state = state
    
    # Add the leaf node to the path (path now includes root to leaf, inclusive)
    path.append(node)
    
    # Expansion / evaluation phase
    if node.is_terminal:
        # Leaf is already terminal, use stored terminal value
        # terminal_value is from the POV of the player to move at the leaf (node.to_play)
        leaf_value = node.terminal_value
        if leaf_value is None:
            # Fallback: compute from state
            if hasattr(leaf_state, 'is_game_over') and leaf_state.is_game_over():
                result = leaf_state.result() if hasattr(leaf_state, 'result') else "*"
                if hasattr(leaf_state, 'is_checkmate') and leaf_state.is_checkmate():
                    # Current player is mated: -1.0 from their POV
                    leaf_value = -1.0
                elif result in ("1-0", "0-1"):
                    leaf_value = -1.0
                else:
                    leaf_value = 0.0
            else:
                leaf_value = 0.0
    else:
        # Expand the leaf node
        # expand_node returns value from the POV of the player to move at the leaf
        leaf_value = expand_node(node, leaf_state, model, config, batch_evaluator)
    
    # Ensure we have a valid value
    if leaf_value is None:
        leaf_value = 0.0
    
    # Backup phase: propagate value back up the tree with POV sign flips
    # 
    # Value representation: All values are stored from the POV of the player to move at each node.
    # The PolicyValueResNet returns v in [-1, 1] from the POV of the side to move in that position.
    # 
    # Backup pattern:
    #   - leaf_value is from the POV of the leaf's side to move (node.to_play)
    #   - As we back up, we flip the sign because the parent's value must be from the parent's POV
    #   - Each node stores value_sum from its own POV (node.to_play)
    #   - When backing up from child to parent: value = -value (flip for opponent's POV)
    #
    # Example: If leaf is white to move and value = +0.5 (white winning), then:
    #   - Leaf node (white): stores +0.5
    #   - Parent node (black): stores -0.5 (flipped, from black's POV)
    #   - Grandparent (white): stores +0.5 (flipped again, from white's POV)
    value = float(leaf_value)  # leaf_value is from POV of the leaf's side to move
    
    # path is a list of nodes from root to leaf, inclusive
    # We iterate in reverse (from leaf up to root) to back up the value
    for node_in_path in reversed(path):
        # Update this node with the current value (which is from this node's POV)
        node_in_path.visit_count += 1
        node_in_path.value_sum += value  # value must be from POV of node_in_path.to_play
        
        # Flip the value because the next node up the tree is the opponent's POV
        # (We don't flip after the last iteration, but that's fine - we're done)
        value = -value


def mcts_search(
    root_state: GameState,
    model: nn.Module,
    config: SearchConfig,
    root_node: Optional[SearchNode] = None,
    available_time_ms: Optional[float] = None,
    max_simulations_override: Optional[int] = None,
    prev_root_value: Optional[float] = None,
    batch_evaluator: Optional[BatchEvaluator] = None,
) -> Tuple[AnyMoveType, torch.Tensor, float]:
    """Run MCTS with PUCT starting from root_state.
    
    This function performs Monte Carlo Tree Search using the PUCT algorithm,
    guided by the neural network's policy and value predictions. The search
    builds a tree of game states, evaluates positions using the model, and
    uses UCB-style selection to balance exploration and exploitation.
    
    Blunder-sensitive simulation budget:
        When the evaluation jumps significantly in our favor (compared to previous
        position), we suspect the opponent may have blundered and invest more search
        time to capitalize on the opportunity. If there is no previous value, or the
        jump is small, we use the base config.n_simulations.
    
    Args:
        root_state: The starting game state (e.g., current chess position).
            Must implement the GameState protocol (see above).
        model: Neural network model that takes board encoding as input.
            Expected to output (policy_logits, value) where:
            - policy_logits: [B, POLICY_SIZE] = [B, 4672] tensor of move logits
            - value: [B, 1] tensor in range [-1, 1] from side-to-move's perspective
        config: SearchConfig instance with MCTS parameters.
        root_node: Optional existing root node (for tree reuse). If None, creates new root.
        available_time_ms: Optional time budget in milliseconds. Structure provided
            for future time manager integration. Currently not fully implemented.
        max_simulations_override: Optional override for number of simulations.
            If provided, overrides config.n_simulations.
        prev_root_value: Optional value from previous position (from opponent's POV).
            Used to detect blunders: if current_root_value - prev_root_value > 0.5,
            we double the simulation budget.
    
    Returns:
        A tuple of:
        - chosen_move: The move selected by MCTS (highest visit count, or
          temperature-sampled if config.temperature > 0).
        - visit_distribution: 1D tensor of shape [POLICY_SIZE] = [4672] containing
          normalized visit counts for each policy index. Legal moves will have
          non-zero values; illegal moves will be 0. The distribution sums to 1.0
          over legal moves.
        - current_root_value: The root value estimate from the network (float in [-1, 1]
          from POV of side to move). Can be passed as prev_root_value in next call.
    
    Algorithm overview:
        1. Initialize root node with model evaluation (capture root value)
        2. Detect blunders: if value jump > 0.5, double simulation budget
        3. For n_simulations iterations:
           a. Selection: Traverse tree using PUCT formula to select leaf
           b. Expansion: Evaluate leaf with model, create child nodes
           c. Backup: Propagate value back up the tree
        4. Select move from root visit counts (with optional temperature)
        5. Return chosen move, visit distribution, and root value
    
    The search uses the model's policy predictions as priors for move selection
    and value predictions to evaluate positions. PUCT balances exploiting moves
    with high value estimates against exploring moves with high prior probability.
    
    When eval jumps up a lot, we suspect a blunder and invest more search there.
    If there is no previous value, or the jump is small, we just use the base config.n_simulations.
    """
    from .move_index import POLICY_SIZE, move_to_index, index_to_move
    import time
    
    # Create initial config for root expansion (before blunder detection)
    # We'll adjust n_sims after capturing root value
    initial_config = SearchConfig(
        n_simulations=config.n_simulations,
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_frac=config.dirichlet_frac,
        max_depth=config.max_depth,
        temperature=config.temperature,
        use_dirichlet_noise=config.use_dirichlet_noise,
        device=config.device,
        min_policy_moves=config.min_policy_moves,
        epsilon_prior=config.epsilon_prior,
        tactical_bonus=config.tactical_bonus,
        min_simulations=config.min_simulations,
        max_simulations=config.max_simulations,
        fast_mode=config.fast_mode,
    )
    
    # Use provided root node or create new one
    # Expand root and capture the root value from the network
    if root_node is not None:
        root = root_node
        # If root is already expanded, we can reuse it
        if not root.is_expanded:
            # Expand root and capture value
            current_root_value = expand_node(root, root_state, model, initial_config)
        else:
            # Root already expanded, get value from root's Q-value or evaluate
            # If root has been visited, use Q-value; otherwise evaluate
            if root.visit_count > 0:
                current_root_value = root.q_value
            else:
                # Evaluate root state to get current value
                device = torch.device(initial_config.device) if isinstance(initial_config.device, str) else initial_config.device
                _, current_root_value = evaluate_state_with_model(root_state, model, device)
    else:
        # Determine to_play from root_state
        to_play = 1  # Default to white
        if hasattr(root_state, 'turn'):
            try:
                import chess as chess_module
                if root_state.turn == chess_module.BLACK:
                    to_play = -1
                elif root_state.turn == chess_module.WHITE:
                    to_play = 1
            except (ImportError, AttributeError):
                if isinstance(root_state.turn, bool):
                    to_play = 1 if root_state.turn else -1
                elif isinstance(root_state.turn, int):
                    to_play = root_state.turn if root_state.turn in (1, -1) else 1
        
        # Root initialization
        root = SearchNode(parent=None, prior=1.0, state=root_state, to_play=to_play)
        
        # Immediately expand the root and capture the root value
        # expand_node returns the value from the network (from POV of side to move)
        current_root_value = expand_node(root, root_state, model, initial_config)
    
    # BLUNDER-SENSITIVE SIMULATION BUDGET
    # When eval jumps up a lot, we suspect a blunder and invest more search there.
    # If there is no previous value, or the jump is small, we just use the base config.n_simulations.
    # Determine number of simulations to run
    if max_simulations_override is not None:
        n_sims = max_simulations_override
    elif config.fast_mode:
        n_sims = config.min_simulations
    else:
        n_sims = config.n_simulations
    
    # Blunder detection: if value jumped significantly in our favor, double simulations
    # prev_root_value is from opponent's POV, current_root_value is from our POV
    # To compare: flip prev_root_value to our POV (opponent's value from their POV -> our POV)
    # Tuning: Try threshold 0.3 instead of 0.5 if your value head tends to be conservative
    value_jump_threshold = 0.5
    if prev_root_value is not None:
        # Flip prev_root_value to our POV (opponent's value from their POV -> our POV)
        prev_root_value_our_pov = -prev_root_value
        value_jump = current_root_value - prev_root_value_our_pov
        if value_jump > value_jump_threshold:
            # Big jump in eval in our favor => opponent may have blundered
            # Double the simulation budget to capitalize on the opportunity
            n_sims = int(config.n_simulations * 2)
    
    # Clamp to min/max bounds
    n_sims = max(config.min_simulations, min(n_sims, config.max_simulations))
    
    # Adjust max_depth in fast mode if needed
    effective_max_depth = config.max_depth
    if config.fast_mode and config.max_depth is not None:
        # Reduce depth by ~30% in fast mode for speed
        effective_max_depth = int(config.max_depth * 0.7)
    
    # Create effective config with adjusted parameters
    effective_config = SearchConfig(
        n_simulations=n_sims,
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_frac=config.dirichlet_frac,
        max_depth=effective_max_depth,
        temperature=config.temperature,
        use_dirichlet_noise=config.use_dirichlet_noise,
        device=config.device,
        min_policy_moves=config.min_policy_moves,
        epsilon_prior=config.epsilon_prior,
        tactical_bonus=config.tactical_bonus,
        min_simulations=config.min_simulations,
        max_simulations=config.max_simulations,
        fast_mode=config.fast_mode,
    )
    
    # Check if root is terminal (checkmate, stalemate, etc.)
    if root.is_terminal:
        # Terminal position: return appropriate move and distribution
        # For checkmate/stalemate, we still need to return a move
        # If there are no legal moves, we can't return a move - this is an error case
        legal_moves = list(root_state.generate_legal_moves()) if hasattr(root_state, 'generate_legal_moves') else []
        if not legal_moves:
            # No legal moves - game is over
            # Create zero distribution and return None move (or first move if any exist)
            visit_dist = torch.zeros(POLICY_SIZE, dtype=torch.float32)
            # Use terminal value as current_root_value
            terminal_value = root.terminal_value if root.terminal_value is not None else 0.0
            return None, visit_dist, terminal_value  # Or handle this case differently
    
    # Apply Dirichlet noise to root priors if enabled
    # Only apply noise if explicitly enabled and root is not terminal
    # For normal play, use_dirichlet_noise=False, so root priors are left as the network gave them
    if config.use_dirichlet_noise and not root.is_terminal and len(root.children) > 0:
        # Collect original priors
        original_priors = [child.prior for child in root.children.values()]
        n_children = len(original_priors)
        
        # Sample Dirichlet noise: η ~ Dir(alpha)
        alpha = config.dirichlet_alpha
        dirichlet_noise = np.random.dirichlet([alpha] * n_children)
        
        # Mix priors with noise: p_new = (1 - frac) * p_old + frac * eta
        frac = config.dirichlet_frac
        for i, (move, child) in enumerate(root.children.items()):
            child.prior = (1.0 - frac) * original_priors[i] + frac * dirichlet_noise[i]
    
    # Use provided batch evaluator or create a new one if not provided
    # (SearchTree provides one for cross-search caching)
    if batch_evaluator is None:
        device = torch.device(effective_config.device) if isinstance(effective_config.device, str) else effective_config.device
        # Better batch size heuristic: larger batches for GPU
        if device.type == "cuda":
            batch_size = min(64, max(8, n_sims // 2))  # GPU: prefer larger batches
        else:
            batch_size = min(16, max(4, n_sims // 4))  # CPU: smaller is okay
        batch_evaluator = BatchEvaluator(model, device, batch_size=batch_size)
    
    # Run simulations with batch inference
    # Batching happens automatically in evaluate_state_with_model when should_eval_now() is true
    # We only need to flush remaining states at the end
    start_time = time.time() if available_time_ms is not None else None
    
    for sim_idx in range(n_sims):
        # Check time budget if provided (basic implementation)
        if start_time is not None and available_time_ms is not None:
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms >= available_time_ms:
                # Time budget exhausted
                break
        
        run_simulation(root, root_state, model, effective_config, batch_evaluator)
    
    # Final batch evaluation to flush any remaining pending states
    # (Most batching happens automatically in evaluate_state_with_model)
    if batch_evaluator is not None:
        # Check if there are pending states (thread-safe)
        batch_evaluator.evaluate_batch()  # This is safe to call even if empty
    
    # Log batch evaluation statistics
    stats = batch_evaluator.get_stats()
    if stats['total'] > 0:
        batched_pct = (stats['batched'] / stats['total']) * 100
        cached_pct = (stats['cached'] / stats['total']) * 100
        print(f"Batch inference stats: {stats['batched']} batched ({batched_pct:.1f}%), "
              f"{stats['cached']} cached ({cached_pct:.1f}%), {stats['single']} single")
    
    # Move selection at root with policy-aware blending
    # Collect visit counts for root children
    moves = list(root.children.keys())
    children = list(root.children.values())
    visit_counts = torch.tensor([child.visit_count for child in children], dtype=torch.float32)
    
    if len(moves) == 0:
        # No children (shouldn't happen if root was expanded, but handle gracefully)
        visit_dist = torch.zeros(POLICY_SIZE, dtype=torch.float32)
        return None, visit_dist, current_root_value
    
    # Normalize visit counts to get visit distribution pi
    if visit_counts.sum() > 0:
        pi = visit_counts / visit_counts.sum()
    else:
        # Fallback: uniform distribution if no visits
        pi = torch.full_like(visit_counts, 1.0 / len(visit_counts))
    
    # Get policy distribution over the same moves
    # Recompute policy logits for root state to get fresh policy distribution
    # This ensures we have the original network policy (not affected by Dirichlet noise)
    from .encoding import legal_mask_4672
    device = torch.device(effective_config.device) if isinstance(effective_config.device, str) else effective_config.device
    # Pass batch_evaluator to reuse cached result (root was already evaluated during expansion)
    policy_logits, _ = evaluate_state_with_model(root_state, model, device, batch_evaluator)
    
    # Get legal moves and build policy distribution p over the same move order
    legal_moves = list(root_state.generate_legal_moves())
    legal_mask = legal_mask_4672(root_state)
    legal_mask_tensor = torch.from_numpy(legal_mask).to(policy_logits.device)
    
    # Mask illegal moves and softmax to get policy probabilities
    masked_logits = torch.where(
        legal_mask_tensor > 0.5,
        policy_logits,
        torch.full_like(policy_logits, float('-inf'))
    )
    policy_probs = F.softmax(masked_logits, dim=0)
    
    # Build policy distribution p over the same moves as pi
    p = torch.zeros(len(moves), dtype=torch.float32)
    for i, move in enumerate(moves):
        try:
            idx = move_to_index(root_state, move)
            if idx is not None and 0 <= idx < len(policy_probs):
                # If multiple indices map to same move, take max
                prob = float(policy_probs[idx].item())
                if prob > p[i].item():
                    p[i] = prob
        except (ValueError, AttributeError, TypeError):
            continue
    
    # Normalize policy distribution p over legal moves
    if p.sum() > 0:
        p = p / p.sum()
    else:
        # Fallback: uniform if no valid policy found
        p = torch.full_like(p, 1.0 / len(moves))
    
    # Blend visit distribution and policy with small weight alpha
    # We slightly bias toward the network's top move when visits are close,
    # so that an obviously strong policy move isn't overridden by noisy search
    # unless there is strong evidence from the search tree.
    alpha = 0.1  # small bias toward policy
    combined = (1.0 - alpha) * pi + alpha * p
    
    # Optional: policy override when search is indecisive
    # If the top policy move has nearly as much probability as the top search move,
    # prefer the policy move. This reduces cases where noisy search overrides an
    # obviously strong engine-like move.
    policy_best_idx = int(p.argmax().item())
    search_best_idx = int(pi.argmax().item())
    
    # Only consider override if they disagree
    if policy_best_idx != search_best_idx:
        search_best = float(pi[search_best_idx].item())
        policy_best = float(pi[policy_best_idx].item())  # search prob on policy-top move
        
        # If policy-top move has at least 80% of the search-top move's probability,
        # we treat them as "close enough" and will bias toward policy later.
        close_threshold = 0.8
        policy_close = policy_best >= close_threshold * search_best
    else:
        policy_close = False
    
    # Move selection based on temperature
    combined_np = combined.numpy()
    
    if effective_config.temperature < 1e-3:
        # Deterministic: pick argmax of combined, but keep the full combined
        best_idx = int(combined_np.argmax().item())
        chosen_move = moves[best_idx]
        # Use combined itself as the search-improved policy over moves
        probs = combined_np.copy()
        # Normalize for safety
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs[:] = 1.0 / len(moves)
        
        # Policy override when search is indecisive
        # If the top policy move has nearly as much search probability as the top search move,
        # prefer the policy move. This reduces cases where noisy search overrides an
        # obviously strong engine-like move.
        if policy_close:
            # Override selection to the policy-best move if it's close in search prob
            best_idx = policy_best_idx
            chosen_move = moves[best_idx]
    else:
        # Apply temperature to combined distribution
        temp = effective_config.temperature
        # Use log-space to avoid overflow
        log_combined = np.array([math.log(max(1e-10, float(c))) for c in combined_np])
        log_powered = log_combined / temp
        # Shift to avoid overflow: subtract max before exp
        log_powered_max = log_powered.max()
        powered_combined = np.exp(log_powered - log_powered_max)
        
        total = powered_combined.sum()
        if total > 0:
            probs = powered_combined / total
        else:
            probs = np.ones(len(moves)) / len(moves)
        
        chosen_idx = np.random.choice(len(moves), p=probs)
        chosen_move = moves[chosen_idx]
    
    # Build visit distribution tensor aligned with policy indices
    # Use the combined distribution (blend of visits and policy) as the "search-improved policy"
    # The tensor has shape [POLICY_SIZE] with non-zero values only for legal moves
    visit_dist = torch.zeros(POLICY_SIZE, dtype=torch.float32)
    
    # Map moves to policy indices and set probabilities from combined distribution
    for move, prob in zip(moves, probs):
        try:
            idx = move_to_index(root_state, move)
            if idx is not None and 0 <= idx < POLICY_SIZE:
                # If multiple indices map to same move, accumulate probability
                visit_dist[idx] += float(prob)
        except (ValueError, AttributeError, TypeError):
            # Skip moves that can't be indexed
            continue
    
    # Normalize the distribution (in case some moves were skipped or multiple indices per move)
    total_prob = visit_dist.sum().item()
    if total_prob > 0:
        visit_dist = visit_dist / total_prob
    else:
        # Fallback: if no moves were successfully mapped, create uniform over legal moves
        legal_moves = list(root_state.generate_legal_moves()) if hasattr(root_state, 'generate_legal_moves') else []
        if legal_moves:
            uniform_prob = 1.0 / len(legal_moves)
            for move in legal_moves:
                try:
                    idx = move_to_index(root_state, move)
                    if idx is not None and 0 <= idx < POLICY_SIZE:
                        visit_dist[idx] = uniform_prob
                except (ValueError, AttributeError, TypeError):
                    continue
            # Renormalize
            total_prob = visit_dist.sum().item()
            if total_prob > 0:
                visit_dist = visit_dist / total_prob
    
    # Return chosen move, visit distribution, and current root value
    # The root value can be passed as prev_root_value in the next call to detect blunders
    return chosen_move, visit_dist, current_root_value


def debug_root_stats(root: SearchNode, root_state: GameState, model: nn.Module, config: SearchConfig, top_k: int = 5) -> None:
    """Prints top-K moves at root ranked by:
    
    - search visits (N)
    - Q-values
    - policy probabilities
    
    Useful for diagnosing why MCTS might be overriding the top policy move.
    """
    from .move_index import move_to_index
    from .encoding import legal_mask_4672
    
    device = torch.device(config.device) if isinstance(config.device, str) else config.device
    policy_logits, _ = evaluate_state_with_model(root_state, model, device)
    
    legal_moves = list(root_state.generate_legal_moves())
    legal_mask = legal_mask_4672(root_state)
    legal_mask_tensor = torch.from_numpy(legal_mask).to(policy_logits.device)
    masked_logits = torch.where(
        legal_mask_tensor > 0.5,
        policy_logits,
        torch.full_like(policy_logits, float('-inf')),
    )
    policy_probs = F.softmax(masked_logits, dim=0)
    
    rows = []
    for move, child in root.children.items():
        try:
            idx = move_to_index(root_state, move)
            p = float(policy_probs[idx].item()) if idx is not None else 0.0
        except Exception:
            p = 0.0
        rows.append((move, child.visit_count, child.q_value, p))
    
    # Sort by visit count descending
    rows.sort(key=lambda r: r[1], reverse=True)
    print("Top moves by visits:")
    for move, n, q, p in rows[:top_k]:
        print(f"  {move}: N={n}, Q={q:.3f}, P={p:.3f}")
    
    # Sort by policy probability descending
    rows.sort(key=lambda r: r[3], reverse=True)
    print("Top moves by policy:")
    for move, n, q, p in rows[:top_k]:
        print(f"  {move}: N={n}, Q={q:.3f}, P={p:.3f}")

