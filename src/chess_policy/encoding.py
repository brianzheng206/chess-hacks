from __future__ import annotations

from typing import Tuple

import numpy as np
import chess

from .move_index import POLICY_SIZE, legal_indices

# Total number of feature planes produced by `board_to_tensor`.
NUM_FEATURE_PLANES: int = 25

"""
Board encoding / move masking utilities.

Channel layout for `board_to_tensor` (25 planes, shape [C, 8, 8]):
  0-5   : white pieces  (P, N, B, R, Q, K)
  6-11  : black pieces  (p, n, b, r, q, k)
  12    : side to move (all ones if White, else zeros)
  13    : any white castling right (K or Q)
  14    : any black castling right (K or Q)
  15    : en-passant target square (one-hot if present)
  16    : ply count plane, min(ply, 100) / 100 over the board
  17    : game phase plane in [0, 1] based on material weights (N=1, B=1, R=2, Q=4)
  18    : normalized game-progress plane, ply / MAX_PLY with MAX_PLY = 200, clamped to [0, 1]
  19    : fifty-move counter plane, min(halfmove_clock / 100, 1.0) broadcast over the board
  20    : repetition danger plane, 1.0 if a threefold claim is available / repetition(3), else 0.0
  21    : white kingside castling right (WK)
  22    : white queenside castling right (WQ)
  23    : black kingside castling right (BK)
  24    : black queenside castling right (BQ)

Move masking:
  - `legal_mask_4672` builds a flat [4672] mask over the move index space,
    with 1.0 for legal moves and 0.0 for illegal moves.
"""


def board_to_tensor(board: chess.Board, include_halfmove: bool = False) -> np.ndarray:
    """Encode a `chess.Board` into a fixed multi-plane tensor.

    Shape:
        np.ndarray of shape [C, 8, 8] with C = NUM_FEATURE_PLANES (currently 25), dtype float32.

    Channel layout:
        0-5   : white pieces (P, N, B, R, Q, K)
        6-11  : black pieces (p, n, b, r, q, k)
        12    : side to move (1.0 if White to move, else 0.0)
        13    : any white castling right (kingside or queenside)
        14    : any black castling right (kingside or queenside)
        15    : en-passant target square (one-hot, zeros if no ep square)
        16    : ply count plane, broadcast min(ply, 100) / 100
        17    : game phase plane in [0, 1] from simple material weights
        18    : game-progress plane, broadcast clamp(ply / 200, 0.0, 1.0)
        19    : fifty-move counter plane, broadcast min(halfmove_clock / 100, 1.0)
        20    : repetition danger plane, broadcast 1.0 if threefold repetition can be claimed
        21    : white kingside castling right (1.0 if WK available, else 0.0)
        22    : white queenside castling right (1.0 if WQ available, else 0.0)
        23    : black kingside castling right (1.0 if BK available, else 0.0)
        24    : black queenside castling right (1.0 if BQ available, else 0.0)

    Note:
        The `include_halfmove` flag is kept for backward compatibility but is
        currently ignored; the encoding always uses the fixed NUM_FEATURE_PLANES layout.
    """
    planes = np.zeros((NUM_FEATURE_PLANES, 8, 8), dtype=np.float32)

    # Pieces
    for sq, p in board.piece_map().items():
        x, y = chess.square_file(sq), chess.square_rank(sq)
        pt = p.piece_type  # 1..6
        color_offset = 0 if p.color == chess.WHITE else 6
        ch = (pt - 1) + color_offset
        planes[ch, y, x] = 1.0

    # Side to move
    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0

    # Castling rights (any side per color)
    has_wk = board.has_kingside_castling_rights(chess.WHITE)
    has_wq = board.has_queenside_castling_rights(chess.WHITE)
    has_bk = board.has_kingside_castling_rights(chess.BLACK)
    has_bq = board.has_queenside_castling_rights(chess.BLACK)

    if has_wk or has_wq:
        planes[13, :, :] = 1.0
    if has_bk or has_bq:
        planes[14, :, :] = 1.0

    # En passant target square one-hot
    if board.ep_square is not None:
        fx, fy = chess.square_file(board.ep_square), chess.square_rank(board.ep_square)
        planes[15, fy, fx] = 1.0

    # Ply plane: estimate ply from fullmove number and side to move
    fullmoves = int(board.fullmove_number)
    ply = (fullmoves - 1) * 2 + (0 if board.turn == chess.WHITE else 1)
    ply_norm = min(100.0, float(ply)) / 100.0
    planes[16, :, :] = ply_norm

    # Phase plane: normalized material phase based on piece weights
    # Weights: N=1, B=1, R=2, Q=4; sum both sides, normalize by 24
    piece_weights = {
        chess.KNIGHT: 1.0,
        chess.BISHOP: 1.0,
        chess.ROOK: 2.0,
        chess.QUEEN: 4.0,
    }
    total = 0.0
    for p in board.piece_map().values():
        total += piece_weights.get(p.piece_type, 0.0)
    phase = min(max(total / 24.0, 0.0), 1.0)
    planes[17, :, :] = phase

    # Game-progress plane: normalized ply with MAX_PLY = 200
    max_ply = 200.0
    progress = float(ply) / max_ply if max_ply > 0 else 0.0
    if progress < 0.0:
        progress = 0.0
    if progress > 1.0:
        progress = 1.0
    planes[18, :, :] = progress

    # Fifty-move rule halfmove clock plane: min(halfmove_clock / 100, 1.0)
    fifty_move_scalar = float(board.halfmove_clock) / 100.0
    if fifty_move_scalar < 0.0:
        fifty_move_scalar = 0.0
    if fifty_move_scalar > 1.0:
        fifty_move_scalar = 1.0
    planes[19, :, :] = fifty_move_scalar

    # Repetition danger plane: 1.0 if a threefold repetition claim is available
    # Prefer can_claim_threefold_repetition (fast, rule-accurate), fall back to
    # is_repetition(3) if needed.
    rep_flag = False
    try:
        if hasattr(board, "can_claim_threefold_repetition") and board.can_claim_threefold_repetition():
            rep_flag = True
        elif hasattr(board, "is_repetition") and board.is_repetition(3):
            rep_flag = True
    except Exception:
        # Very defensive: never crash encoding on repetition checks.
        rep_flag = False
    planes[20, :, :] = 1.0 if rep_flag else 0.0

    # Explicit castling-right planes (binary per side and rook file)
    if has_wk:
        planes[21, :, :] = 1.0
    if has_wq:
        planes[22, :, :] = 1.0
    if has_bk:
        planes[23, :, :] = 1.0
    if has_bq:
        planes[24, :, :] = 1.0

    return planes


def legal_mask_4672(board: chess.Board) -> np.ndarray:
    """Return a [4672] float32 numpy array with 1.0 for legal moves, else 0.0."""
    mask = np.zeros((POLICY_SIZE,), dtype=np.float32)
    for idx in legal_indices(board):
        mask[idx] = 1.0
    return mask


# Convenience wrappers for prior torch-based call sites
def board_to_tensor_torch(board: chess.Board, include_halfmove: bool = False):
    import torch

    return torch.from_numpy(board_to_tensor(board, include_halfmove=include_halfmove))


def legal_move_mask(board: chess.Board):
    import torch

    return torch.from_numpy(legal_mask_4672(board))


def print_initial_position_stats(include_halfmove: bool = False) -> None:
    import chess

    b = chess.Board()
    t = board_to_tensor(b, include_halfmove=include_halfmove)
    m = legal_mask_4672(b)
    print("Board tensor shape:", t.shape)
    print("Piece planes total (should be 32):", float(t[:12].sum()))
    print("Side-to-move sum:", float(t[12].sum()))
    print("Castling plane sums (W_any,B_any):", [float(t[i].sum()) for i in (13, 14)])
    print("Castling specific planes (WK,WQ,BK,BQ):", [float(t[i].sum()) for i in (21, 22, 23, 24)])
    print("EP sum:", float(t[15].sum()))
    print("Ply plane mean:", float(t[16].mean()))
    print("Phase plane mean:", float(t[17].mean()))
    print("Game-progress plane mean:", float(t[18].mean()))
    print("Fifty-move plane mean:", float(t[19].mean()))
    print("Repetition-danger plane mean:", float(t[20].mean()))
    print("Legal moves count (should be 20):", int(m.sum()))


def quick_self_test() -> None:
    print_initial_position_stats(include_halfmove=False)


if __name__ == "__main__":
    quick_self_test()
