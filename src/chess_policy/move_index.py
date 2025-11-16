"""Move indexing for a 73-plane per-origin scheme.

This module implements a fixed mapping between chess moves and a policy index of
size 4672 = 64 origin squares × 73 planes. The 73 planes per origin are:
  - 56 sliding planes: 8 directions × steps 1..7
    Directions: N, S, E, W, NE, NW, SE, SW using python-chess square numbering
  - 8 knight planes: the 8 standard L-shaped offsets
  - 9 promotion planes: underpromotions only to N/B/R × {forward, capture-left, capture-right}

Queen promotions are encoded via the appropriate sliding plane for the pawn
movement to the last rank, not a special promotion plane. The reverse mapping
best-effort reconstructs a legal move, preferring a queen promotion when needed.

All functions are side-agnostic and work for either color without flipping.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import chess


NUM_SQUARES = 64
PLANES_PER_SQUARE = 73
POLICY_SIZE = NUM_SQUARES * PLANES_PER_SQUARE  # 4672

# Cardinal and diagonal directions in python-chess square numbering.
# Offsets are applied to the linear 0..63 square indices.
DIR_NAMES = ("N", "S", "E", "W", "NE", "NW", "SE", "SW")
DIR_OFFSETS = (
    8,   # N
    -8,  # S
    1,   # E
    -1,  # W
    9,   # NE
    7,   # NW
    -7,  # SE
    -9,  # SW
)

# Knight offsets relative to from-square index. The order defines planes 56..63.
KNIGHT_OFFSETS = (
    -17, -15, -10, -6, 6, 10, 15, 17
)

# Underpromotion target piece types (no queen)
PROMO_TYPES = (chess.KNIGHT, chess.BISHOP, chess.ROOK)


def _base(from_sq: int) -> int:
    return from_sq * PLANES_PER_SQUARE


def _in_board(sq: int) -> bool:
    return 0 <= sq < 64


def _queen_dir_and_steps(from_sq: int, to_sq: int) -> Optional[Tuple[int, int]]:
    """Return (direction_index, steps) if move is queen-like; otherwise None."""
    fx, fy = chess.square_file(from_sq), chess.square_rank(from_sq)
    tx, ty = chess.square_file(to_sq), chess.square_rank(to_sq)
    dx, dy = tx - fx, ty - fy
    adx, ady = abs(dx), abs(dy)

    if dx == 0 and dy != 0:
        direction = 0 if dy > 0 else 1  # N or S
        steps = ady
    elif dy == 0 and dx != 0:
        direction = 2 if dx > 0 else 3  # E or W
        steps = adx
    elif adx == ady and adx != 0:
        if dx > 0 and dy > 0:
            direction = 4  # NE
        elif dx < 0 and dy > 0:
            direction = 5  # NW
        elif dx > 0 and dy < 0:
            direction = 6  # SE
        else:
            direction = 7  # SW
        steps = adx
    else:
        return None

    if steps < 1 or steps > 7:
        return None
    return direction, steps


def _queen_like_index(from_sq: int, to_sq: int) -> Optional[int]:
    ds = _queen_dir_and_steps(from_sq, to_sq)
    if ds is None:
        return None
    direction, steps = ds
    plane = direction * 7 + (steps - 1)  # 0..55
    return _base(from_sq) + plane


def _knight_index(from_sq: int, to_sq: int) -> Optional[int]:
    delta = to_sq - from_sq
    try:
        k = KNIGHT_OFFSETS.index(delta)
    except ValueError:
        return None
    return _base(from_sq) + 56 + k


def _underpromo_index(board: chess.Board, move: chess.Move) -> Optional[int]:
    if move.promotion not in PROMO_TYPES:
        return None
    from_sq, to_sq = move.from_square, move.to_square
    fx, fy = chess.square_file(from_sq), chess.square_rank(from_sq)
    tx, ty = chess.square_file(to_sq), chess.square_rank(to_sq)

    piece = board.piece_at(from_sq)
    color = piece.color if piece is not None else board.turn

    if color == chess.WHITE:
        # Must move to rank 7 (0-based) → 8th rank
        if ty != 7:
            return None
        # forward (0,+1), diag-left (-1,+1), diag-right (+1,+1)
        if tx == fx and ty == fy + 1:
            dir_idx = 0
        elif tx == fx - 1 and ty == fy + 1:
            dir_idx = 1
        elif tx == fx + 1 and ty == fy + 1:
            dir_idx = 2
        else:
            return None
    else:
        # Must move to rank 0 (0-based) → 1st rank
        if ty != 0:
            return None
        # forward (0,-1), diag-left (+1,-1), diag-right (-1,-1) in absolute coords
        if tx == fx and ty == fy - 1:
            dir_idx = 0
        elif tx == fx + 1 and ty == fy - 1:
            dir_idx = 1
        elif tx == fx - 1 and ty == fy - 1:
            dir_idx = 2
        else:
            return None

    try:
        promo_idx = PROMO_TYPES.index(move.promotion)
    except ValueError:
        return None
    plane = 64 + promo_idx * 3 + dir_idx  # 64..72
    return _base(from_sq) + plane


def index_from_move(board: chess.Board, move: chess.Move) -> int:
    """Return the 0-based policy index for a given move.

    Priority:
      1) Underpromotions (N/B/R) to last rank use the 9 promotion planes
      2) Knight moves use the 8 knight planes
      3) All queen-like moves (including king, rook, bishop, queen, pawn moves,
         castling, en passant captures, and promotions to queen) use one of the
         56 sliding planes (direction × steps)

    Raises ValueError if the move cannot be represented (should be rare).
    """
    idx = _underpromo_index(board, move)
    if idx is not None:
        return idx

    piece = board.piece_at(move.from_square)
    if piece is not None and piece.piece_type == chess.KNIGHT:
        idx = _knight_index(move.from_square, move.to_square)
        if idx is not None:
            return idx

    idx = _queen_like_index(move.from_square, move.to_square)
    if idx is None:
        raise ValueError("Move not representable in 73-plane mapping")
    return idx


def move_from_index(board: chess.Board, index: int) -> Optional[chess.Move]:
    """Reconstruct a move from an index for the given position.

    Best-effort: for sliding planes that land a pawn on the last rank, attempt
    queen promotion first (skip trying queen as an underpromotion plane since
    those planes are only for N/B/R). Returns None if no legal move matches.
    """
    if index < 0 or index >= POLICY_SIZE:
        return None
    from_sq = index // PLANES_PER_SQUARE
    plane = index % PLANES_PER_SQUARE

    # Sliding 0..55
    if plane < 56:
        direction = plane // 7
        steps = (plane % 7) + 1
        offset = DIR_OFFSETS[direction]
        to_sq = from_sq
        for _ in range(steps):
            to_sq += offset
            if not _in_board(to_sq):
                return None
            # Prevent wrap across files for horizontal/diagonals
            if offset in (1, -1) and chess.square_rank(to_sq) != chess.square_rank(from_sq):
                return None
            if offset in (8, -8) and chess.square_file(to_sq) != chess.square_file(from_sq):
                return None
            if offset in (9, -9) and (chess.square_file(to_sq) - chess.square_file(from_sq)) != (chess.square_rank(to_sq) - chess.square_rank(from_sq)):
                return None
            if offset in (7, -7) and (chess.square_file(to_sq) - chess.square_file(from_sq)) != -(chess.square_rank(to_sq) - chess.square_rank(from_sq)):
                return None
        mv = chess.Move(from_sq, to_sq)
        if mv in board.legal_moves:
            return mv
        # Try promotions if the from piece is a pawn reaching last rank
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            for promo in (chess.QUEEN, chess.KNIGHT, chess.BISHOP, chess.ROOK):
                mvp = chess.Move(from_sq, to_sq, promotion=promo)
                if mvp in board.legal_moves:
                    return mvp
        return None

    # Knight 56..63
    if plane < 64:
        k = plane - 56
        to_sq = from_sq + KNIGHT_OFFSETS[k]
        if not _in_board(to_sq):
            return None
        # Sanity check: knight L distance
        if chess.square_distance(from_sq, to_sq) != 3:
            return None
        mv = chess.Move(from_sq, to_sq)
        return mv if mv in board.legal_moves else None

    # Underpromotions 64..72 (to N/B/R)
    promo_plane = plane - 64
    promo_idx = promo_plane // 3  # 0:N,1:B,2:R
    dir_idx = promo_plane % 3     # 0:forward,1:diag-left,2:diag-right
    piece = board.piece_at(from_sq)
    color = piece.color if piece else board.turn

    fx, fy = chess.square_file(from_sq), chess.square_rank(from_sq)
    if color == chess.WHITE:
        candidates = [
            (fx, fy + 1),  # forward
            (fx - 1, fy + 1),  # diag-left
            (fx + 1, fy + 1),  # diag-right
        ]
        nx, ny = candidates[dir_idx]
    else:
        candidates = [
            (fx, fy - 1),  # forward
            (fx + 1, fy - 1),  # diag-left (relative)
            (fx - 1, fy - 1),  # diag-right (relative)
        ]
        nx, ny = candidates[dir_idx]

    if not (0 <= nx <= 7 and 0 <= ny <= 7):
        return None
    to_sq = chess.square(nx, ny)
    promo_piece = PROMO_TYPES[promo_idx]
    mv = chess.Move(from_sq, to_sq, promotion=promo_piece)
    return mv if mv in board.legal_moves else None


def legal_indices(board: chess.Board) -> List[int]:
    """Return indices for all legal moves in the given position."""
    result: List[int] = []
    for mv in board.legal_moves:
        try:
            idx = index_from_move(board, mv)
        except ValueError:
            continue
        result.append(idx)
    return result


# Backwards-compat aliases
def move_to_index(board: chess.Board, move: chess.Move) -> Optional[int]:
    try:
        return index_from_move(board, move)
    except ValueError:
        return None


def index_to_move(board: chess.Board, index: int) -> Optional[chess.Move]:
    return move_from_index(board, index)
