from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import chess

from .encoding import board_to_tensor, legal_mask_4672
from .move_index import index_to_move, POLICY_SIZE


def unpack_policy(model_output: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], dict]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return (policy_logits, value_pred_or_none) from model output.

    - If output is a Tensor: returns (policy_logits, None)
    - If output is a (policy_logits, value_pred) tuple: returns both
    - If output is a dict (ChessNet): returns (dict["policy"], dict.get("value"))
    """
    if isinstance(model_output, dict):
        # ChessNet returns {"policy": ..., "value": ...}
        policy = model_output["policy"]
        # Flatten if needed: (B, 73, 8, 8) -> (B, 4672)
        if policy.dim() == 4:
            policy = policy.flatten(1)
        value = model_output.get("value", None)
        return policy, value
    if isinstance(model_output, tuple):
        policy, value = model_output
        return policy, value
    return model_output, None


def _encode_board_for_model(board: chess.Board, model, device: str) -> torch.Tensor:
    """Encode a board and align channels to the model's expected input.

    If the encoding has more channels than `model.in_channels`, extra
    planes are dropped from the end. If it has fewer, zero-planes are
    appended. This keeps old checkpoints compatible when new planes are
    added to the encoder.
    """
    x_np = board_to_tensor(board)
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

    return torch.from_numpy(x_np).unsqueeze(0).to(device)


@torch.no_grad()
def policy_logits(model, board: chess.Board, device: Optional[str] = None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    x = _encode_board_for_model(board, model, device)
    output = model(x)
    logits, _ = unpack_policy(output)
    # Remove batch dimension
    if logits.dim() == 2 and logits.size(0) == 1:
        logits = logits[0]
    return logits


def mask_logits(logits: torch.Tensor, legal_mask: torch.Tensor, illegal_value: float = -1e9) -> torch.Tensor:
    """Set illegal entries (mask==0) to a large negative value.

    Accepts logits of shape [4672] or [B,4672] and a mask of shape [4672] or [B,4672].
    
    Uses -1e9 by default for float32, which is more negative and better for masking.
    For float16 (AMP), values are clamped to -1e4 to avoid overflow.
    """
    # For float16, clamp to -1e4 to avoid overflow (float16 max negative ~-65504)
    # For float32, -1e9 is safe and more negative, which is better for masking
    if logits.dtype == torch.float16:
        safe_illegal_value = max(illegal_value, -1e4)
    else:
        safe_illegal_value = illegal_value
    if legal_mask.dim() == 1 and logits.dim() == 2:
        legal_mask = legal_mask.unsqueeze(0).expand_as(logits)
    return torch.where(legal_mask > 0.5, logits, torch.full_like(logits, safe_illegal_value))


def probs_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Softmax with temperature; expects logits already masked for legality."""
    if temperature <= 0:
        temperature = 1e-6
    return F.softmax(logits / temperature, dim=-1)


@torch.no_grad()
def choose_move(
    board: chess.Board,
    model,
    device: Optional[str] = None,
    temperature: float = 0.8,
    sample: bool = False,
) -> Tuple[Optional[chess.Move], torch.Tensor, float]:
    """Compute probs over legal moves and choose a move.

    Works with any model architecture that outputs policy logits of shape [B, POLICY_SIZE].
    The policy size is detected dynamically from the model output.

    Returns (move, probs) where probs is a [POLICY_SIZE] tensor on CPU for debugging.
    """
    # Run the model once to allow optional value extraction
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    x = _encode_board_for_model(board, model, device)
    out = model(x)
    logits_b, v_pred = unpack_policy(out)
    logits = logits_b[0] if logits_b.dim() == 2 and logits_b.size(0) == 1 else logits_b
    
    # Dynamically detect policy size from model output
    policy_size = logits.shape[-1] if logits.dim() > 0 else POLICY_SIZE
    
    # Generate legal mask - use dynamic size if different from expected
    if policy_size == POLICY_SIZE:
        legal = torch.from_numpy(legal_mask_4672(board)).to(logits.device)
    else:
        # Fallback: create a mask that allows all moves (model should handle legality)
        # This is a safety net for models with different policy sizes
        legal = torch.ones(policy_size, dtype=torch.float32, device=logits.device)
        # Still try to mask illegal moves if we can map them
        try:
            from .move_index import move_to_index
            legal_moves = list(board.legal_moves)
            for mv in legal_moves:
                idx = move_to_index(board, mv)
                if idx is not None and 0 <= idx < policy_size:
                    legal[idx] = 1.0
            # Zero out indices beyond our move space
            if policy_size > POLICY_SIZE:
                legal[POLICY_SIZE:] = 0.0
        except Exception:
            # If mapping fails, trust the model to output valid probabilities
            pass
    
    masked = mask_logits(logits, legal)
    probs = probs_from_logits(masked, temperature=temperature)

    if sample:
        idx = int(torch.multinomial(probs, num_samples=1)[0].item())
    else:
        idx = int(torch.argmax(probs).item())

    # Map index to move - handle both standard and extended policy sizes
    mv = None
    if 0 <= idx < POLICY_SIZE:
        mv = index_to_move(board, idx)
    
    # Fallback: if move is invalid or index is out of range, find best legal move
    if mv is None or mv not in board.legal_moves:
        order = torch.argsort(probs, descending=True).tolist()
        mv = None
        for i in order:
            if 0 <= i < POLICY_SIZE:
                mv_try = index_to_move(board, int(i))
                if mv_try is not None and mv_try in board.legal_moves:
                    mv = mv_try
                    break
        # If still no move found, use first legal move as fallback
        if mv is None:
            legal_moves = list(board.legal_moves)
            if legal_moves:
                mv = legal_moves[0]
    v_out = 0.0
    if v_pred is not None:
        v = v_pred
        if v.dim() == 2 and v.size(0) == 1:
            v = v[0]
        v_out = float(v.squeeze(-1).detach().cpu().item())
    return mv, probs.detach().cpu(), v_out


@torch.no_grad()
def pick_move(
    model,
    board: chess.Board,
    device: Optional[str] = None,
    temperature: float = 1.0,
    sample: bool = False,
    topk: Optional[int] = None,
) -> Optional[chess.Move]:
    mv, probs, _ = choose_move(board, model, device=device, temperature=temperature, sample=sample)
    return mv
