from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import time
import warnings
import gc

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from .model import TinyPolicyResNet, PolicyOnlyResNet, make_model
from .move_index import POLICY_SIZE


def mask_logits(logits: torch.Tensor, mask: torch.Tensor, illegal_value: float = -1e9) -> torch.Tensor:
    """Set illegal moves to a large negative value before loss.
    
    Uses -1e9 by default for float32, which is more negative and better for masking.
    For float16 (AMP), values are clamped to -1e4 to avoid overflow.
    """
    # For float16, clamp to -1e4 to avoid overflow (float16 max negative ~-65504)
    # For float32, -1e9 is safe and more negative, which is better for masking
    if logits.dtype == torch.float16:
        safe_illegal_value = max(illegal_value, -1e4)
    else:
        safe_illegal_value = illegal_value
    return torch.where(mask > 0.5, logits, torch.full_like(logits, safe_illegal_value))


def smoothed_targets(targets: torch.Tensor, masks: torch.Tensor, epsilon: float = 0.0, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Create label-smoothed distributions over legal moves.
    targets: [B] long indices
    masks: [B, 4672] float 0/1
    dtype: Optional dtype for output tensor (defaults to masks.dtype to match AMP precision)
    returns: [B, 4672] float distribution
    """
    bsz = targets.shape[0]
    # Use masks.dtype to match logits dtype (important for AMP/float16)
    output_dtype = dtype if dtype is not None else masks.dtype
    dist = torch.zeros((bsz, POLICY_SIZE), dtype=output_dtype, device=targets.device)
    if epsilon <= 0.0:
        dist.scatter_(1, targets.view(-1, 1), 1.0)
        return dist
    legal_counts = masks.sum(dim=-1, keepdim=True).clamp_min(1.0)
    uniform = masks / legal_counts
    dist = epsilon * uniform
    dist.scatter_(1, targets.view(-1, 1), (1.0 - epsilon) + epsilon * 0.0)
    return dist


@dataclass
class TrainConfig:
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # Policy regularization
    label_smoothing: float = 0.0
    amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grad_clip: Optional[float] = 1.0
    # LR schedule
    warmup_steps: int = 1000
    total_steps: Optional[int] = None  # if None, auto from loader length * epochs when available
    # Value head training
    value_loss_weight: float = 1.0
    # "mse" (default) or "smooth_l1"
    value_loss_type: str = "mse"
    # Beta parameter for SmoothL1/Huber loss
    value_smooth_l1_beta: float = 1.0


def _make_warmup_cosine(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return max(1e-8, float(step + 1) / float(max(1, warmup_steps)))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _accuracy_top1(masked_logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = masked_logits.argmax(dim=-1)
    correct = (preds == targets).float().mean().item()
    return correct


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    cfg: TrainConfig,
    train: bool = True,
) -> Tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_pol = 0.0
    running_val = 0.0
    running_acc = 0.0
    n_batches = 0

    print(f"Starting {'training' if train else 'validation'} epoch...", flush=True)
    print(f"Loading first batch...", flush=True)
    pbar = tqdm(loader, desc=("train" if train else "val"))
    for batch_idx, batch in enumerate(pbar):
        if batch_idx == 0:
            print(f"✓ First batch loaded, processing...", flush=True)
        # Periodic GPU cache clearing to prevent fragmentation (every 100 batches)
        if train and batch_idx > 0 and batch_idx % 100 == 0:
            if cfg.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Support both 4-tuple (x, y, mask, value) and 3-tuple (x, y, value) batches.
        # For 3-tuples from CachedPolicyValueDataset, we set masks=None and train
        # on raw logits over all POLICY_SIZE moves (labels are always legal).
        masks = None
        yv = None
        if isinstance(batch, (list, tuple)):
            if len(batch) == 4:
                xs, ys, masks, yv = batch
            elif len(batch) == 3:
                xs, ys, yv = batch
                masks = None
            else:
                raise ValueError(f"Unexpected batch size {len(batch)}; expected 3 or 4 elements.")
        else:
            raise ValueError("Batch must be a tuple or list.")

        xs = xs.to(cfg.device, non_blocking=True)
        ys = ys.to(cfg.device, non_blocking=True)
        if masks is not None:
            masks = masks.to(cfg.device, non_blocking=True)
        if yv is not None and isinstance(yv, torch.Tensor):
            yv = yv.to(cfg.device, non_blocking=True)
            # Clamp value targets to [-1,1]
            yv = torch.clamp(yv, -1.0, 1.0)

        # Ensure feature planes match the model's expected input channels.
        # This allows us to append new encoding planes without breaking
        # older checkpoints that were trained with fewer channels.
        try:
            expected_c = int(getattr(model, "in_channels", xs.shape[1]))
        except Exception:
            expected_c = xs.shape[1]
        if xs.shape[1] > expected_c:
            xs = xs[:, :expected_c]
        elif xs.shape[1] < expected_c:
            pad_c = expected_c - xs.shape[1]
            pad = torch.zeros(xs.shape[0], pad_c, xs.shape[2], xs.shape[3], device=xs.device, dtype=xs.dtype)
            xs = torch.cat([xs, pad], dim=1)

        # Safety: if any mask rows are all zeros, replace with ones (no masking).
        # Only applies when masks is provided (PGN/old-cache paths).
        if masks is not None:
            try:
                bad = (masks.sum(dim=1) <= 0)
                if bool(bad.any()):
                    n_bad = int(bad.sum().item())
                    n_total = masks.shape[0]
                    warnings.warn(
                        f"Found {n_bad}/{n_total} samples with all-zero legal move masks. "
                        f"This indicates a data issue (no legal moves is impossible in chess). "
                        f"Replacing with all-ones masks to avoid crash, but this may hide upstream bugs.",
                        RuntimeWarning,
                        stacklevel=2
                    )
                    masks = masks.clone()
                    masks[bad] = 1.0
            except Exception:
                pass

        if train:
            opt.zero_grad(set_to_none=True)
        context = torch.autocast(device_type=("cuda" if cfg.device.startswith("cuda") else ("mps" if cfg.device.startswith("mps") else "cpu")), enabled=cfg.amp)
        with context:
            out = model(xs)
            if isinstance(out, tuple):
                logits, v_pred = out
            else:
                logits, v_pred = out, None
            # CRITICAL: Mask illegal moves BEFORE computing loss.
            # If illegal moves aren't masked, the network is punished for putting mass
            # on thousands of impossible moves, which cripples training.
            if masks is not None:
                masked = mask_logits(logits, masks)  # Sets illegal moves to -1e9 (or -1e4 for float16)
            else:
                masked = logits

            if cfg.label_smoothing > 0:
                if masks is None:
                    # Treat all moves as "legal" for smoothing when no mask is provided.
                    masks_for_smooth = torch.ones(
                        (xs.shape[0], POLICY_SIZE),
                        dtype=logits.dtype,
                        device=logits.device,
                    )
                else:
                    masks_for_smooth = masks
                target_dist = smoothed_targets(ys, masks_for_smooth, epsilon=cfg.label_smoothing)
                log_probs = F.log_softmax(masked, dim=-1)
                policy_loss = -(target_dist * log_probs).sum(dim=-1).mean()
            else:
                policy_loss = F.cross_entropy(masked, ys)

            value_loss = torch.tensor(0.0, device=xs.device, dtype=policy_loss.dtype)
            if v_pred is not None and yv is not None:
                # Clamp targets defensively in case upstream data drifts
                yv_clamped = torch.clamp(yv, -1.0, 1.0)
                if cfg.value_loss_type.lower() == "smooth_l1":
                    value_loss = F.smooth_l1_loss(
                        v_pred.squeeze(-1),
                        yv_clamped,
                        beta=getattr(cfg, "value_smooth_l1_beta", 1.0),
                    )
                else:
                    # Default: plain MSE
                    value_loss = F.mse_loss(v_pred.squeeze(-1), yv_clamped)

            loss = policy_loss + cfg.value_loss_weight * value_loss

        if train:
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if cfg.grad_clip is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
            if scheduler is not None:
                scheduler.step()

        acc = _accuracy_top1(masked.detach(), ys)
        
        # Measure GPU memory BEFORE deleting tensors (shows actual usage during training)
        if cfg.device.startswith("cuda") and torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            mem_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        else:
            mem_allocated = mem_reserved = 0.0
        
        # Extract scalars and explicitly delete intermediate tensors to avoid memory leaks
        loss_val = float(loss.detach().cpu().item())
        pol_val = float(policy_loss.detach().cpu().item())
        val_val = float(value_loss.detach().cpu().item())

        # For early batches, log raw loss magnitudes to help balance
        # policy vs value loss scales (see value_loss_weight).
        if train and batch_idx < 3:
            ratio = pol_val / max(val_val, 1e-8) if val_val > 0 else float("inf")
            print(
                f"[batch {batch_idx}] policy_loss={pol_val:.4f}, "
                f"value_loss={val_val:.4f}, "
                f"policy/value≈{ratio:.2f}",
                flush=True,
            )
        
        # Delete intermediate tensors to free memory immediately
        del loss, policy_loss, value_loss, masked, logits
        if v_pred is not None:
            del v_pred
        
        running_loss += loss_val
        running_pol += pol_val
        running_val += val_val
        running_acc += acc
        n_batches += 1
        
        postfix = {
            "loss": f"{running_loss / n_batches:.4f}",
            "pl": f"{running_pol / n_batches:.4f}",
            "vl": f"{running_val / n_batches:.4f}",
            "acc": f"{running_acc / n_batches:.3f}",
        }
        
        # Add GPU memory info (measured before deletion, shows actual usage)
        if cfg.device.startswith("cuda") and torch.cuda.is_available():
            postfix["gpu_mem"] = f"{mem_allocated:.1f}/{mem_reserved:.1f}GB"
        
        pbar.set_postfix(postfix)

    return running_loss / max(1, n_batches), running_acc / max(1, n_batches)


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    cfg: TrainConfig,
    val_loader: Optional[DataLoader] = None,
    ckpt_path: Optional[str] = None,
) -> Dict[str, float]:
    model.to(cfg.device)
    
    # Verify model is on correct device
    if cfg.device.startswith("cuda"):
        first_param_device = next(model.parameters()).device
        if first_param_device.type != "cuda":
            warnings.warn(f"Model parameters are on {first_param_device}, but cfg.device is {cfg.device}. This may indicate a problem.", RuntimeWarning)
    
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Use new GradScaler API if available, otherwise fall back to old API
    # New API: torch.amp.GradScaler(device_type, enabled=...)
    # Old API: torch.cuda.amp.GradScaler(enabled=...) (deprecated but still works)
    if cfg.amp and (cfg.device.startswith("cuda") or cfg.device.startswith("mps")):
        try:
            # Try new API first (PyTorch 2.0+)
            device_type = "cuda" if cfg.device.startswith("cuda") else "mps"
            scaler = torch.amp.GradScaler(device_type, enabled=True)
        except (TypeError, AttributeError):
            # Fall back to old API for older PyTorch versions
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        try:
            scaler = torch.amp.GradScaler("cpu", enabled=False)
        except (TypeError, AttributeError):
            # For CPU, old API doesn't have a direct equivalent, so disable scaler
            scaler = torch.cuda.amp.GradScaler(enabled=False)

    # Scheduler
    # Note: total_steps is calculated once at the start of training.
    # If train_loader length changes between runs (dynamic datasets, varying cache sizes, etc.),
    # the cosine schedule may be miscalibrated. Consider setting cfg.total_steps explicitly
    # if you need precise LR scheduling across different dataset sizes.
    steps_per_epoch = None
    try:
        steps_per_epoch = len(train_loader)
    except Exception:
        steps_per_epoch = None
    total_steps = cfg.total_steps
    if total_steps is None:
        if steps_per_epoch is not None:
            total_steps = cfg.epochs * max(1, steps_per_epoch)
        else:
            total_steps = cfg.epochs * 10000  # fallback
    scheduler = _make_warmup_cosine(opt, cfg.warmup_steps, total_steps)

    best_val = float('inf')
    last_train = {"loss": 0.0, "acc": 0.0}
    last_val = {"loss": 0.0, "acc": 0.0}

    # Overall training progress bar with ETA
    epoch_times = []
    overall_pbar = tqdm(
        range(cfg.epochs),
        desc="Training",
        unit="epoch",
        position=0,
        leave=True,
    )
    
    training_start_time = time.time()
    
    for epoch_idx, epoch in enumerate(overall_pbar):
        epoch_start_time = time.time()
        
        # Force garbage collection at the start of each epoch to free accumulated memory
        gc.collect()
        if cfg.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        train_loss, train_acc = _run_epoch(model, train_loader, opt, scaler, scheduler, cfg, train=True)
        last_train = {"loss": train_loss, "acc": train_acc}

        if val_loader is not None:
            with torch.no_grad():
                val_loss, val_acc = _run_epoch(model, val_loader, opt, scaler, None, cfg, train=False)
            last_val = {"loss": val_loss, "acc": val_acc}
            if val_loss < best_val and ckpt_path:
                meta = {
                    "arch": model.__class__.__name__,
                    "in_channels": getattr(model, "in_channels", None),
                    "width": getattr(model, "width", None),
                    "n_blocks": getattr(model, "n_blocks", None),
                }
                torch.save({"model": model.state_dict(), "meta": meta}, ckpt_path)
                best_val = val_loss
        else:
            # No validation: save checkpoint after each epoch to avoid losing progress on interruption
            if ckpt_path:
                meta = {
                    "arch": model.__class__.__name__,
                    "in_channels": getattr(model, "in_channels", None),
                    "width": getattr(model, "width", None),
                    "n_blocks": getattr(model, "n_blocks", None),
                }
                torch.save({"model": model.state_dict(), "meta": meta}, ckpt_path)
        
        # Calculate epoch time and ETA
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Calculate average epoch time and ETA
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = cfg.epochs - (epoch_idx + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        
        # Format time strings
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            if hours > 0:
                return f"{hours}h {minutes}m {secs}s"
            elif minutes > 0:
                return f"{minutes}m {secs}s"
            else:
                return f"{secs}s"
        
        # Update overall progress bar
        elapsed_time = time.time() - training_start_time
        postfix_dict = {
            "train_loss": f"{train_loss:.4f}",
            "train_acc": f"{train_acc:.3f}",
        }
        if val_loader is not None:
            postfix_dict.update({
                "val_loss": f"{val_loss:.4f}",
                "val_acc": f"{val_acc:.3f}",
                "best_val": f"{best_val:.4f}",
            })
        postfix_dict.update({
            "epoch_time": format_time(epoch_time),
            "eta": format_time(eta_seconds),
        })
        overall_pbar.set_postfix(postfix_dict)

    overall_pbar.close()

    # Final checkpoint already saved during training loop (after each epoch if no val, or on best val if val exists)

    out = {"train_loss": last_train["loss"], "train_acc": last_train["acc"]}
    if val_loader is not None:
        out.update({"val_loss": last_val["loss"], "val_acc": last_val["acc"]})
        # Print training summary
        print(f"\nTraining Summary:", flush=True)
        print(f"  Best validation loss: {best_val:.4f}", flush=True)
        print(f"  Final train loss: {last_train['loss']:.4f}, acc: {last_train['acc']:.3f}", flush=True)
        print(f"  Final val loss: {last_val['loss']:.4f}, acc: {last_val['acc']:.3f}", flush=True)
        if last_val['loss'] > last_train['loss']:
            print(f"  ⚠️  Warning: Validation loss ({last_val['loss']:.4f}) > Train loss ({last_train['loss']:.4f})", flush=True)
            print(f"     This suggests overfitting. Consider:", flush=True)
            print(f"     - Reducing epochs", flush=True)
            print(f"     - Increasing regularization (weight_decay, label_smoothing)", flush=True)
            print(f"     - Using a smaller model", flush=True)
    return out


def save_checkpoint(model: nn.Module, path: str) -> None:
    meta = {
        "arch": model.__class__.__name__,
        "in_channels": getattr(model, "in_channels", None),
        "width": getattr(model, "width", None),
        "n_blocks": getattr(model, "n_blocks", None),
    }
    torch.save({"model": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path: str, map_location: Optional[str] = None) -> nn.Module:
    state = torch.load(path, map_location=map_location or ("cuda" if torch.cuda.is_available() else "cpu"))
    meta = state.get("meta", {})
    arch = meta.get("arch", None)
    state_dict = state["model"]
    
    # Strip _orig_mod. prefix if present (from torch.compile or similar wrappers)
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        print("Stripped _orig_mod. prefix from checkpoint keys")
    
    # Map tower -> trunk if present (some checkpoints use "tower" instead of "trunk")
    if any("tower" in k for k in state_dict.keys()) and not any("trunk" in k for k in state_dict.keys()):
        state_dict = {k.replace("tower", "trunk"): v for k, v in state_dict.items()}
        print("Mapped 'tower' -> 'trunk' in checkpoint keys")
    
    # Auto-detect architecture if not in metadata
    if arch is None:
        # Check for value head components to identify PolicyValueResNet
        has_value_head = any("value_conv" in k or "value_fc" in k for k in state_dict.keys())
        # Check for trunk structure (after mapping)
        has_trunk = any("trunk" in k for k in state_dict.keys())
        # Check policy head structure
        has_simple_policy = "policy_head.weight" in state_dict and "policy_head.bias" in state_dict
        has_sequential_policy = any("policy_head.0" in k or "policy_head.1" in k for k in state_dict.keys())
        
        if has_value_head and has_trunk:
            arch = "PolicyValueResNet"
            print("Auto-detected architecture: PolicyValueResNet (has value head and trunk)")
        elif has_simple_policy and not has_trunk:
            arch = "TinyPolicyResNet"
            print("Auto-detected architecture: TinyPolicyResNet (simple policy head, no trunk)")
        elif has_sequential_policy and has_trunk:
            arch = "PolicyValueResNet"
            print("Auto-detected architecture: PolicyValueResNet (sequential policy head and trunk)")
        else:
            # Default to PolicyValueResNet if we have trunk structure
            if has_trunk:
                arch = "PolicyValueResNet"
                print(f"Auto-detected architecture: PolicyValueResNet (has trunk structure)")
            else:
                arch = "TinyPolicyResNet"
                print(f"Auto-detected architecture: TinyPolicyResNet (default fallback)")
    
    # Handle backward compatibility for old checkpoints with policy_head.4 -> policy_head.3
    if "policy_head.4.weight" in state_dict and "policy_head.3.weight" not in state_dict:
        state_dict = dict(state_dict)
        state_dict["policy_head.3.weight"] = state_dict.pop("policy_head.4.weight")
        state_dict["policy_head.3.bias"] = state_dict.pop("policy_head.4.bias")
        state["model"] = state_dict
    
    # Handle checkpoints with single-layer policy_head (policy_head.weight) vs sequential (policy_head.0.weight)
    # This happens with some PolicyValueResNet checkpoints
    # We'll let non-strict loading handle this mismatch
    
    # Known architectures
    if arch == "PolicyOnlyResNet":
        from .model import PolicyOnlyResNet
        in_ch = meta.get("in_channels", 18)
        width = meta.get("width", 64)
        blocks = meta.get("n_blocks", 8)
        model = PolicyOnlyResNet(in_channels=in_ch, width=width, n_blocks=blocks)
    elif arch == "TinyPolicyResNet":
        in_ch = meta.get("in_channels", 18)
        channels = meta.get("width", 64)
        blocks = meta.get("n_blocks", 4)
        model = TinyPolicyResNet(in_channels=in_ch, channels=channels, blocks=blocks)
    elif arch == "PolicyValueResNet":
        from .model import PolicyValueResNet
        in_ch = meta.get("in_channels", None)
        width = meta.get("width", None)
        blocks = meta.get("n_blocks", None)
        
        # Infer width and in_channels from checkpoint keys if not in metadata
        if width is None or in_ch is None:
            # Check stem.0.weight shape: [width, in_channels, 3, 3]
            if "stem.0.weight" in state_dict:
                stem_shape = state_dict["stem.0.weight"].shape
                if len(stem_shape) >= 2:
                    if width is None:
                        width = int(stem_shape[0])
                    if in_ch is None:
                        in_ch = int(stem_shape[1])
                    print(f"Inferred width={width}, in_channels={in_ch} from stem.0.weight shape {stem_shape}")
            # Or check trunk.0.conv1.weight shape: [width, width, 3, 3]
            elif "trunk.0.conv1.weight" in state_dict:
                trunk_shape = state_dict["trunk.0.conv1.weight"].shape
                if len(trunk_shape) >= 2:
                    if width is None:
                        width = int(trunk_shape[0])
                    if in_ch is None:
                        in_ch = 18  # Default fallback
                        print(f"Could not infer in_channels, using default={in_ch}")
                    print(f"Inferred width={width} from trunk.0.conv1.weight shape {trunk_shape}")
            else:
                if width is None:
                    width = 128  # Default fallback
                    print(f"Could not infer width, using default={width}")
                if in_ch is None:
                    in_ch = 18  # Default fallback
                    print(f"Could not infer in_channels, using default={in_ch}")
        
        # Infer number of blocks from checkpoint keys if not in metadata
        if blocks is None:
            import re
            max_block = -1
            for k in state_dict.keys():
                # Match patterns like "trunk.19"
                match = re.search(r'trunk\.(\d+)', k)
                if match:
                    block_idx = int(match.group(1))
                    max_block = max(max_block, block_idx)
            if max_block >= 0:
                blocks = max_block + 1  # Blocks are 0-indexed
                print(f"Inferred n_blocks={blocks} from checkpoint keys")
            else:
                blocks = 20  # Default fallback
                print(f"Could not infer n_blocks, using default={blocks}")
        
        model = PolicyValueResNet(in_channels=in_ch, width=width, n_blocks=blocks)
    else:
        # Unknown architecture - raise error instead of silently falling back
        # This prevents loading mismatched architectures if classes are renamed
        raise ValueError(
            f"Unknown model architecture '{arch}' in checkpoint. "
            f"Supported architectures: PolicyOnlyResNet, TinyPolicyResNet, PolicyValueResNet. "
            f"If you renamed a model class, update the checkpoint or this loader."
        )
    
    # Try strict loading first, fall back to non-strict if needed
    try:
        incompatible = model.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys:
            print(f"Warning: Missing keys in checkpoint: {len(incompatible.missing_keys)} keys")
            if len(incompatible.missing_keys) > 10:
                print(f"  First 10: {incompatible.missing_keys[:10]}")
        if incompatible.unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {len(incompatible.unexpected_keys)} keys")
            if len(incompatible.unexpected_keys) > 10:
                print(f"  First 10: {incompatible.unexpected_keys[:10]}")
    except RuntimeError as e:
        # If strict loading fails, try non-strict
        print(f"Strict loading failed: {e}")
        print("Attempting non-strict loading...")
        try:
            incompatible = model.load_state_dict(state_dict, strict=False)
            if incompatible.missing_keys:
                print(f"Warning: Missing keys (non-strict): {len(incompatible.missing_keys)} keys")
                if len(incompatible.missing_keys) > 10:
                    print(f"  First 10: {incompatible.missing_keys[:10]}")
            if incompatible.unexpected_keys:
                print(f"Warning: Unexpected keys (non-strict): {len(incompatible.unexpected_keys)} keys")
                if len(incompatible.unexpected_keys) > 10:
                    print(f"  First 10: {incompatible.unexpected_keys[:10]}")
        except RuntimeError as e2:
            # Even non-strict can fail on size mismatches - skip those keys
            print(f"Non-strict loading also failed: {e2}")
            print("Attempting to load compatible keys only...")
            model_dict = model.state_dict()
            compatible_dict = {}
            skipped = []
            for k, v in state_dict.items():
                if k in model_dict:
                    if model_dict[k].shape == v.shape:
                        compatible_dict[k] = v
                    else:
                        skipped.append(f"{k}: checkpoint {v.shape} vs model {model_dict[k].shape}")
                else:
                    skipped.append(f"{k}: not in model")
            model.load_state_dict(compatible_dict, strict=False)
            print(f"Loaded {len(compatible_dict)} compatible keys, skipped {len(skipped)} incompatible keys")
            if len(skipped) > 0 and len(skipped) <= 20:
                print(f"Skipped keys: {skipped}")
    
    return model
